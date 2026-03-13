from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .rerank import FastEmbedReranker, RerankInput
from .store import ChromaVectorStore, VectorHit


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    base_score: float
    vector_score: float
    keyword_score: float
    rerank_score: float


class BM25Index:
    def __init__(self, catalog: dict[str, Any]) -> None:
        self.catalog = catalog
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.doc_frequencies: dict[str, int] = defaultdict(int)
        self.avg_doc_length = 0.0
        self.total_docs = 0

        total_length = 0
        for chunk_id, payload in catalog.get("chunks", {}).items():
            tokens = tokenize(payload["text"])
            self.doc_tokens[chunk_id] = tokens
            self.doc_lengths[chunk_id] = len(tokens)
            total_length += len(tokens)
            self.total_docs += 1
            for token in set(tokens):
                self.doc_frequencies[token] += 1

        if self.total_docs:
            self.avg_doc_length = total_length / self.total_docs

    def score(self, query: str, k: int, b: float = 0.75, k1: float = 1.5) -> list[tuple[str, float]]:
        if not self.total_docs:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: dict[str, float] = defaultdict(float)
        for chunk_id, tokens in self.doc_tokens.items():
            if not tokens:
                continue
            term_counts = Counter(tokens)
            doc_length = self.doc_lengths[chunk_id]
            for term in query_tokens:
                if term not in term_counts:
                    continue
                doc_freq = self.doc_frequencies.get(term, 0)
                idf = math.log(1 + (self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                frequency = term_counts[term]
                norm = frequency + k1 * (1 - b + b * (doc_length / max(self.avg_doc_length, 1.0)))
                scores[chunk_id] += idf * ((frequency * (k1 + 1)) / max(norm, 1e-9))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:k]


class HybridRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        catalog: dict[str, Any],
        vector_candidates: int,
        keyword_candidates: int,
        rerank_candidates: int,
        final_chunks: int,
        reranker: FastEmbedReranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.catalog = catalog
        self.vector_candidates = vector_candidates
        self.keyword_candidates = keyword_candidates
        self.rerank_candidates = max(final_chunks, rerank_candidates)
        self.final_chunks = final_chunks
        self.reranker = reranker
        self.keyword_index = BM25Index(catalog)
        self.neighbors = self._build_neighbor_lookup()

    def _build_neighbor_lookup(self) -> dict[tuple[str, int], str]:
        lookup: dict[tuple[str, int], str] = {}
        for chunk_id, payload in self.catalog.get("chunks", {}).items():
            metadata = payload["metadata"]
            lookup[(metadata["relative_path"], int(metadata["chunk_index"]))] = chunk_id
        return lookup

    def reload(self, catalog: dict[str, Any]) -> None:
        self.catalog = catalog
        self.keyword_index = BM25Index(catalog)
        self.neighbors = self._build_neighbor_lookup()

    def search(self, query: str, query_embedding: list[float]) -> list[RetrievalHit]:
        if not self.catalog.get("chunks"):
            return []

        vector_hits = self.vector_store.query(query_embedding, self.vector_candidates)
        keyword_hits = self.keyword_index.score(query, self.keyword_candidates)

        rrf_scores: dict[str, float] = defaultdict(float)
        vector_by_id: dict[str, VectorHit] = {hit.chunk_id: hit for hit in vector_hits}
        keyword_by_id = dict(keyword_hits)

        for rank, hit in enumerate(vector_hits, start=1):
            rrf_scores[hit.chunk_id] += 1.4 / (50 + rank)
        for rank, (chunk_id, _) in enumerate(keyword_hits, start=1):
            rrf_scores[chunk_id] += 1.0 / (50 + rank)

        ordered_ids = [
            chunk_id for chunk_id, _ in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        candidate_ids = self._expand_neighbors(
            ordered_ids[: self.rerank_candidates],
            limit=max(self.rerank_candidates * 2, self.final_chunks),
        )
        rerank_scores = self._rerank(query, candidate_ids)
        if rerank_scores:
            ordered_ids = [
                chunk_id
                for chunk_id in sorted(
                    candidate_ids,
                    key=lambda item: (rerank_scores.get(item, float("-inf")), rrf_scores.get(item, 0.0)),
                    reverse=True,
                )
            ]
        else:
            ordered_ids = candidate_ids

        results: list[RetrievalHit] = []
        for chunk_id in ordered_ids:
            payload = self.catalog["chunks"].get(chunk_id)
            if not payload:
                continue
            results.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    text=payload["text"],
                    metadata=payload["metadata"],
                    score=rerank_scores.get(chunk_id, rrf_scores.get(chunk_id, 0.0)),
                    base_score=rrf_scores.get(chunk_id, 0.0),
                    vector_score=vector_by_id.get(chunk_id).score if chunk_id in vector_by_id else 0.0,
                    keyword_score=keyword_by_id.get(chunk_id, 0.0),
                    rerank_score=rerank_scores.get(chunk_id, 0.0),
                )
            )
            if len(results) >= self.final_chunks:
                break
        return results

    def _rerank(self, query: str, candidate_ids: list[str]) -> dict[str, float]:
        if self.reranker is None:
            return {}

        candidates: list[RerankInput] = []
        for chunk_id in candidate_ids:
            payload = self.catalog.get("chunks", {}).get(chunk_id)
            if not payload:
                continue
            metadata = payload["metadata"]
            enriched_text = (
                f"File: {metadata.get('file_name', '')}\n"
                f"Path: {metadata.get('relative_path', '')}\n"
                f"Page: {metadata.get('page_number', '')}\n\n"
                f"{payload['text']}"
            )
            candidates.append(RerankInput(chunk_id=chunk_id, text=enriched_text))
        return self.reranker.rerank(query, candidates)

    def _expand_neighbors(self, ordered_ids: list[str], limit: int) -> list[str]:
        expanded: list[str] = []
        seen: set[str] = set()

        for chunk_id in ordered_ids:
            if chunk_id in seen:
                continue
            expanded.append(chunk_id)
            seen.add(chunk_id)

            payload = self.catalog["chunks"].get(chunk_id)
            if not payload:
                continue

            metadata = payload["metadata"]
            relative_path = metadata["relative_path"]
            chunk_index = int(metadata["chunk_index"])

            for offset in (-1, 1):
                neighbor_id = self.neighbors.get((relative_path, chunk_index + offset))
                if neighbor_id and neighbor_id not in seen:
                    expanded.append(neighbor_id)
                    seen.add(neighbor_id)
                    if len(expanded) >= limit:
                        return expanded
        return expanded
