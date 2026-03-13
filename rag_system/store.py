from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import chromadb


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.flush()
        Path(handle.name).replace(path)


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(path: Path) -> dict[str, Any]:
    return load_json(path, {"files": {}})


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _atomic_json_write(path, manifest)


def load_chunk_catalog(path: Path) -> dict[str, Any]:
    return load_json(path, {"chunks": {}})


def save_chunk_catalog(path: Path, catalog: dict[str, Any]) -> None:
    _atomic_json_write(path, catalog)


class ChromaVectorStore:
    def __init__(self, chroma_dir: Path, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]], batch_size: int = 128) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count must match embedding count.")

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            self.collection.add(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[chunk.metadata for chunk in batch_chunks],
                embeddings=batch_embeddings,
            )

    def delete_ids(self, ids: list[str], batch_size: int = 256) -> None:
        if not ids:
            return
        for start in range(0, len(ids), batch_size):
            self.collection.delete(ids=ids[start : start + batch_size])

    def query(self, query_embedding: list[float], n_results: int) -> list[VectorHit]:
        if self.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        hits: list[VectorHit] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = max(0.0, 1.0 - float(distance))
            hits.append(
                VectorHit(
                    chunk_id=chunk_id,
                    text=document,
                    metadata=metadata,
                    score=similarity,
                )
            )
        return hits
