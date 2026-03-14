from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from .config import Settings
from .ocr import RapidOCRProvider
from .pdf_ingest import PDFChunk, compute_file_sha256, extract_pdf_chunks
from .rerank import FastEmbedReranker
from .retrieval import HybridRetriever, RetrievalHit
from .store import (
    ChromaVectorStore,
    ChunkRecord,
    load_chunk_catalog,
    load_manifest,
    save_chunk_catalog,
    save_manifest,
)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return value or "document"


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


@dataclass(frozen=True)
class AnswerPlan:
    question: str
    messages: list[dict[str, str]]
    hits: list[RetrievalHit]
    fallback_answer: str | None = None


class RAGSystem:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.openai: OpenAI | None = None
        self.ocr_provider = RapidOCRProvider() if settings.ocr_enabled else None
        self.reranker = (
            FastEmbedReranker(
                model_name=settings.rerank_model_name,
                cache_dir=settings.model_cache_dir,
                max_chars=settings.rerank_max_chars,
            )
            if settings.rerank_enabled
            else None
        )
        self.vector_store = ChromaVectorStore(settings.chroma_dir, settings.collection_name)
        self.manifest = load_manifest(settings.manifest_path)
        self.catalog = load_chunk_catalog(settings.chunk_catalog_path)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            catalog=self.catalog,
            vector_candidates=settings.vector_candidates,
            keyword_candidates=settings.keyword_candidates,
            rerank_candidates=settings.rerank_candidates,
            final_chunks=settings.final_chunks,
            reranker=self.reranker,
        )

    def refresh_state(self) -> None:
        self.manifest = load_manifest(self.settings.manifest_path)
        self.catalog = load_chunk_catalog(self.settings.chunk_catalog_path)
        self.retriever.reload(self.catalog)

    def require_openai_key(self) -> None:
        if not self.settings.has_openai_key:
            raise RuntimeError("OPENAI_API_KEY is not configured. Copy .env.example to .env and set your key.")

    def get_openai_client(self) -> OpenAI:
        if self.openai is None:
            self.require_openai_key()
            self.openai = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                timeout=90.0,
                max_retries=2,
            )
        return self.openai

    def list_pdf_files(self) -> list[Path]:
        return sorted(self.settings.pdf_dir.rglob("*.pdf"))

    def get_status(self) -> dict[str, Any]:
        self.refresh_state()
        return {
            "api_key_configured": self.settings.has_openai_key,
            "pdf_count": len(self.list_pdf_files()),
            "indexed_file_count": len(self.manifest.get("files", {})),
            "chunk_count": len(self.catalog.get("chunks", {})),
            "collection_count": self.vector_store.count(),
            "reranking_enabled": self.settings.rerank_enabled,
            "ocr_enabled": self.settings.ocr_enabled,
            "pdf_dir": str(self.settings.pdf_dir),
            "files": sorted(self.manifest.get("files", {}).keys()),
        }

    def sync_documents(self, force: bool = False) -> dict[str, Any]:
        self.require_openai_key()

        manifest = load_manifest(self.settings.manifest_path)
        catalog = load_chunk_catalog(self.settings.chunk_catalog_path)
        current_files = self.list_pdf_files()
        current_relative = {str(path.relative_to(self.settings.pdf_dir)): path for path in current_files}

        removed: list[str] = []
        added: list[str] = []
        updated: list[str] = []
        skipped: list[str] = []

        for relative_path in list(manifest.get("files", {}).keys()):
            if relative_path in current_relative:
                continue
            entry = manifest["files"].pop(relative_path)
            old_ids = entry.get("chunk_ids", [])
            self.vector_store.delete_ids(old_ids)
            for chunk_id in old_ids:
                catalog.get("chunks", {}).pop(chunk_id, None)
            removed.append(relative_path)

        for relative_path, path in current_relative.items():
            file_hash = compute_file_sha256(path)
            previous = manifest.get("files", {}).get(relative_path)

            if previous and previous.get("sha256") == file_hash and not force:
                skipped.append(relative_path)
                continue

            if previous:
                old_ids = previous.get("chunk_ids", [])
                self.vector_store.delete_ids(old_ids)
                for chunk_id in old_ids:
                    catalog.get("chunks", {}).pop(chunk_id, None)

            extraction_result = extract_pdf_chunks(
                path,
                chunk_size_words=self.settings.chunk_size_words,
                overlap_words=self.settings.chunk_overlap_words,
                ocr_provider=self.ocr_provider,
                ocr_enabled=self.settings.ocr_enabled,
                ocr_min_text_chars=self.settings.ocr_min_text_chars,
                ocr_image_area_threshold=self.settings.ocr_image_area_threshold,
                ocr_render_dpi=self.settings.ocr_render_dpi,
            )
            extracted_chunks = extraction_result.chunks
            page_count = extraction_result.page_count

            if not extracted_chunks:
                manifest.setdefault("files", {})[relative_path] = {
                    "sha256": file_hash,
                    "chunk_ids": [],
                    "page_count": page_count,
                    "native_pages": extraction_result.native_pages,
                    "ocr_pages": extraction_result.ocr_pages,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
                if previous:
                    updated.append(relative_path)
                else:
                    added.append(relative_path)
                continue

            chunk_records = self._build_chunk_records(relative_path, file_hash, extracted_chunks)
            embeddings = self._embed_texts([chunk.text for chunk in chunk_records])
            self.vector_store.add_chunks(chunk_records, embeddings)

            for chunk in chunk_records:
                catalog.setdefault("chunks", {})[chunk.chunk_id] = {
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }

            manifest.setdefault("files", {})[relative_path] = {
                "sha256": file_hash,
                "chunk_ids": [chunk.chunk_id for chunk in chunk_records],
                "page_count": page_count,
                "native_pages": extraction_result.native_pages,
                "ocr_pages": extraction_result.ocr_pages,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }

            if previous:
                updated.append(relative_path)
            else:
                added.append(relative_path)

        save_manifest(self.settings.manifest_path, manifest)
        save_chunk_catalog(self.settings.chunk_catalog_path, catalog)
        self.refresh_state()

        return {
            "added": added,
            "updated": updated,
            "removed": removed,
            "skipped": skipped,
            "pdf_count": len(current_files),
            "chunk_count": len(self.catalog.get("chunks", {})),
            "collection_count": self.vector_store.count(),
        }

    def _build_chunk_records(
        self,
        relative_path: str,
        file_hash: str,
        extracted_chunks: list[PDFChunk],
    ) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        source_slug = slugify(relative_path)

        for index, chunk in enumerate(extracted_chunks):
            chunk_id = f"{source_slug}-{file_hash[:12]}-{index:05d}"
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    metadata={
                        "relative_path": relative_path,
                        "file_name": Path(relative_path).name,
                        "sha256": file_hash,
                        "page_number": int(chunk.page_number),
                        "chunk_index": index,
                        "word_count": int(chunk.word_count),
                        "extraction_method": chunk.extraction_method,
                    },
                )
            )
        return records

    def answer_question(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> tuple[str, list[RetrievalHit]]:
        plan = self.prepare_answer(question, chat_history)
        if plan.fallback_answer is not None:
            return plan.fallback_answer, plan.hits

        answer = self.generate_answer(plan)
        return answer, plan.hits

    def prepare_answer(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> AnswerPlan:
        self.require_openai_key()
        self.refresh_state()

        if not question.strip():
            return AnswerPlan(
                question=question,
                messages=[],
                hits=[],
                fallback_answer="Ask a question about the indexed PDFs.",
            )
        if not self.catalog.get("chunks"):
            return AnswerPlan(
                question=question,
                messages=[],
                hits=[],
                fallback_answer=(
                    "No indexed PDF content is available yet. Add PDFs to data/pdfs and run ingestion first."
                ),
            )

        retrieval_query = self._build_retrieval_query(question, chat_history or [])
        query_embedding = self._embed_texts([retrieval_query])[0]
        hits = self.retriever.search(retrieval_query, query_embedding)
        if not hits:
            return AnswerPlan(
                question=question,
                messages=[],
                hits=[],
                fallback_answer="I could not find relevant indexed passages for that question.",
            )

        messages = self._build_generation_messages(question, chat_history or [], hits)
        return AnswerPlan(question=question, messages=messages, hits=hits)

    def generate_answer(self, plan: AnswerPlan) -> str:
        if plan.fallback_answer is not None:
            return plan.fallback_answer
        return self._chat_completion(plan.messages)

    def stream_answer(self, plan: AnswerPlan) -> Iterator[str]:
        if plan.fallback_answer is not None:
            yield plan.fallback_answer
            return
        yield from self._chat_completion_stream(plan.messages)

    def _embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        client = self.get_openai_client()
        try:
            for batch in batched(texts, batch_size):
                response = client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=batch,
                    encoding_format="float",
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                embeddings.extend(item.embedding for item in ordered)
        except (APIConnectionError, APITimeoutError) as exc:
            raise RuntimeError(
                "Failed to reach the OpenAI API. Check internet access and whether "
                f"network access is allowed in this environment. Original error: {exc}"
            ) from exc
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc
        return embeddings

    def _chat_completion(self, messages: list[dict[str, str]]) -> str:
        client = self.get_openai_client()
        try:
            response = client.chat.completions.create(
                model=self.settings.chat_model,
                messages=messages,
                temperature=self.settings.temperature,
                max_completion_tokens=self.settings.response_max_tokens,
            )
        except (APIConnectionError, APITimeoutError) as exc:
            raise RuntimeError(
                "Failed to reach the OpenAI API. Check internet access and whether "
                f"network access is allowed in this environment. Original error: {exc}"
            ) from exc
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        content = response.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        return ""

    def _chat_completion_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        client = self.get_openai_client()
        try:
            stream = client.chat.completions.create(
                model=self.settings.chat_model,
                messages=messages,
                temperature=self.settings.temperature,
                max_completion_tokens=self.settings.response_max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if isinstance(delta, str) and delta:
                    yield delta
        except (APIConnectionError, APITimeoutError) as exc:
            raise RuntimeError(
                "Failed to reach the OpenAI API. Check internet access and whether "
                f"network access is allowed in this environment. Original error: {exc}"
            ) from exc
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

    def _build_retrieval_query(self, question: str, history: list[dict[str, str]]) -> str:
        recent_user_turns = [item["content"] for item in history if item.get("role") == "user"][-2:]
        if not recent_user_turns:
            return question
        context = "\n".join(recent_user_turns)
        return f"Conversation context:\n{context}\n\nCurrent question:\n{question}"

    def _build_generation_messages(
        self,
        question: str,
        history: list[dict[str, str]],
        hits: list[RetrievalHit],
    ) -> list[dict[str, str]]:
        context_blocks: list[str] = []
        for index, hit in enumerate(hits[: self.settings.max_context_chunks], start=1):
            context_blocks.append(
                (
                    f"[S{index}] file={hit.metadata['file_name']} "
                    f"page={hit.metadata['page_number']} "
                    f"path={hit.metadata['relative_path']}\n{hit.text}"
                )
            )

        conversation_summary: list[str] = []
        for item in history[-6:]:
            role = item.get("role", "user")
            content = item.get("content", "").strip()
            if content:
                conversation_summary.append(f"{role}: {content}")

        user_prompt = (
            "Use only the retrieved PDF context below to answer the question.\n"
            "If the answer is not supported by the context, say you do not have enough information.\n"
            "Cite supporting sources inline as [S1], [S2], etc.\n\n"
            f"Conversation:\n{chr(10).join(conversation_summary) or 'No prior conversation.'}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{chr(10).join(context_blocks)}"
        )

        return [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-augmented assistant. Answer only from the supplied PDF excerpts. "
                    "Do not invent facts, and keep the answer concise and direct."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    def format_status_markdown(self, sync_summary: dict[str, Any] | None = None) -> str:
        status = self.get_status()
        lines = [
            "## System Status",
            f"- API key configured: {'yes' if status['api_key_configured'] else 'no'}",
            f"- OCR enabled: {'yes' if status['ocr_enabled'] else 'no'}",
            f"- Reranking enabled: {'yes' if status['reranking_enabled'] else 'no'}",
            f"- PDF folder: `{status['pdf_dir']}`",
            f"- PDFs found: {status['pdf_count']}",
            f"- Indexed files: {status['indexed_file_count']}",
            f"- Indexed chunks: {status['chunk_count']}",
            f"- Chroma records: {status['collection_count']}",
        ]

        if sync_summary:
            lines.extend(
                [
                    "",
                    "## Last Sync",
                    f"- Added: {len(sync_summary['added'])}",
                    f"- Updated: {len(sync_summary['updated'])}",
                    f"- Removed: {len(sync_summary['removed'])}",
                    f"- Unchanged: {len(sync_summary['skipped'])}",
                ]
            )

        if status["files"]:
            preview = status["files"][:12]
            lines.extend(["", "## Indexed Files"])
            lines.extend([f"- {path}" for path in preview])
            if len(status["files"]) > len(preview):
                lines.append(f"- ... and {len(status['files']) - len(preview)} more")

        return "\n".join(lines)

    def format_sources_markdown(self, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "## Sources\nNo sources used."

        lines = ["## Sources"]
        for index, hit in enumerate(hits, start=1):
            preview = hit.text[:280].strip()
            if len(hit.text) > 280:
                preview += "..."
            lines.extend(
                [
                    f"### [S{index}] {hit.metadata['file_name']} page {hit.metadata['page_number']}",
                    f"- Path: `{hit.metadata['relative_path']}`",
                    f"- Final score: {hit.score:.4f}",
                    f"- Hybrid score: {hit.base_score:.4f}",
                    f"- Rerank score: {hit.rerank_score:.4f}",
                    f"- Extraction method: `{hit.metadata.get('extraction_method', 'unknown')}`",
                    f"- Preview: {preview}",
                ]
            )
        return "\n".join(lines)
