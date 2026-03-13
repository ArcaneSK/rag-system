from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path
    pdf_dir: Path
    chroma_dir: Path
    state_dir: Path
    model_cache_dir: Path
    manifest_path: Path
    chunk_catalog_path: Path
    collection_name: str
    openai_api_key: str
    openai_base_url: str
    chat_model: str
    embedding_model: str
    chunk_size_words: int
    chunk_overlap_words: int
    vector_candidates: int
    keyword_candidates: int
    rerank_enabled: bool
    rerank_candidates: int
    rerank_model_name: str
    rerank_max_chars: int
    ocr_enabled: bool
    ocr_min_text_chars: int
    ocr_image_area_threshold: float
    ocr_render_dpi: int
    final_chunks: int
    max_context_chunks: int
    response_max_tokens: int
    temperature: float

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        pdf_dir = (root / os.getenv("PDF_DIR", "data/pdfs")).resolve()
        chroma_dir = (root / os.getenv("CHROMA_DIR", "data/chroma")).resolve()
        state_dir = (root / os.getenv("STATE_DIR", "data/state")).resolve()
        model_cache_dir = (root / os.getenv("MODEL_CACHE_DIR", "data/models")).resolve()
        chunk_size_words = max(120, int(os.getenv("CHUNK_SIZE_WORDS", "320")))
        chunk_overlap_words = min(
            max(20, int(os.getenv("CHUNK_OVERLAP_WORDS", "60"))),
            chunk_size_words - 20,
        )
        return cls(
            project_root=root,
            pdf_dir=pdf_dir,
            chroma_dir=chroma_dir,
            state_dir=state_dir,
            model_cache_dir=model_cache_dir,
            manifest_path=(state_dir / "manifest.json").resolve(),
            chunk_catalog_path=(state_dir / "chunk_catalog.json").resolve(),
            collection_name=os.getenv("COLLECTION_NAME", "pdf_rag"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
            vector_candidates=max(1, int(os.getenv("VECTOR_CANDIDATES", "8"))),
            keyword_candidates=max(1, int(os.getenv("KEYWORD_CANDIDATES", "8"))),
            rerank_enabled=os.getenv("RERANK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            rerank_candidates=max(1, int(os.getenv("RERANK_CANDIDATES", "12"))),
            rerank_model_name=os.getenv("RERANK_MODEL_NAME", "jinaai/jina-reranker-v1-turbo-en").strip(),
            rerank_max_chars=max(400, int(os.getenv("RERANK_MAX_CHARS", "1800"))),
            ocr_enabled=os.getenv("OCR_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            ocr_min_text_chars=max(40, int(os.getenv("OCR_MIN_TEXT_CHARS", "120"))),
            ocr_image_area_threshold=min(
                1.0,
                max(0.0, float(os.getenv("OCR_IMAGE_AREA_THRESHOLD", "0.55"))),
            ),
            ocr_render_dpi=max(96, int(os.getenv("OCR_RENDER_DPI", "144"))),
            final_chunks=max(1, int(os.getenv("FINAL_CHUNKS", "6"))),
            max_context_chunks=max(1, int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))),
            response_max_tokens=max(200, int(os.getenv("RESPONSE_MAX_TOKENS", "700"))),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
        )

    def ensure_directories(self) -> None:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)
