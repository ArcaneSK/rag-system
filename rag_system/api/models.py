from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=16_000)


class AttachmentReference(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=512)
    uri: str | None = Field(default=None, max_length=2_048)
    content_type: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=12_000)
    session_id: str | None = Field(default=None, max_length=200)
    user_id: str | None = Field(default=None, max_length=200)
    channel: str | None = Field(default=None, max_length=100)
    history: list[ChatMessage] = Field(default_factory=list)
    attachments: list[AttachmentReference] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    debug: bool = False


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    chunk_id: str
    file_name: str
    relative_path: str
    page_number: int


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    chunk_id: str
    file_name: str
    relative_path: str
    page_number: int
    snippet: str


class SourceDebug(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    final_score: float
    hybrid_score: float
    vector_score: float
    keyword_score: float
    rerank_score: float
    extraction_method: str


class RetrievalDebug(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hit_count: int
    sources: list[SourceDebug] = Field(default_factory=list)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    session_id: str | None = None
    channel: str | None = None
    model: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SourceDocument] = Field(default_factory=list)
    retrieval_debug: RetrievalDebug | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"]
    api_key_configured: bool
    indexed_file_count: int
    chunk_count: int
    collection_count: int
