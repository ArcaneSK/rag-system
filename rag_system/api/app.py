from __future__ import annotations

import json
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any, Iterator
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from ..config import Settings
from ..retrieval import RetrievalHit
from ..service import AnswerPlan, RAGSystem
from .models import (
    ChatRequest,
    ChatResponse,
    Citation,
    HealthResponse,
    ReadinessResponse,
    RetrievalDebug,
    SourceDebug,
    SourceDocument,
)


logger = logging.getLogger(__name__)


def _history_payload(payload: ChatRequest) -> list[dict[str, str]]:
    return [item.model_dump(mode="python") for item in payload.history]


def _snippet(text: str, max_chars: int = 320) -> str:
    preview = text.strip()
    if len(preview) <= max_chars:
        return preview
    return f"{preview[:max_chars].rstrip()}..."


def _serialize_hits(hits: list[RetrievalHit], include_debug: bool) -> tuple[
    list[Citation],
    list[SourceDocument],
    RetrievalDebug | None,
]:
    citations: list[Citation] = []
    sources: list[SourceDocument] = []
    debug_sources: list[SourceDebug] = []

    for index, hit in enumerate(hits, start=1):
        label = f"S{index}"
        citations.append(
            Citation(
                label=label,
                chunk_id=hit.chunk_id,
                file_name=hit.metadata["file_name"],
                relative_path=hit.metadata["relative_path"],
                page_number=int(hit.metadata["page_number"]),
            )
        )
        sources.append(
            SourceDocument(
                label=label,
                chunk_id=hit.chunk_id,
                file_name=hit.metadata["file_name"],
                relative_path=hit.metadata["relative_path"],
                page_number=int(hit.metadata["page_number"]),
                snippet=_snippet(hit.text),
            )
        )
        if include_debug:
            debug_sources.append(
                SourceDebug(
                    label=label,
                    final_score=hit.score,
                    hybrid_score=hit.base_score,
                    vector_score=hit.vector_score,
                    keyword_score=hit.keyword_score,
                    rerank_score=hit.rerank_score,
                    extraction_method=str(hit.metadata.get("extraction_method", "unknown")),
                )
            )

    retrieval_debug = None
    if include_debug:
        retrieval_debug = RetrievalDebug(hit_count=len(hits), sources=debug_sources)

    return citations, sources, retrieval_debug


def _build_chat_response(
    *,
    request_id: str,
    payload: ChatRequest,
    rag: RAGSystem,
    answer: str,
    hits: list[RetrievalHit],
) -> ChatResponse:
    citations, sources, retrieval_debug = _serialize_hits(hits, include_debug=payload.debug)
    return ChatResponse(
        request_id=request_id,
        session_id=payload.session_id,
        channel=payload.channel,
        model=rag.settings.chat_model,
        answer=answer,
        citations=citations,
        sources=sources,
        retrieval_debug=retrieval_debug,
    )


def _event_payload(data: Any) -> bytes:
    if hasattr(data, "model_dump"):
        payload = data.model_dump(mode="json")
    else:
        payload = data
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _sse_event(event: str, data: Any) -> bytes:
    return b"".join([b"event: ", event.encode("utf-8"), b"\n", b"data: ", _event_payload(data), b"\n\n"])


def _status_code_for_exception(exc: Exception) -> int:
    message = str(exc)
    if "OPENAI_API_KEY" in message:
        return status.HTTP_503_SERVICE_UNAVAILABLE
    if "Failed to reach the OpenAI API" in message:
        return status.HTTP_502_BAD_GATEWAY
    if "No indexed PDF content is available yet" in message:
        return status.HTTP_409_CONFLICT
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _bearer_auth(request: Request) -> None:
    expected_token = request.app.state.settings.api_bearer_token
    if not expected_token:
        return

    authorization = request.headers.get("authorization", "")
    scheme, _, provided_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not provided_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_rag(request: Request) -> RAGSystem:
    return request.app.state.rag


def get_request_id(request: Request) -> str:
    return request.state.request_id


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app_settings = settings or Settings.from_env()
        app.state.settings = app_settings
        app.state.rag = RAGSystem(app_settings)
        yield

    app = FastAPI(
        title="PDF RAG API",
        version="0.1.0",
        summary="Platform-neutral HTTP API for the local PDF RAG system.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id", "").strip() or uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/healthz", response_model=HealthResponse, tags=["system"])
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadinessResponse, tags=["system"])
    async def readyz(rag: RAGSystem = Depends(get_rag)) -> ReadinessResponse:
        service_status = await run_in_threadpool(rag.get_status)
        if not service_status["api_key_configured"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OPENAI_API_KEY is not configured.",
            )
        return ReadinessResponse(
            status="ready",
            api_key_configured=service_status["api_key_configured"],
            indexed_file_count=service_status["indexed_file_count"],
            chunk_count=service_status["chunk_count"],
            collection_count=service_status["collection_count"],
        )

    @app.post(
        "/v1/chat/query",
        response_model=ChatResponse,
        dependencies=[Depends(_bearer_auth)],
        tags=["chat"],
    )
    async def query_chat(
        payload: ChatRequest,
        request_id: str = Depends(get_request_id),
        rag: RAGSystem = Depends(get_rag),
    ) -> ChatResponse:
        history = _history_payload(payload)
        try:
            answer, hits = await run_in_threadpool(rag.answer_question, payload.message, history)
        except Exception as exc:
            logger.exception("Chat query failed", extra={"request_id": request_id})
            raise HTTPException(status_code=_status_code_for_exception(exc), detail=str(exc)) from exc
        return _build_chat_response(
            request_id=request_id,
            payload=payload,
            rag=rag,
            answer=answer,
            hits=hits,
        )

    @app.post(
        "/v1/chat/stream",
        dependencies=[Depends(_bearer_auth)],
        tags=["chat"],
    )
    async def stream_chat(
        payload: ChatRequest,
        request_id: str = Depends(get_request_id),
        rag: RAGSystem = Depends(get_rag),
    ) -> StreamingResponse:
        history = _history_payload(payload)

        def event_stream() -> Iterator[bytes]:
            yield _sse_event(
                "metadata",
                {
                    "request_id": request_id,
                    "session_id": payload.session_id,
                    "channel": payload.channel,
                    "model": rag.settings.chat_model,
                },
            )

            try:
                plan: AnswerPlan = rag.prepare_answer(payload.message, history)
                citations, sources, retrieval_debug = _serialize_hits(plan.hits, include_debug=payload.debug)
                yield _sse_event(
                    "sources",
                    {
                        "citations": [item.model_dump(mode="json") for item in citations],
                        "sources": [item.model_dump(mode="json") for item in sources],
                        "retrieval_debug": (
                            retrieval_debug.model_dump(mode="json") if retrieval_debug is not None else None
                        ),
                    },
                )

                if plan.fallback_answer is not None:
                    answer = plan.fallback_answer
                else:
                    pieces: list[str] = []
                    for delta in rag.stream_answer(plan):
                        pieces.append(delta)
                        yield _sse_event("delta", {"text": delta})
                    answer = "".join(pieces).strip()

                yield _sse_event(
                    "done",
                    _build_chat_response(
                        request_id=request_id,
                        payload=payload,
                        rag=rag,
                        answer=answer,
                        hits=plan.hits,
                    ),
                )
            except Exception as exc:
                logger.exception("Streaming chat failed", extra={"request_id": request_id})
                yield _sse_event(
                    "error",
                    {
                        "request_id": request_id,
                        "message": str(exc),
                    },
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        create_app(settings),
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )
