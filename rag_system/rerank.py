from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankInput:
    chunk_id: str
    text: str


class FastEmbedReranker:
    def __init__(self, model_name: str, cache_dir: Path, max_chars: int) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_chars = max_chars
        self._model = None
        self._disabled_reason: str | None = None

    def rerank(self, query: str, candidates: list[RerankInput]) -> dict[str, float]:
        if not candidates or self._disabled_reason:
            return {}

        model = self._get_model()
        if model is None:
            return {}

        documents = [candidate.text[: self.max_chars] for candidate in candidates]
        try:
            scores = list(model.rerank(query, documents))
        except Exception as exc:  # pragma: no cover - model/runtime specific failure
            logger.warning("FastEmbed reranking failed, falling back to base retrieval: %s", exc)
            return {}

        if len(scores) != len(candidates):
            logger.warning("FastEmbed returned %s scores for %s candidates.", len(scores), len(candidates))
            return {}

        return {
            candidate.chunk_id: float(score)
            for candidate, score in zip(candidates, scores, strict=False)
        }

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            self._disabled_reason = "fastembed is not installed."
            logger.warning(self._disabled_reason)
            return None

        kwargs: dict[str, object] = {"model_name": self.model_name}
        signature = inspect.signature(TextCrossEncoder)
        if "cache_dir" in signature.parameters:
            kwargs["cache_dir"] = str(self.cache_dir)

        try:
            self._model = TextCrossEncoder(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on local model download/cache
            self._disabled_reason = f"FastEmbed reranker failed to initialize: {exc}"
            logger.warning(self._disabled_reason)
            return None
        return self._model
