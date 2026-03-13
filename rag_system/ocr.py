from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRResult:
    text: str
    average_confidence: float | None = None


class OCRProvider(Protocol):
    def extract_text(self, image: np.ndarray) -> OCRResult:
        ...


class RapidOCRProvider:
    def __init__(self) -> None:
        self._engine = None
        self._disabled_reason: str | None = None

    def extract_text(self, image: np.ndarray) -> OCRResult:
        if self._disabled_reason:
            raise RuntimeError(self._disabled_reason)

        if self._engine is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                self._disabled_reason = "RapidOCR is not installed."
                raise RuntimeError(self._disabled_reason) from exc

            try:
                self._engine = RapidOCR(params={"Global.log_level": "error"})
            except Exception as exc:  # pragma: no cover - depends on local OCR assets
                self._disabled_reason = f"RapidOCR failed to initialize: {exc}"
                logger.warning(self._disabled_reason)
                raise RuntimeError(self._disabled_reason) from exc

        try:
            result = self._engine(image)
        except Exception as exc:  # pragma: no cover - runtime OCR engine failure
            raise RuntimeError(f"RapidOCR failed during inference: {exc}") from exc

        if len(result) == 0 or result.txts is None:
            return OCRResult(text="", average_confidence=None)

        lines = [text.strip() for text in result.txts if str(text).strip()]
        confidences: list[float] = []
        if result.scores is not None:
            for score in result.scores:
                try:
                    confidences.append(float(score))
                except (TypeError, ValueError):
                    continue

        average_confidence = None
        if confidences:
            average_confidence = sum(confidences) / len(confidences)

        return OCRResult(text="\n".join(lines).strip(), average_confidence=average_confidence)
