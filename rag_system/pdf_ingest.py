from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
from pypdf import PdfReader

from .ocr import OCRProvider


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PDFChunk:
    text: str
    page_number: int
    word_count: int
    extraction_method: str


@dataclass(frozen=True)
class PDFExtractionResult:
    chunks: list[PDFChunk]
    page_count: int
    native_pages: int
    ocr_pages: int


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_segment(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_words(text: str) -> list[str]:
    return text.split()


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    words = split_words(text)
    if not words:
        return []

    step = max(1, chunk_size_words - overlap_words)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunk = " ".join(window).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size_words >= len(words):
            break
    return chunks


def _bbox_area(bbox: tuple[float, float, float, float] | list[float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))


def _page_text_from_dict(page_dict: dict) -> tuple[str, float]:
    segments: list[str] = []
    image_area = 0.0

    for block in page_dict.get("blocks", []):
        bbox = block.get("bbox") or (0.0, 0.0, 0.0, 0.0)
        if block.get("type") == 1:
            image_area += _bbox_area(bbox)
            continue
        if block.get("type") != 0:
            continue

        block_lines: list[str] = []
        for line in block.get("lines", []):
            spans = [span.get("text", "") for span in line.get("spans", [])]
            line_text = normalize_segment(" ".join(spans))
            if line_text:
                block_lines.append(line_text)

        block_text = "\n".join(block_lines).strip()
        if block_text:
            segments.append(block_text)

    return normalize_text("\n\n".join(segments)), image_area


def _pixmap_to_ndarray(pixmap: pymupdf.Pixmap) -> np.ndarray:
    channels = 3 if pixmap.alpha else pixmap.n
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.alpha:
        image = image[:, :, :3]
    elif channels == 1:
        image = np.repeat(image, 3, axis=2)
    return image


def _extract_with_pymupdf(
    pdf_path: Path,
    chunk_size_words: int,
    overlap_words: int,
    ocr_provider: OCRProvider | None,
    ocr_enabled: bool,
    ocr_min_text_chars: int,
    ocr_image_area_threshold: float,
    ocr_render_dpi: int,
) -> PDFExtractionResult:
    chunks: list[PDFChunk] = []
    native_pages = 0
    ocr_pages = 0

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_dict = page.get_text("dict", sort=True)
            native_text, image_area = _page_text_from_dict(page_dict)
            page_area = max(_bbox_area(page.rect), 1.0)
            image_ratio = image_area / page_area

            selected_text = native_text
            extraction_method = "native"
            should_try_ocr = (
                ocr_enabled
                and ocr_provider is not None
                and (len(native_text) < ocr_min_text_chars or image_ratio >= ocr_image_area_threshold)
            )

            if should_try_ocr:
                try:
                    pixmap = page.get_pixmap(dpi=ocr_render_dpi, alpha=False)
                    ocr_result = ocr_provider.extract_text(_pixmap_to_ndarray(pixmap))
                except Exception as exc:
                    logger.warning("OCR failed for %s page %s: %s", pdf_path.name, page_number, exc)
                else:
                    ocr_text = normalize_text(ocr_result.text)
                    if len(ocr_text) > len(native_text):
                        selected_text = ocr_text
                        extraction_method = "ocr"

            if extraction_method == "ocr":
                ocr_pages += 1
            elif selected_text:
                native_pages += 1

            for chunk in chunk_text(selected_text, chunk_size_words, overlap_words):
                chunks.append(
                    PDFChunk(
                        text=chunk,
                        page_number=page_number,
                        word_count=len(split_words(chunk)),
                        extraction_method=extraction_method,
                    )
                )

        return PDFExtractionResult(
            chunks=chunks,
            page_count=len(document),
            native_pages=native_pages,
            ocr_pages=ocr_pages,
        )


def _extract_with_pypdf(
    pdf_path: Path,
    chunk_size_words: int,
    overlap_words: int,
) -> PDFExtractionResult:
    reader = PdfReader(str(pdf_path))
    output: list[PDFChunk] = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = normalize_text(raw_text)
        for chunk in chunk_text(text, chunk_size_words, overlap_words):
            output.append(
                PDFChunk(
                    text=chunk,
                    page_number=page_number,
                    word_count=len(split_words(chunk)),
                    extraction_method="native",
                )
            )

    return PDFExtractionResult(
        chunks=output,
        page_count=len(reader.pages),
        native_pages=len(reader.pages),
        ocr_pages=0,
    )


def extract_pdf_chunks(
    pdf_path: Path,
    chunk_size_words: int,
    overlap_words: int,
    *,
    ocr_provider: OCRProvider | None = None,
    ocr_enabled: bool = True,
    ocr_min_text_chars: int = 120,
    ocr_image_area_threshold: float = 0.55,
    ocr_render_dpi: int = 144,
) -> PDFExtractionResult:
    try:
        return _extract_with_pymupdf(
            pdf_path=pdf_path,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            ocr_provider=ocr_provider,
            ocr_enabled=ocr_enabled,
            ocr_min_text_chars=ocr_min_text_chars,
            ocr_image_area_threshold=ocr_image_area_threshold,
            ocr_render_dpi=ocr_render_dpi,
        )
    except Exception as exc:
        logger.warning("PyMuPDF extraction failed for %s, falling back to pypdf: %s", pdf_path.name, exc)
        return _extract_with_pypdf(pdf_path, chunk_size_words, overlap_words)
