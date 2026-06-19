"""
OCR Engine module for ForeSight.
Extracts text from PDF documents and images using Surya OCR (offline).

Supports: .pdf, .png, .jpg, .jpeg, .tiff, .bmp, .webp

Pipeline
--------
1. If the file is a text-based PDF → fast extract via pdfplumber (no OCR needed).
2. If the file is a scanned PDF   → render each page to a PIL Image with PyMuPDF,
   then run Surya OCR on each page.
3. If the file is a raster image  → run Surya OCR directly on the image.
"""

import os
# Force offline mode for HuggingFace hub
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz                    # PyMuPDF — PDF → image conversion
from PIL import Image
import pdfplumber

# Surya OCR imports (v0.20+ API)
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor, clean_block_html

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
_PDF_RENDER_DPI = 200                  # render DPI for scanned PDFs
_PDF_RENDER_ZOOM = _PDF_RENDER_DPI / 72
_PDFPLUMBER_MIN_CHARS = 50             # chars threshold: below this → use OCR
_OCR_PAGE_TIMEOUT_SECONDS = 120        # max seconds to wait for OCR on a single page
_OCR_TOTAL_TIMEOUT_SECONDS = 600       # max seconds for the entire file


# ---------------------------------------------------------------------------
# OCR Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    """
    Result of an OCR extraction attempt.

    Attributes
    ----------
    text : str
        The extracted text (may be empty on failure).
    status : str
        One of ``"success"``, ``"partial"``, or ``"failed"``.
        - ``"success"``  — all pages/images were processed and text was extracted.
        - ``"partial"``  — some pages timed out or errored, but at least some text was extracted.
        - ``"failed"``   — no text could be extracted at all.
    total_chars : int
        Number of characters in the extracted text.
    pages_total : int
        Total pages/images that were attempted.
    pages_succeeded : int
        Pages that returned text successfully.
    pages_failed : int
        Pages that timed out or raised errors.
    method : str
        Which extraction method was used: ``"pdfplumber"``, ``"surya_ocr"``, or ``"none"``.
    elapsed_seconds : float
        Wall-clock time in seconds for the extraction.
    diagnostics : list[str]
        Human-readable messages describing what happened (useful for UI display).
    """
    text: str = ""
    status: str = "failed"       # "success" | "partial" | "failed"
    total_chars: int = 0
    pages_total: int = 0
    pages_succeeded: int = 0
    pages_failed: int = 0
    method: str = "none"
    elapsed_seconds: float = 0.0
    diagnostics: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lazy-loaded Surya predictor (initialized on first use)
# ---------------------------------------------------------------------------

_rec_predictor = None


def _get_surya_predictor():
    """Load Surya RecognitionPredictor on first call and cache it."""
    global _rec_predictor

    if _rec_predictor is None:
        logger.info("Loading Surya OCR predictor …")
        manager = SuryaInferenceManager()
        _rec_predictor = RecognitionPredictor(manager)
        logger.info("Surya OCR predictor loaded successfully.")

    return _rec_predictor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pdf_to_pil_images(pdf_path: str) -> list[Image.Image]:
    """Render every page of a PDF as a PIL Image (RGB)."""
    images: list[Image.Image] = []
    mat = fitz.Matrix(_PDF_RENDER_ZOOM, _PDF_RENDER_ZOOM)

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
            logger.debug(
                "Rendered PDF page %d (%dx%d)", page_num + 1, pix.width, pix.height
            )

    return images


def _load_image_file(image_path: str) -> Image.Image:
    """Load a raster image file as a PIL Image (RGB)."""
    return Image.open(image_path).convert("RGB")


def _ocr_single_image(img: Image.Image) -> str:
    """
    Run Surya OCR on a single PIL Image and return the transcribed text.
    """
    try:
        predictor = _get_surya_predictor()

        logger.info("Running Surya OCR …")
        results = predictor([img], full_page=True)

        # results is a list of PageOCRResult (one per image)
        if not results:
            return ""

        # Extract text from blocks, stripping any HTML tags
        text_parts = []
        for block in results[0].blocks:
            if block.html and not block.skipped:
                cleaned = clean_block_html(block.html)
                if cleaned:
                    text_parts.append(cleaned)
        return "\n".join(text_parts)

    except Exception as exc:
        logger.error("Surya OCR failed: %s", exc)
        raise exc


def _ocr_single_image_with_timeout(
    img: Image.Image, timeout: int = _OCR_PAGE_TIMEOUT_SECONDS
) -> tuple[str, bool, str]:
    """
    Run OCR on a single image with a timeout guard.

    Returns
    -------
    tuple[str, bool, str]
        (extracted_text, succeeded, diagnostic_message)
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_ocr_single_image, img)
        try:
            text = future.result(timeout=timeout)
            return (text, True, "")
        except FuturesTimeoutError:
            msg = (
                f"OCR timed out after {timeout}s — the document may be too "
                f"noisy, blurry, or low-quality for text extraction"
            )
            logger.warning(msg)
            future.cancel()
            return ("", False, msg)
        except Exception as exc:
            msg = f"OCR error: {exc}"
            logger.error(msg)
            return ("", False, msg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(
    file_path: str,
    page_timeout: int = _OCR_PAGE_TIMEOUT_SECONDS,
    total_timeout: int = _OCR_TOTAL_TIMEOUT_SECONDS,
) -> OCRResult:
    """
    Extract all readable text from a document.

    Workflow
    --------
    1. If the file is a PDF:
       a. Attempt fast text extraction via pdfplumber (no OCR cost).
       b. If that yields fewer than ``_PDFPLUMBER_MIN_CHARS`` characters
          (scanned / image-based PDF), fall back to rendering each page and
          running Surya OCR.
    2. If the file is a raster image, run Surya OCR directly.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a PDF or image file.
    page_timeout : int
        Maximum seconds to spend on OCR for a single page/image.
    total_timeout : int
        Maximum total seconds for the entire file.

    Returns
    -------
    OCRResult
        Structured result with text, status, and diagnostics.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the file extension is not supported.
    """
    start_time = time.monotonic()
    path = Path(file_path)
    result = OCRResult()
    result.diagnostics = []

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    # ------------------------------------------------------------------
    # PDF: try pdfplumber first; fall back to Surya OCR if needed
    # ------------------------------------------------------------------
    if ext == ".pdf":
        logger.info("Processing PDF (pdfplumber fast-path): %s", path.name)
        plumber_text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                result.pages_total = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        plumber_text += page_text + "\n"
        except Exception as exc:
            logger.warning("pdfplumber failed (%s); will use Surya OCR.", exc)
            result.diagnostics.append(f"pdfplumber extraction failed: {exc}")

        if len(plumber_text.strip()) >= _PDFPLUMBER_MIN_CHARS:
            logger.info(
                "pdfplumber extracted %d chars — skipping OCR.", len(plumber_text)
            )
            result.text = plumber_text
            result.status = "success"
            result.total_chars = len(plumber_text)
            result.pages_succeeded = result.pages_total
            result.method = "pdfplumber"
            result.elapsed_seconds = time.monotonic() - start_time
            result.diagnostics.append(
                f"Text-based PDF — pdfplumber extracted {len(plumber_text)} chars."
            )
            return result

        # Scanned PDF → render pages and OCR each
        logger.info(
            "pdfplumber yielded only %d chars — rendering pages for Surya OCR …",
            len(plumber_text.strip()),
        )
        result.diagnostics.append(
            f"pdfplumber yielded only {len(plumber_text.strip())} chars — "
            f"falling back to Surya OCR."
        )
        result.method = "surya_ocr"

        try:
            images = _pdf_to_pil_images(file_path)
        except Exception as exc:
            logger.error("Failed to render PDF pages: %s", exc)
            result.status = "failed"
            result.elapsed_seconds = time.monotonic() - start_time
            result.diagnostics.append(f"Failed to render PDF pages: {exc}")
            return result

        result.pages_total = len(images)
        page_texts: list[str] = []

        for idx, img in enumerate(images):
            # Check total timeout
            elapsed = time.monotonic() - start_time
            if elapsed >= total_timeout:
                remaining = result.pages_total - idx
                msg = (
                    f"Total timeout ({total_timeout}s) reached after page {idx}. "
                    f"Skipping remaining {remaining} page(s)."
                )
                logger.warning(msg)
                result.diagnostics.append(msg)
                result.pages_failed += remaining
                break

            logger.info("Running Surya OCR on page %d / %d …", idx + 1, len(images))
            page_text, succeeded, diag = _ocr_single_image_with_timeout(
                img, timeout=page_timeout
            )

            if succeeded and page_text:
                page_texts.append(page_text)
                result.pages_succeeded += 1
            elif succeeded and not page_text:
                # OCR ran but extracted nothing (blank page, etc.)
                result.pages_succeeded += 1
                result.diagnostics.append(
                    f"Page {idx + 1}: OCR completed but no text found."
                )
            else:
                result.pages_failed += 1
                result.diagnostics.append(f"Page {idx + 1}: {diag}")

        full_text = "\n\n".join(page_texts)
        result.text = full_text
        result.total_chars = len(full_text)
        result.elapsed_seconds = time.monotonic() - start_time

        # Determine overall status
        if result.pages_failed == 0 and result.total_chars > 0:
            result.status = "success"
        elif result.pages_succeeded > 0 and result.total_chars > 0:
            result.status = "partial"
        else:
            result.status = "failed"

        logger.info(
            "Surya OCR complete — %d page(s), %d succeeded, %d failed, %d chars",
            result.pages_total,
            result.pages_succeeded,
            result.pages_failed,
            result.total_chars,
        )

        if result.status == "failed":
            result.diagnostics.append(
                "OCR could not extract any text from this document. "
                "This is likely due to a noisy, blurry, or low-quality scan. "
                "Please upload a clearer copy."
            )
        elif result.status == "partial":
            result.diagnostics.append(
                f"Partial extraction: {result.pages_succeeded}/{result.pages_total} "
                f"pages succeeded ({result.total_chars} chars)."
            )
        else:
            result.diagnostics.append(
                f"Full extraction: {result.pages_total} page(s), "
                f"{result.total_chars} chars."
            )

        return result

    # ------------------------------------------------------------------
    # Raster image: OCR directly
    # ------------------------------------------------------------------
    elif ext in _SUPPORTED_IMAGE_EXTS:
        logger.info("Processing image with Surya OCR: %s", path.name)
        result.method = "surya_ocr"
        result.pages_total = 1

        try:
            img = _load_image_file(file_path)
        except Exception as exc:
            logger.error("Failed to load image: %s", exc)
            result.status = "failed"
            result.pages_failed = 1
            result.elapsed_seconds = time.monotonic() - start_time
            result.diagnostics.append(f"Failed to load image: {exc}")
            return result

        text, succeeded, diag = _ocr_single_image_with_timeout(
            img, timeout=page_timeout
        )

        result.text = text
        result.total_chars = len(text)
        result.elapsed_seconds = time.monotonic() - start_time

        if succeeded and text:
            result.status = "success"
            result.pages_succeeded = 1
            result.diagnostics.append(
                f"OCR completed — {len(text)} chars extracted."
            )
        elif succeeded and not text:
            result.status = "failed"
            result.pages_succeeded = 1
            result.diagnostics.append(
                "OCR completed but no text could be extracted from this image. "
                "This is likely due to a noisy, blurry, or low-quality document. "
                "Please upload a clearer copy."
            )
        else:
            result.status = "failed"
            result.pages_failed = 1
            result.diagnostics.append(diag)

        logger.info("Surya OCR complete — %d chars", result.total_chars)
        return result

    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: .pdf, {', '.join(sorted(_SUPPORTED_IMAGE_EXTS))}"
        )