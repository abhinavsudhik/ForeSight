"""
OCR Engine module for ForeSight.
Extracts text from PDF documents and images using the Google Gemini API.

Supports: .pdf, .png, .jpg, .jpeg, .tiff, .bmp, .webp

Pipeline
--------
1. If the file is a text-based PDF → fast extract via pdfplumber (no OCR needed).
2. If the file is a scanned PDF   → render each page to a PIL Image with PyMuPDF,
   then run Gemini OCR on each page.
3. If the file is a raster image  → run Gemini OCR directly on the image.
"""

import os
import logging
from pathlib import Path

import fitz                    # PyMuPDF — PDF → image conversion
import numpy as np
from PIL import Image
import pdfplumber
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
_PDF_RENDER_DPI = 200                  # render DPI for scanned PDFs
_PDF_RENDER_ZOOM = _PDF_RENDER_DPI / 72
_PDFPLUMBER_MIN_CHARS = 50             # chars threshold: below this → use OCR


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
    Run Gemini OCR on a single PIL Image and return the transcribed text.
    """
    try:
        logger.info("Calling Gemini OCR API (%s) …", GEMINI_MODEL)
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = (
            "You are a precise document OCR reader. Transcribe all text from this image exactly as it appears. "
            "Do not add any preamble, metadata, or post-conversation remarks. "
            "Output only the extracted text."
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[img, prompt]
        )
        return response.text or ""
    except Exception as exc:
        logger.error("Gemini OCR failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(file_path: str) -> str:
    """
    Extract all readable text from a document.

    Workflow
    --------
    1. If the file is a PDF:
       a. Attempt fast text extraction via pdfplumber (no OCR cost).
       b. If that yields fewer than ``_PDFPLUMBER_MIN_CHARS`` characters
          (scanned / image-based PDF), fall back to rendering each page and
          running Gemini OCR.
    2. If the file is a raster image, run Gemini OCR directly.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a PDF or image file.

    Returns
    -------
    str
        Full extracted text with lines/pages separated by newlines.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    # ------------------------------------------------------------------
    # PDF: try pdfplumber first; fall back to Gemini OCR if needed
    # ------------------------------------------------------------------
    if ext == ".pdf":
        logger.info("Processing PDF (pdfplumber fast-path): %s", path.name)
        plumber_text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        plumber_text += page_text + "\n"
        except Exception as exc:
            logger.warning("pdfplumber failed (%s); will use Gemini OCR.", exc)

        if len(plumber_text.strip()) >= _PDFPLUMBER_MIN_CHARS:
            logger.info(
                "pdfplumber extracted %d chars — skipping OCR.", len(plumber_text)
            )
            return plumber_text

        # Scanned PDF → render pages and OCR each
        logger.info(
            "pdfplumber yielded only %d chars — rendering pages for Gemini OCR …",
            len(plumber_text.strip()),
        )
        images = _pdf_to_pil_images(file_path)
        page_texts: list[str] = []
        for idx, img in enumerate(images):
            logger.info("Running Gemini OCR on page %d / %d …", idx + 1, len(images))
            page_text = _ocr_single_image(img)
            if page_text:
                page_texts.append(page_text)

        full_text = "\n\n".join(page_texts)
        logger.info(
            "Gemini OCR complete — %d page(s), %d chars", len(images), len(full_text)
        )
        return full_text

    # ------------------------------------------------------------------
    # Raster image: OCR directly
    # ------------------------------------------------------------------
    elif ext in _SUPPORTED_IMAGE_EXTS:
        logger.info("Processing image with Gemini OCR: %s", path.name)
        img = _load_image_file(file_path)
        text = _ocr_single_image(img)
        logger.info("Gemini OCR complete — %d chars", len(text))
        return text

    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: .pdf, {', '.join(sorted(_SUPPORTED_IMAGE_EXTS))}"
        )