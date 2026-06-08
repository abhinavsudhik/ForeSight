"""
OCR Engine module for ForeSight.
Extracts text from PDF documents and images using PaddleOCR.
Supports: .pdf, .png, .jpg, .jpeg, .tiff, .bmp, .webp
"""

import os
import logging
from pathlib import Path

import fitz                    # PyMuPDF — PDF → image conversion
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import pdfplumber

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level PaddleOCR singleton (lazy-loaded to avoid slow cold starts)
# ---------------------------------------------------------------------------
_ocr_instance: PaddleOCR | None = None

_SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
_PDF_RENDER_DPI = 300          # high DPI for better OCR accuracy
_PDF_RENDER_ZOOM = _PDF_RENDER_DPI / 72  # fitz default is 72 dpi


def _get_ocr() -> PaddleOCR:
    """Return a lazily-initialised PaddleOCR instance."""
    global _ocr_instance
    if _ocr_instance is None:
        logger.info("Initialising PaddleOCR engine …")
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,   # auto-rotate skewed text
            lang="en",            # English model
        )
    return _ocr_instance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pdf_to_images(pdf_path: str) -> list[np.ndarray]:
    """Convert every page of a PDF to a list of NumPy RGB arrays."""
    images: list[np.ndarray] = []
    mat = fitz.Matrix(_PDF_RENDER_ZOOM, _PDF_RENDER_ZOOM)

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            images.append(img)
            logger.debug("Rendered PDF page %d (%dx%d)", page_num + 1, pix.width, pix.height)

    return images


def _image_file_to_array(image_path: str) -> np.ndarray:
    """Load an image file and return it as a NumPy RGB array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def _ocr_single_image(img: np.ndarray) -> list[str]:
    """Run PaddleOCR on a single image array and return detected text lines."""
    ocr = _get_ocr()
    result = ocr.ocr(img, cls=True)

    lines: list[str] = []
    if not result:
        return lines

    for page_result in result:          # result is list[list[line]]
        if page_result is None:
            continue
        for line_info in page_result:
            # line_info = [bbox, (text, confidence)]
            text, confidence = line_info[1]
            lines.append(text)
            logger.debug("  OCR line (%.2f): %s", confidence, text[:80])

    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(file_path: str) -> str:
    """
    Extract all readable text from a document.

    Workflow
    --------
    1. If the file is a PDF → render each page to an image with PyMuPDF.
    2. If the file is an image → load it directly.
    3. Run PaddleOCR on every image.
    4. Join all detected text lines and return a single string.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a PDF or image file.

    Returns
    -------
    str
        The full extracted text, with lines separated by newlines.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the file extension is not supported.
    """
    
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if len(text.strip()) > 50:   # if we got meaningful text
            return text
    except Exception:
        pass


    path = Path(file_path)

    # --- Validate input ---
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    # --- Build a list of images to OCR ---
    if ext == ".pdf":
        logger.info("Processing PDF: %s", path.name)
        images = _pdf_to_images(file_path)
    elif ext in _SUPPORTED_IMAGE_EXTS:
        logger.info("Processing image: %s", path.name)
        images = [_image_file_to_array(file_path)]
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: .pdf, {', '.join(sorted(_SUPPORTED_IMAGE_EXTS))}"
        )

    # --- OCR each image and collect text ---
    all_lines: list[str] = []
    for idx, img in enumerate(images):
        logger.info("Running OCR on page %d / %d …", idx + 1, len(images))
        page_lines = _ocr_single_image(img)
        all_lines.extend(page_lines)

    full_text = "\n".join(all_lines)
    logger.info(
        "Extraction complete — %d page(s), %d line(s), %d chars",
        len(images),
        len(all_lines),
        len(full_text),
    )
    return full_text