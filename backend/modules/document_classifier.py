"""
Document Classifier module for ForeSight.
Takes extracted text and returns a document-type label using keyword matching.
Each category is scored by the number of keyword hits; the highest score wins.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword dictionary — maps each document type to its signal phrases
# ---------------------------------------------------------------------------
KEYWORDS: dict[str, list[str]] = {
    "bank_statement": [
        # Passbook specific
        "passbook", "pass book", "sb account", "savings account",
        "account no", "account number", "ac no", "a/c no",
        "ifsc", "ifsc code", "micr",
        "branch", "branch name",
        "opening balance", "closing balance",
        "withdrawal", "deposit",
        "dr", "cr",           # debit/credit abbreviations
        "narration",
        "chq", "cheque",
        "neft", "rtgs", "imps", "upi",
        "state bank", "sbi", "hdfc", "icici", "axis", "canara",
        "federal bank", "south indian bank", "kerala gramin",
        # Common passbook table headers
        "date", "particulars", "debit", "credit", "balance",
        "transaction",
    ],
    
    "identity_proof": [
        "aadhaar", "aadhar", "uid", "unique identification",
        "pan", "permanent account",
        "passport", "date of birth", "dob",
        "gender", "male", "female",
        "government of india", "uidai",
    ],
    
    "land_record": [
        "survey number", "survey no", "patta", "khata",
        "land record", "revenue", "taluk", "village",
        "extent", "hissa", "khasra",
        "land use", "ownership", "pattadar",
    ],
    
    "sale_deed": [
        "sale deed", "deed of sale", "vendor", "vendee",
        "stamp duty", "registered", "sub registrar",
        "consideration", "conveyance",
    ],
    
    "valuation_report": [
        "valuation", "fair market value", "appraiser",
        "property value", "market value", "valuation report",
        "approved valuer", "guideline value",
    ],
}

_DEFAULT_LABEL = "unknown"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ClassificationResult:
    """Holds the output of the document classifier."""

    label: str
    """Best-matching document type (e.g. ``'sale_deed'``)."""

    confidence: float
    """Ratio of winning keyword hits to total keywords for that category (0.0–1.0)."""

    scores: dict[str, int]
    """Raw keyword-hit counts for every category."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_document(text: str) -> ClassificationResult:
    """
    Classify a document based on its extracted text.

    Workflow
    --------
    1. Normalise the input text to lowercase for case-insensitive matching.
    2. For every category in ``KEYWORDS``, count how many of its keywords
       appear anywhere in the text.
    3. The category with the highest hit count is selected as the label.
    4. If no keywords match at all, the label defaults to ``'unknown'``.

    Parameters
    ----------
    text : str
        The full text extracted from a document (e.g. via ``ocr_engine.extract_text``).

    Returns
    -------
    ClassificationResult
        A dataclass containing the predicted label, a confidence score,
        and the raw per-category keyword-hit counts.
    """
    if not text or not text.strip():
        logger.warning("Empty text received — returning '%s'", _DEFAULT_LABEL)
        return ClassificationResult(
            label=_DEFAULT_LABEL,
            confidence=0.0,
            scores={cat: 0 for cat in KEYWORDS},
        )

    text_lower = text.lower()

    # --- Score each category by counting keyword hits ---
    scores: dict[str, int] = {}
    for category, keywords in KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = hits
        logger.debug("Category %-20s → %d / %d hits", category, hits, len(keywords))

    # --- Pick the winner (highest score) ---
    best_category = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_category]

    if best_score == 0:
        logger.info("No keyword matches found — classifying as '%s'", _DEFAULT_LABEL)
        return ClassificationResult(
            label=_DEFAULT_LABEL,
            confidence=0.0,
            scores=scores,
        )

    # Confidence = fraction of that category's keywords that matched
    confidence = best_score / len(KEYWORDS[best_category])

    logger.info(
        "Classified as '%s' (confidence %.0f%%, %d keyword hits)",
        best_category,
        confidence * 100,
        best_score,
    )
    return ClassificationResult(
        label=best_category,
        confidence=round(confidence, 4),
        scores=scores,
    )
