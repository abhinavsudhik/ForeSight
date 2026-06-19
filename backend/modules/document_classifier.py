"""
Document Classifier module for ForeSight.
Takes extracted text and returns a document-type label using a local
CrossEncoder NLI model (zero-shot classification, fully offline).

Primary strategy: Run zero-shot NLI classification via
cross-encoder/nli-MiniLM2-L6-H768.
Fallback strategy: If the model is unavailable, use keyword-based scoring.
"""

import os
# Force offline mode for HuggingFace hub
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document type registry — comprehensive list of Indian document types
# ---------------------------------------------------------------------------
# Maps internal label → human-readable description (used for NLI candidate pairs)
DOCUMENT_TYPES: dict[str, str] = {
    # ── Identity & Address Proofs (KYC) ──
    "aadhaar_card": "Aadhaar Card — UIDAI-issued 12-digit unique identity card",
    "passport": "Passport — international travel document issued by the Government of India",
    "voter_id": "Voter ID / EPIC — Election Commission photo identity card",
    "driving_licence": "Driving Licence — motor vehicle licence issued by RTO",
    "utility_bill": "Utility Bill — electricity, water, gas, or telephone bill used as address proof",
    "rent_agreement": "Rent Agreement / Lease Deed — rental contract between landlord and tenant",

    # ── Tax & Business Registrations ──
    "pan_card": "PAN Card — Permanent Account Number card issued by Income Tax Department",
    "tan_certificate": "TAN Certificate — Tax Deduction Account Number certificate",
    "gst_certificate": "GST Registration Certificate — Goods and Services Tax registration",
    "shop_establishment_licence": "Shop & Establishment Licence — municipal/labour department licence for business premises",

    # ── Income & Financial Proofs ──
    "salary_slip": "Salary Slip / Pay Slip — monthly salary breakdown from employer",
    "form_16": "Form 16 — TDS certificate issued by employer under Income Tax Act",
    "itr_filing": "ITR Filing / Income Tax Return — annual tax return acknowledgement or form",
    "balance_sheet": "Balance Sheet — financial statement showing assets, liabilities, and equity",
    "profit_loss_statement": "Profit & Loss Statement — income and expenditure statement of a business",
    "bank_statement": "Bank Statement / Passbook — record of transactions from a bank account",

    # ── Property Ownership & Clearance Records ──
    "sale_deed": "Sale Deed — registered deed transferring property ownership from seller to buyer",
    "title_deed": "Title Deed — legal document proving ownership of property",
    "mutation_certificate": "Mutation Certificate / Khata Transfer — revenue record of property ownership change",
    "noc_certificate": "No Objection Certificate (NOC) — clearance certificate from authorities",
    "encumbrance_certificate": "Encumbrance Certificate (EC) — certificate showing property is free of legal dues",

    # ── Corporate Constitution Documents ──
    "partnership_deed": "Partnership Deed — agreement between partners defining business terms",
    "trust_deed": "Trust Deed — document establishing a legal trust and its terms",
    "memorandum_of_association": "Memorandum of Association (MoA) — company's charter document defining objectives",
    "articles_of_association": "Articles of Association (AoA) — rules governing internal management of a company",

    # ── Banking Instruments & Forms ──
    "cheque": "Cheque — negotiable instrument for payment drawn on a bank",
    "demand_draft": "Demand Draft (DD) — pre-paid bank instrument for guaranteed payment",
    "deposit_receipt": "Fixed/Recurring Deposit Receipt — bank receipt for term deposits",
    "account_opening_form": "Account Opening Form — bank application form for opening a new account",
    "loan_application_form": "Loan Application Form — bank/NBFC form for applying for a loan",

    # ── Authorizations & Mandates ──
    "board_resolution": "Board Resolution — formal decision by a company's board of directors",
    "power_of_attorney": "Power of Attorney (PoA) — legal authorization to act on someone's behalf",
    "nach_ecs_mandate": "NACH/ECS Mandate Form — auto-debit authorization for recurring payments",

    # ── Credit & Loan Agreements ──
    "loan_agreement": "Loan Agreement — contract between lender and borrower specifying loan terms",
    "hypothecation_deed": "Hypothecation Deed — agreement pledging movable asset as loan collateral",
    "guarantee_letter": "Guarantee Letter / Letter of Guarantee — commitment by guarantor to repay debt",
}

_DEFAULT_LABEL = "unknown"


# ---------------------------------------------------------------------------
# Keyword fallback dictionary (trimmed version for when model is unavailable)
# ---------------------------------------------------------------------------
KEYWORDS: dict[str, list[str]] = {
    "bank_statement": [
        "passbook", "pass book", "sb account", "savings account",
        "account no", "account number", "ifsc", "branch",
        "opening balance", "closing balance", "withdrawal", "deposit",
        "neft", "rtgs", "imps", "upi", "narration",
        "debit", "credit", "balance", "transaction",
    ],
    "aadhaar_card": [
        "aadhaar", "aadhar", "uid", "unique identification",
        "uidai", "enrolment", "vid",
    ],
    "pan_card": [
        "permanent account number", "pan", "income tax department",
        "nsdl", "utiitsl",
    ],
    "passport": [
        "passport", "republic of india", "travel document",
        "date of issue", "date of expiry", "nationality",
    ],
    "voter_id": [
        "election commission", "voter", "epic", "electoral",
        "elector", "polling station",
    ],
    "driving_licence": [
        "driving licence", "driving license", "rto",
        "motor vehicle", "transport",
    ],
    "sale_deed": [
        "sale deed", "deed of sale", "vendor", "vendee",
        "stamp duty", "registered", "sub registrar",
        "consideration", "conveyance",
    ],
    "title_deed": [
        "title deed", "title document", "ownership deed",
        "freehold", "leasehold",
    ],
    "mutation_certificate": [
        "mutation", "khata", "khata transfer",
        "revenue record", "pattadar",
    ],
    "encumbrance_certificate": [
        "encumbrance", "encumbrance certificate",
        "sub registrar", "ec", "free from encumbrance",
    ],
    "noc_certificate": [
        "no objection", "noc", "clearance certificate",
    ],
    "salary_slip": [
        "salary slip", "pay slip", "payslip",
        "basic salary", "gross salary", "net pay",
        "deductions", "allowances", "hra",
    ],
    "form_16": [
        "form 16", "form no. 16", "tds certificate",
        "certificate under section 203",
    ],
    "itr_filing": [
        "income tax return", "itr", "acknowledgement",
        "assessment year", "return of income",
    ],
    "balance_sheet": [
        "balance sheet", "assets", "liabilities",
        "shareholders equity", "current assets",
        "non-current", "total assets",
    ],
    "profit_loss_statement": [
        "profit and loss", "profit & loss", "p&l",
        "income statement", "revenue", "expenses",
        "net profit", "gross profit", "operating profit",
    ],
    "gst_certificate": [
        "gst", "goods and services tax", "gstin",
        "gst registration", "gst certificate",
    ],
    "tan_certificate": [
        "tan", "tax deduction", "tax deduction account number",
    ],
    "shop_establishment_licence": [
        "shop and establishment", "shop & establishment",
        "licence", "municipal", "labour department",
    ],
    "rent_agreement": [
        "rent agreement", "lease deed", "rental agreement",
        "landlord", "tenant", "lessor", "lessee",
        "monthly rent",
    ],
    "utility_bill": [
        "electricity bill", "water bill", "gas bill",
        "telephone bill", "consumer number", "meter reading",
        "billing period", "due date",
    ],
    "partnership_deed": [
        "partnership deed", "partnership agreement",
        "partner", "firm", "partnership firm",
    ],
    "trust_deed": [
        "trust deed", "trust", "trustee", "settlor",
        "beneficiary", "trust property",
    ],
    "memorandum_of_association": [
        "memorandum of association", "moa",
        "objects clause", "registered office",
        "authorised capital",
    ],
    "articles_of_association": [
        "articles of association", "aoa",
        "bye-laws", "board of directors",
        "share transfer",
    ],
    "cheque": [
        "cheque", "check", "pay to the order of",
        "bearer", "account payee",
    ],
    "demand_draft": [
        "demand draft", "dd", "pay to",
        "banker's draft",
    ],
    "deposit_receipt": [
        "fixed deposit", "recurring deposit",
        "deposit receipt", "fd", "rd",
        "maturity date", "rate of interest",
    ],
    "account_opening_form": [
        "account opening", "application form",
        "kyc", "nomination", "type of account",
    ],
    "loan_application_form": [
        "loan application", "loan request",
        "purpose of loan", "collateral",
        "co-applicant", "co-borrower",
    ],
    "board_resolution": [
        "board resolution", "resolved that",
        "board of directors", "meeting",
        "chairman", "minutes",
    ],
    "power_of_attorney": [
        "power of attorney", "poa", "attorney",
        "principal", "agent", "authorise",
        "hereby appoint",
    ],
    "nach_ecs_mandate": [
        "nach", "ecs", "mandate", "auto debit",
        "standing instruction", "umrn",
    ],
    "loan_agreement": [
        "loan agreement", "loan contract",
        "borrower", "lender", "rate of interest",
        "repayment", "emi", "disbursement",
    ],
    "hypothecation_deed": [
        "hypothecation", "hypothecation deed",
        "movable asset", "security interest",
        "pledge", "collateral",
    ],
    "guarantee_letter": [
        "guarantee", "letter of guarantee",
        "guarantor", "surety", "indemnity",
    ],
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ClassificationResult:
    """Holds the output of the document classifier."""

    label: str
    """Best-matching document type (e.g. ``'sale_deed'``)."""

    confidence: float
    """Confidence score (0.0–1.0). From NLI model or keyword ratio."""

    scores: dict[str, int]
    """Raw keyword-hit counts for every category (empty when model is used)."""


# ---------------------------------------------------------------------------
# Lazy-loaded CrossEncoder NLI model
# ---------------------------------------------------------------------------
_nli_model = None


def _get_nli_model():
    """Load the CrossEncoder NLI model on first call and cache it."""
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading CrossEncoder NLI model (cross-encoder/nli-MiniLM2-L6-H768) …")
        _nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
        logger.info("CrossEncoder NLI model loaded successfully.")
    return _nli_model


# ---------------------------------------------------------------------------
# NLI-powered classification
# ---------------------------------------------------------------------------

def _nli_classify(text: str) -> Optional[ClassificationResult]:
    """
    Use a CrossEncoder NLI model for zero-shot document classification.

    Builds (text_snippet, label_description) pairs for every document type,
    predicts entailment scores, and picks the highest-scoring label.

    Returns a ClassificationResult on success, or None if the model fails.
    """
    try:
        model = _get_nli_model()
    except Exception as exc:
        logger.error("Failed to load NLI model: %s", exc)
        return None

    try:
        labels = list(DOCUMENT_TYPES.keys())
        descriptions = list(DOCUMENT_TYPES.values())

        # Truncate text to first 1000 chars for efficiency
        text_snippet = text[:1000]

        # Build candidate pairs
        pairs = [(text_snippet, desc) for desc in descriptions]

        # Predict entailment scores
        scores = model.predict(pairs)
        scores = np.array(scores)

        # Ensure scores is 2D
        if scores.ndim == 1:
            scores = np.expand_dims(scores, axis=0)

        # Extract entailment class scores (index 2)
        entailment_scores = scores[:, 2]

        # Pick the best label
        best_idx = int(np.argmax(entailment_scores))
        best_label = labels[best_idx]

        # Compute confidence: softmax of top score vs second-best
        sorted_scores = np.sort(entailment_scores)[::-1]
        top_two = sorted_scores[:2]
        exp_scores = np.exp(top_two - np.max(top_two))  # numerical stability
        softmax_probs = exp_scores / exp_scores.sum()
        confidence = float(softmax_probs[0])  # probability of top choice

        logger.info(
            "NLI classified as '%s' (confidence %.0f%%)",
            best_label, confidence * 100,
        )

        return ClassificationResult(
            label=best_label,
            confidence=round(confidence, 4),
            scores={},  # No keyword scores when using NLI model
        )

    except Exception as exc:
        logger.error("NLI classification failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Keyword-based fallback classification
# ---------------------------------------------------------------------------

def _keyword_classify(text: str) -> ClassificationResult:
    """
    Classify a document using keyword matching (fallback when the NLI model is
    unavailable). Scores each category by counting keyword hits.
    """
    text_lower = text.lower()

    scores: dict[str, int] = {}
    for category, keywords in KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = hits
        logger.debug("Category %-30s → %d / %d hits", category, hits, len(keywords))

    best_category = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_category]

    if best_score == 0:
        logger.info("No keyword matches found — classifying as '%s'", _DEFAULT_LABEL)
        return ClassificationResult(
            label=_DEFAULT_LABEL,
            confidence=0.0,
            scores=scores,
        )

    confidence = best_score / len(KEYWORDS[best_category])

    logger.info(
        "Keyword classifier: '%s' (confidence %.0f%%, %d hits)",
        best_category, confidence * 100, best_score,
    )
    return ClassificationResult(
        label=best_category,
        confidence=round(confidence, 4),
        scores=scores,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_document(text: str) -> ClassificationResult:
    """
    Classify a document based on its extracted text.

    Workflow
    --------
    1. If the text is empty, return ``'unknown'`` immediately.
    2. **Primary — NLI Model**: Run zero-shot classification via the
       CrossEncoder NLI model.
    3. **Fallback — Keywords**: If the model is unavailable or fails, fall back
       to keyword-based scoring across all categories.

    Parameters
    ----------
    text : str
        The full text extracted from a document (e.g. via ``ocr_engine.extract_text``).

    Returns
    -------
    ClassificationResult
        A dataclass containing the predicted label, a confidence score,
        and the raw per-category keyword-hit counts (empty if model was used).
    """
    if not text or not text.strip():
        logger.warning("Empty text received — returning '%s'", _DEFAULT_LABEL)
        return ClassificationResult(
            label=_DEFAULT_LABEL,
            confidence=0.0,
            scores={},
        )

    # --- Primary: NLI classification ---
    nli_result = _nli_classify(text)
    if nli_result is not None:
        return nli_result

    # --- Fallback: keyword classification ---
    logger.info("Falling back to keyword-based classification")
    return _keyword_classify(text)
