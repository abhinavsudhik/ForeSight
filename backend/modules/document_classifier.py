"""
Document Classifier module for ForeSight.
Takes extracted text and returns a document-type label using Google Gemini LLM.

Primary strategy: Send the OCR text to Gemini and ask it to pick the closest
document type from a predefined list.
Fallback strategy: If Gemini is unavailable, use keyword-based scoring.
"""

import os
import json
import logging
from dataclasses import dataclass

from google import genai
from dotenv import load_dotenv
from backend.modules import gemini_client

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Document type registry — comprehensive list of Indian document types
# ---------------------------------------------------------------------------
# Maps internal label → human-readable description (used in the Gemini prompt)
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
# Keyword fallback dictionary (trimmed version for when Gemini is unavailable)
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
    """Confidence score (0.0–1.0). From Gemini or keyword ratio."""

    scores: dict[str, int]
    """Raw keyword-hit counts for every category (empty when Gemini is used)."""


# ---------------------------------------------------------------------------
# Gemini-powered classification
# ---------------------------------------------------------------------------

def _gemini_classify(text: str) -> ClassificationResult | None:
    """
    Use Gemini LLM to classify the document text.

    Sends the first ~4000 characters of extracted text to Gemini along with
    the full list of document types, and asks it to pick the best match.

    Returns a ClassificationResult on success, or None if Gemini fails.
    """
    if not gemini_client.is_gemini_available():
        logger.warning("Gemini API not configured — skipping Gemini classification")
        return None

    # Build the document type list for the prompt
    type_list = "\n".join(
        f'  - "{key}": {desc}' for key, desc in DOCUMENT_TYPES.items()
    )

    prompt = f"""You are a document classification expert specializing in Indian financial, legal, and identity documents.

You will be given raw OCR text extracted from a scanned document. Your job is to classify it into exactly ONE of the following document types.

AVAILABLE DOCUMENT TYPES:
{type_list}

CLASSIFICATION RULES:
1. Read the OCR text carefully — look for headers, titles, logos, form numbers, legal language, and structural cues.
2. Pick the SINGLE best-matching document type from the list above.
3. If the document clearly doesn't match ANY type, use "unknown".
4. Assign a confidence score between 0.0 and 1.0 based on how certain you are.
5. Return ONLY a valid JSON object with exactly two keys: "label" and "confidence". No explanation, no markdown.

EXAMPLES:
  - Text mentioning "Unique Identification Authority of India" → {{"label": "aadhaar_card", "confidence": 0.95}}
  - Text with "Sale Deed", "vendor", "vendee", "stamp duty" → {{"label": "sale_deed", "confidence": 0.92}}
  - Text with salary breakdown, basic pay, HRA, deductions → {{"label": "salary_slip", "confidence": 0.88}}

RAW OCR TEXT:
---
{text[:4000]}
---

JSON output:"""

    try:
        logger.info("Calling Gemini for document classification …")
        response = gemini_client.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
        )

        raw_response = (response.text or "").strip()
        logger.debug("Gemini classification raw response: %s", raw_response[:300])

        # Strip markdown code fences if present
        if raw_response.startswith("```"):
            lines = raw_response.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            raw_response = "\n".join(lines).strip()

        parsed = json.loads(raw_response)
        label = parsed.get("label", _DEFAULT_LABEL)
        confidence = float(parsed.get("confidence", 0.0))

        # Validate the label is in our known types
        if label != _DEFAULT_LABEL and label not in DOCUMENT_TYPES:
            logger.warning(
                "Gemini returned unknown label '%s' — mapping to '%s'",
                label, _DEFAULT_LABEL,
            )
            label = _DEFAULT_LABEL
            confidence = 0.0

        logger.info(
            "Gemini classified as '%s' (confidence %.0f%%)",
            label, confidence * 100,
        )

        return ClassificationResult(
            label=label,
            confidence=round(confidence, 4),
            scores={},  # No keyword scores when using Gemini
        )

    except json.JSONDecodeError as exc:
        logger.warning("Gemini returned invalid JSON: %s", exc)
        return None
    except Exception as exc:
        logger.error("Gemini classification failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Keyword-based fallback classification
# ---------------------------------------------------------------------------

def _keyword_classify(text: str) -> ClassificationResult:
    """
    Classify a document using keyword matching (fallback when Gemini is
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
    2. **Primary — Gemini LLM**: Send the text to Gemini and ask it to pick
       the closest document type from the predefined list.
    3. **Fallback — Keywords**: If Gemini is unavailable or fails, fall back
       to keyword-based scoring across all categories.

    Parameters
    ----------
    text : str
        The full text extracted from a document (e.g. via ``ocr_engine.extract_text``).

    Returns
    -------
    ClassificationResult
        A dataclass containing the predicted label, a confidence score,
        and the raw per-category keyword-hit counts (empty if Gemini was used).
    """
    if not text or not text.strip():
        logger.warning("Empty text received — returning '%s'", _DEFAULT_LABEL)
        return ClassificationResult(
            label=_DEFAULT_LABEL,
            confidence=0.0,
            scores={},
        )

    # --- Primary: Gemini classification ---
    gemini_result = _gemini_classify(text)
    if gemini_result is not None:
        return gemini_result

    # --- Fallback: keyword classification ---
    logger.info("Falling back to keyword-based classification")
    return _keyword_classify(text)
