"""
Field Extractor module for ForeSight.

Takes raw text + document type → returns a structured dictionary of key fields.

Document → Fields:
  Identity Proof   → name, dob, address, id_number
  Land Record      → owner_name, survey_number, property_id, area, address
  Sale Deed        → seller_name, buyer_name, property_id, date, amount
  Valuation Report → property_id, valuation_amount, date, appraiser
  Bank Statement   → account_holder, account_number, monthly_credits, monthly_debits

Extraction approach (two layers):
  1. Regex first  — for structured fields like dates, ID numbers, amounts
  2. spaCy NER    — for names and addresses (uses PERSON, GPE, ORG labels)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import spacy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded spaCy model
# ---------------------------------------------------------------------------
_nlp: Optional[spacy.language.Language] = None


def _get_nlp() -> spacy.language.Language:
    """Return a lazily-loaded spaCy English NER model."""
    global _nlp
    if _nlp is None:
        try:
            logger.info("Loading spaCy model 'en_core_web_sm' …")
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found — "
                "falling back to blank English pipeline"
            )
            _nlp = spacy.blank("en")
    return _nlp


# ---------------------------------------------------------------------------
# Regex pattern library
# ---------------------------------------------------------------------------

# Dates: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD, Month DD YYYY
_DATE_PATTERNS = [
    r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b",
    r"\b(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})\b",
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"\s+\d{2,4})\b",
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"\s+\d{1,2},?\s+\d{2,4})\b",
]

# Aadhaar: 12 digits, optionally grouped as XXXX XXXX XXXX
_AADHAAR_PATTERN = r"\b(\d{4}\s?\d{4}\s?\d{4})\b"

# PAN: ABCDE1234F
_PAN_PATTERN = r"\b([A-Z]{5}\d{4}[A-Z])\b"

# Passport: A1234567 (letter + 7 digits)
_PASSPORT_PATTERN = r"\b([A-Z]\d{7})\b"

# Generic ID number fallback: alphanumeric strings of 6–15 characters
_GENERIC_ID_PATTERN = r"\b([A-Z0-9]{6,15})\b"

# Monetary amounts: ₹ / Rs / INR followed by digits (with commas and decimals)
_AMOUNT_PATTERN = r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d{1,2})?)"

# Plain numeric amounts (used as fallback for totals)
_PLAIN_AMOUNT_PATTERN = r"\b(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)\b"

# Survey number
_SURVEY_NUMBER_PATTERN = r"(?i)(?:survey\s*(?:no\.?|number)\s*[:.]?\s*)([A-Z0-9/\-]+)"

# Property ID
_PROPERTY_ID_PATTERN = (
    r"(?i)(?:property\s*(?:id|no\.?|number)\s*[:.]?\s*)([A-Z0-9/\-]+)"
)

# Area (sq ft, sq m, acres, hectares, cents, guntha)
_AREA_PATTERN = (
    r"([\d,]+(?:\.\d+)?)\s*"
    r"(?:sq\.?\s*(?:ft|feet|meters?|metres?|m)|"
    r"acres?|hectares?|cents?|gunthas?)"
)

# Account number: 9–18 digit string
_ACCOUNT_NUMBER_PATTERN = r"(?i)(?:a/?c\s*(?:no\.?|number)?\s*[:.]?\s*)(\d{9,18})"
_ACCOUNT_NUMBER_STANDALONE = r"\b(\d{9,18})\b"


# ---------------------------------------------------------------------------
# NER helpers
# ---------------------------------------------------------------------------

def _extract_persons(text: str) -> list[str]:
    """Extract PERSON entities using spaCy."""
    doc = _get_nlp()(text)
    return list(dict.fromkeys(
        ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"
    ))


def _extract_addresses(text: str) -> list[str]:
    """
    Extract address-like spans using spaCy GPE / LOC entities and
    surrounding context heuristics.
    """
    doc = _get_nlp()(text)
    locations = list(dict.fromkeys(
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in ("GPE", "LOC", "FAC")
    ))

    # Also look for lines containing common address keywords
    address_keywords = [
        "road", "street", "lane", "nagar", "colony", "district",
        "village", "taluk", "mandal", "pin", "post", "state",
    ]
    address_lines = []
    for line in text.split("\n"):
        line_lower = line.lower().strip()
        if any(kw in line_lower for kw in address_keywords) and len(line.strip()) > 10:
            address_lines.append(line.strip())

    # Combine NER locations with address lines
    combined = locations + address_lines
    return list(dict.fromkeys(combined)) if combined else locations


def _extract_organisations(text: str) -> list[str]:
    """Extract ORG entities using spaCy."""
    doc = _get_nlp()(text)
    return list(dict.fromkeys(
        ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"
    ))


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def _find_first(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    """Return the first regex match group or None."""
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def _find_all(pattern: str, text: str, flags: int = 0) -> list[str]:
    """Return all regex match groups."""
    return [m.strip() for m in re.findall(pattern, text, flags)]


def _find_first_date(text: str) -> Optional[str]:
    """Find the first date-like string in the text."""
    for pattern in _DATE_PATTERNS:
        result = _find_first(pattern, text, re.IGNORECASE)
        if result:
            return result
    return None


def _find_all_dates(text: str) -> list[str]:
    """Find all date-like strings in the text."""
    dates: list[str] = []
    for pattern in _DATE_PATTERNS:
        dates.extend(_find_all(pattern, text, re.IGNORECASE))
    return list(dict.fromkeys(dates))  # deduplicate, preserve order


def _find_amounts(text: str) -> list[str]:
    """Find all monetary amounts in the text."""
    amounts = _find_all(_AMOUNT_PATTERN, text)
    return amounts if amounts else []


def _find_labelled_value(
    label_pattern: str, text: str, flags: int = re.IGNORECASE
) -> Optional[str]:
    """
    Search for a labelled value like 'Name: John Doe' or 'Seller Name : …'.
    The label_pattern should match the label and capture the value.
    """
    match = re.search(label_pattern, text, flags)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# Per-document-type extraction functions
# ---------------------------------------------------------------------------

def _extract_identity_proof(text: str) -> dict[str, Optional[str]]:
    """
    Identity Proof → name, dob, address, id_number

    Layer 1 (Regex): dob, id_number (Aadhaar / PAN / Passport)
    Layer 2 (NER):   name (PERSON), address (GPE)
    """
    fields: dict[str, Optional[str]] = {
        "name": None,
        "dob": None,
        "address": None,
        "id_number": None,
    }

    # --- Layer 1: Regex ---
    # Date of Birth
    dob_labelled = _find_labelled_value(
        r"(?:date\s*of\s*birth|dob|d\.o\.b)\s*[:.]?\s*(.+?)(?:\n|$)", text
    )
    fields["dob"] = dob_labelled or _find_first_date(text)

    # ID number — try specific formats first, then generic
    aadhaar = _find_first(_AADHAAR_PATTERN, text)
    pan = _find_first(_PAN_PATTERN, text)
    passport = _find_first(_PASSPORT_PATTERN, text)

    if aadhaar and len(aadhaar.replace(" ", "")) == 12:
        fields["id_number"] = aadhaar
    elif pan:
        fields["id_number"] = pan
    elif passport:
        fields["id_number"] = passport
    else:
        # Labelled fallback: "UID: …", "No: …"
        labelled_id = _find_labelled_value(
            r"(?:uid|aadhaar|pan|passport)\s*(?:no\.?|number)?\s*[:.]?\s*"
            r"([A-Z0-9\s]{6,15})", text
        )
        fields["id_number"] = labelled_id

    # --- Layer 2: spaCy NER ---
    # Name: prefer labelled, then NER
    labelled_name = _find_labelled_value(
        r"(?:name|holder)\s*[:.]?\s*([A-Za-z\s.]+?)(?:\n|,|$)", text
    )
    if labelled_name and len(labelled_name) > 2:
        fields["name"] = labelled_name
    else:
        persons = _extract_persons(text)
        fields["name"] = persons[0] if persons else None

    # Address
    labelled_addr = _find_labelled_value(
        r"(?:address|residence)\s*[:.]?\s*(.+?)(?:\n\n|\Z)", text
    )
    if labelled_addr and len(labelled_addr) > 5:
        fields["address"] = labelled_addr
    else:
        addresses = _extract_addresses(text)
        fields["address"] = ", ".join(addresses) if addresses else None

    return fields


def _extract_land_record(text: str) -> dict[str, Optional[str]]:
    """
    Land Record → owner_name, survey_number, property_id, area, address

    Layer 1 (Regex): survey_number, property_id, area
    Layer 2 (NER):   owner_name (PERSON), address (GPE)
    """
    fields: dict[str, Optional[str]] = {
        "owner_name": None,
        "survey_number": None,
        "property_id": None,
        "area": None,
        "address": None,
    }

    # --- Layer 1: Regex ---
    fields["survey_number"] = _find_first(_SURVEY_NUMBER_PATTERN, text)
    fields["property_id"] = _find_first(_PROPERTY_ID_PATTERN, text)

    # Area
    area_match = re.search(_AREA_PATTERN, text, re.IGNORECASE)
    if area_match:
        fields["area"] = area_match.group(0).strip()

    # --- Layer 2: spaCy NER ---
    # Owner name: try labelled first
    labelled_owner = _find_labelled_value(
        r"(?:owner|pattadar|khatedar)\s*(?:name)?\s*[:.]?\s*([A-Za-z\s.]+?)(?:\n|,|$)",
        text,
    )
    if labelled_owner and len(labelled_owner) > 2:
        fields["owner_name"] = labelled_owner
    else:
        persons = _extract_persons(text)
        fields["owner_name"] = persons[0] if persons else None

    # Address
    labelled_addr = _find_labelled_value(
        r"(?:address|location|village|district)\s*[:.]?\s*(.+?)(?:\n\n|\Z)", text
    )
    if labelled_addr and len(labelled_addr) > 5:
        fields["address"] = labelled_addr
    else:
        addresses = _extract_addresses(text)
        fields["address"] = ", ".join(addresses) if addresses else None

    return fields


def _extract_sale_deed(text: str) -> dict[str, Optional[str]]:
    """
    Sale Deed → seller_name, buyer_name, property_id, date, amount

    Layer 1 (Regex): property_id, date, amount
    Layer 2 (NER):   seller_name, buyer_name (PERSON)
    """
    fields: dict[str, Optional[str]] = {
        "seller_name": None,
        "buyer_name": None,
        "property_id": None,
        "date": None,
        "amount": None,
    }

    # --- Layer 1: Regex ---
    fields["property_id"] = _find_first(_PROPERTY_ID_PATTERN, text)
    fields["date"] = _find_first_date(text)

    amounts = _find_amounts(text)
    fields["amount"] = amounts[0] if amounts else None

    # --- Layer 2: spaCy NER ---
    # Seller: try labelled "vendor" / "seller"
    labelled_seller = _find_labelled_value(
        r"(?:seller|vendor)\s*(?:name)?\s*[:.]?\s*([A-Za-z\s.]+?)(?:\n|,|$)", text
    )
    # Buyer: try labelled "vendee" / "buyer" / "purchaser"
    labelled_buyer = _find_labelled_value(
        r"(?:buyer|vendee|purchaser)\s*(?:name)?\s*[:.]?\s*([A-Za-z\s.]+?)(?:\n|,|$)",
        text,
    )

    if labelled_seller and len(labelled_seller) > 2:
        fields["seller_name"] = labelled_seller
    if labelled_buyer and len(labelled_buyer) > 2:
        fields["buyer_name"] = labelled_buyer

    # Fall back to NER if labelled extraction missed either party
    if not fields["seller_name"] or not fields["buyer_name"]:
        persons = _extract_persons(text)
        if persons:
            if not fields["seller_name"]:
                fields["seller_name"] = persons[0]
            if not fields["buyer_name"] and len(persons) > 1:
                fields["buyer_name"] = persons[1]

    return fields


def _extract_valuation_report(text: str) -> dict[str, Optional[str]]:
    """
    Valuation Report → property_id, valuation_amount, date, appraiser

    Layer 1 (Regex): property_id, valuation_amount, date
    Layer 2 (NER):   appraiser (PERSON / ORG)
    """
    fields: dict[str, Optional[str]] = {
        "property_id": None,
        "valuation_amount": None,
        "date": None,
        "appraiser": None,
    }

    # --- Layer 1: Regex ---
    fields["property_id"] = _find_first(_PROPERTY_ID_PATTERN, text)
    fields["date"] = _find_first_date(text)

    # Valuation amount — look for labelled first, then any amount
    labelled_val = _find_labelled_value(
        r"(?:valuation|fair\s*market\s*value|property\s*value|assessed\s*value)"
        r"\s*[:.]?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d{1,2})?)",
        text,
    )
    if labelled_val:
        fields["valuation_amount"] = labelled_val
    else:
        amounts = _find_amounts(text)
        fields["valuation_amount"] = amounts[0] if amounts else None

    # --- Layer 2: spaCy NER ---
    # Appraiser: try labelled, then PERSON, then ORG
    labelled_appraiser = _find_labelled_value(
        r"(?:appraiser|valuer|assessor)\s*(?:name)?\s*[:.]?\s*([A-Za-z\s.]+?)(?:\n|,|$)",
        text,
    )
    if labelled_appraiser and len(labelled_appraiser) > 2:
        fields["appraiser"] = labelled_appraiser
    else:
        persons = _extract_persons(text)
        if persons:
            fields["appraiser"] = persons[0]
        else:
            orgs = _extract_organisations(text)
            fields["appraiser"] = orgs[0] if orgs else None

    return fields


def _extract_bank_statement(text: str) -> dict[str, Optional[str]]:
    """
    Bank Statement → account_holder, account_number, monthly_credits, monthly_debits

    Layer 1 (Regex): account_number, monthly_credits, monthly_debits
    Layer 2 (NER):   account_holder (PERSON)
    """
    fields: dict[str, Optional[str]] = {
        "account_holder": None,
        "account_number": None,
        "monthly_credits": None,
        "monthly_debits": None,
    }

    # --- Layer 1: Regex ---
    # Account number
    acc_no = _find_first(_ACCOUNT_NUMBER_PATTERN, text)
    if not acc_no:
        # Try to find a standalone 9-18 digit number
        candidates = _find_all(_ACCOUNT_NUMBER_STANDALONE, text)
        # Filter for plausible account numbers (9-18 digits)
        acc_candidates = [c for c in candidates if 9 <= len(c) <= 18]
        acc_no = acc_candidates[0] if acc_candidates else None
    fields["account_number"] = acc_no

    # Credits / Debits — look for labelled totals
    credit_val = _find_labelled_value(
        r"(?:total\s*)?(?:credit|credits|cr)\s*[:.]?\s*(?:₹|Rs\.?|INR)?\s*"
        r"([\d,]+(?:\.\d{1,2})?)",
        text,
    )
    debit_val = _find_labelled_value(
        r"(?:total\s*)?(?:debit|debits|dr)\s*[:.]?\s*(?:₹|Rs\.?|INR)?\s*"
        r"([\d,]+(?:\.\d{1,2})?)",
        text,
    )
    fields["monthly_credits"] = credit_val
    fields["monthly_debits"] = debit_val

    # --- Layer 2: spaCy NER ---
    labelled_holder = _find_labelled_value(
        r"(?:account\s*holder|name|customer)\s*[:.]?\s*([A-Za-z\s.]+?)(?:\n|,|$)",
        text,
    )
    if labelled_holder and len(labelled_holder) > 2:
        fields["account_holder"] = labelled_holder
    else:
        persons = _extract_persons(text)
        fields["account_holder"] = persons[0] if persons else None

    return fields


# ---------------------------------------------------------------------------
# Dispatcher — maps document type to its extraction function
# ---------------------------------------------------------------------------
_EXTRACTORS: dict[str, callable] = {
    "identity_proof": _extract_identity_proof,
    "land_record": _extract_land_record,
    "sale_deed": _extract_sale_deed,
    "valuation_report": _extract_valuation_report,
    "bank_statement": _extract_bank_statement,
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """Holds the output of the field extractor."""

    document_type: str
    """The document type used for extraction (e.g. ``'sale_deed'``)."""

    fields: dict[str, Optional[str]]
    """Extracted key-value fields (values are ``None`` if not found)."""

    raw_text_length: int
    """Character count of the input text (for diagnostics)."""

    fields_found: int
    """Number of fields that were successfully extracted (non-None)."""

    fields_missing: list[str]
    """Names of fields that could not be extracted."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_fields(text: str, document_type: str) -> ExtractionResult:
    """
    Extract structured fields from raw document text.

    Workflow
    --------
    1. Look up the extraction function for the given ``document_type``.
    2. Run **Regex** patterns first to capture structured data
       (dates, IDs, amounts, account numbers, survey numbers, etc.).
    3. Run **spaCy NER** second to capture names (PERSON) and
       addresses (GPE / LOC) that regex cannot reliably handle.
    4. Return an ``ExtractionResult`` with the extracted fields and
       diagnostic metadata.

    Parameters
    ----------
    text : str
        The full text extracted from a document (e.g. via ``ocr_engine.extract_text``).
    document_type : str
        The classified document type (e.g. ``'sale_deed'``, ``'identity_proof'``).
        Must match a key in the internal ``_EXTRACTORS`` dictionary.

    Returns
    -------
    ExtractionResult
        A dataclass containing the extracted fields, field counts, and diagnostics.

    Raises
    ------
    ValueError
        If ``document_type`` is not a recognised type.
    """
    if not text or not text.strip():
        logger.warning("Empty text received for extraction")
        if document_type not in _EXTRACTORS:
            raise ValueError(
                f"Unknown document type '{document_type}'. "
                f"Supported: {', '.join(_EXTRACTORS)}"
            )
        extractor = _EXTRACTORS[document_type]
        empty_fields = extractor("")
        return ExtractionResult(
            document_type=document_type,
            fields=empty_fields,
            raw_text_length=0,
            fields_found=0,
            fields_missing=list(empty_fields.keys()),
        )

    if document_type not in _EXTRACTORS:
        raise ValueError(
            f"Unknown document type '{document_type}'. "
            f"Supported: {', '.join(_EXTRACTORS)}"
        )

    extractor = _EXTRACTORS[document_type]
    logger.info("Extracting fields for document type '%s' …", document_type)

    # --- Run the two-layer extraction ---
    fields = extractor(text)

    # --- Build diagnostics ---
    found = [k for k, v in fields.items() if v is not None]
    missing = [k for k, v in fields.items() if v is None]

    logger.info(
        "Extraction complete — %d/%d fields found (missing: %s)",
        len(found),
        len(fields),
        ", ".join(missing) if missing else "none",
    )

    return ExtractionResult(
        document_type=document_type,
        fields=fields,
        raw_text_length=len(text),
        fields_found=len(found),
        fields_missing=missing,
    )