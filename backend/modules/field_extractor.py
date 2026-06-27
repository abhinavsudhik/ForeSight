"""
Field Extractor module for ForeSight.

Takes raw text + document type → returns a structured dictionary of key fields.

Document → Fields:
  Identity Proof   → name, dob, address, id_number
  Land Record      → owner_name, survey_number, property_id, area, address
  Sale Deed        → seller_name, buyer_name, property_id, date, amount
  Valuation Report → property_id, valuation_amount, date, appraiser
  Bank Statement   → account_holder, account_number, monthly_credits, monthly_debits

Extraction approach (three layers — QA Model → Regex → spaCy NER):
  1. QA Model      — primary: uses deepset/roberta-base-squad2 extractive QA
                     to pull field values from raw OCR text.
  2. Regex         — fallback: fills gaps with pattern-matched fields
                     (dates, ID numbers, amounts, survey numbers, etc.)
  3. spaCy NER     — final fallback: picks up remaining names (PERSON)
                     and addresses (GPE, LOC) that regex missed.
"""
import os
if os.environ.get("FORESIGHT_ONLINE") == "1":
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
else:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


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


def extract_issue_date_by_keywords(text: str) -> Optional[str]:
    """
    Search for date keywords in text and extract the corresponding date value.
    Supports a list of synonymous keywords for the document's issue date.
    """
    if not text or not text.strip():
        return None

    keywords = [
        "issue date",
        "date of issue",
        "issuance date",
        "date of issuance",
        "date of enrolment",
        "registration date",
        "date of sanction",
        "date of approval",
        "date of grant",
        "date of execution",
        "date of allotment",
        "date of invoice",
        "date of incorporation",
        "date of declaration",
        "result date",
        "date of passing",
        "deed date",
        "date of deed",
        "bill date",
        "date of bill",
        "agreement date",
        "date of agreement",
        "statement date",
        "date of statement",
        "dated"
    ]

    # Try line-by-line matching first for clean local extraction
    for line in text.splitlines():
        line_lower = line.lower()
        for kw in keywords:
            if kw in line_lower:
                # Found the keyword! Extract the suffix after the keyword
                idx = line_lower.find(kw)
                suffix = line[idx + len(kw):].strip()
                # Remove common separators like :, -, =, spaces
                suffix = re.sub(r"^[:\-\s=]+", "", suffix).strip()
                # Now try to find the first date pattern in this suffix
                for pattern in _DATE_PATTERNS:
                    match = re.search(pattern, suffix, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()

    # Fallback: search the whole text for a keyword followed closely by a date
    for kw in keywords:
        for pattern in _DATE_PATTERNS:
            # Match keyword, followed by up to 30 characters of optional punctuation/spaces, followed by the date pattern
            full_pattern = rf"(?i)\b{re.escape(kw)}\b[\s\S]{{0,30}}?({pattern})"
            match = re.search(full_pattern, text)
            if match:
                return match.group(1).strip()

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


def _parse_transactions_to_monthly(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse transactions sequentially from the bank statement OCR text.
    Groups them by month (YYYY-MM), sums up deposits and withdrawals,
    and returns (monthly_credits_str, monthly_debits_str, month_labels_str).
    """
    # Standard months regex
    months_regex_str = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    
    # Regex patterns for different date formats
    # Month-Day (e.g. Feb 03, February 03)
    month_day_pattern = re.compile(rf"\b({months_regex_str})\b[-.\s,]*\b(\d{{1,2}})\b", re.IGNORECASE)
    # Day-Month (e.g. 03 Feb, 3 February)
    day_month_pattern = re.compile(rf"\b(\d{{1,2}})\b[-.\s,]*\b({months_regex_str})\b", re.IGNORECASE)
    # Numeric date (e.g. 03/02/2025, 2025-02-03)
    numeric_date_pattern = re.compile(r"\b(\d{1,4})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b")
    # Plausible year pattern (1990 to 2099)
    year_pattern = re.compile(r"\b(19[9]\d|20\d{2})\b")

    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }

    # Helper to clean amounts
    amount_pattern = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b")

    # Scan text to find the most common year as fallback default_year
    all_years = [int(y) for y in year_pattern.findall(text)]
    if all_years:
        from collections import Counter
        default_year = Counter(all_years).most_common(1)[0][0]
    else:
        import datetime
        default_year = datetime.datetime.now().year

    current_year = default_year
    current_date = None
    lines_since_date = 999
    transactions = []

    lines = text.split("\n")
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
        
        if "TOTAL" in line_strip.upper():
            continue

        # Look for a year update in the line
        year_match = year_pattern.search(line_strip)
        if year_match:
            current_year = int(year_match.group(1))

        # Check for date match
        date_found = False
        month = None
        year = current_year

        # Try Month-Day first (e.g. Feb 03)
        m = month_day_pattern.search(line_strip)
        if m:
            month_str = m.group(1).lower()[:3]
            month = month_map.get(month_str, 1)
            date_found = True
        else:
            # Try Day-Month (e.g. 03 Feb)
            m = day_month_pattern.search(line_strip)
            if m:
                month_str = m.group(2).lower()[:3]
                month = month_map.get(month_str, 1)
                date_found = True
            else:
                # Try numeric date
                m = numeric_date_pattern.search(line_strip)
                if m:
                    g1, g2, g3 = m.group(1), m.group(2), m.group(3)
                    if len(g1) == 4: # YYYY-MM-DD
                        year = int(g1)
                        month = int(g2)
                    elif len(g3) in (2, 4): # DD/MM/YYYY or MM/DD/YYYY
                        year = int(g3)
                        if year < 100:
                            year += 2000
                        if int(g2) > 12:
                            month = int(g1)
                        elif int(g1) > 12:
                            month = int(g2)
                        else:
                            month = int(g1)
                    date_found = True

        if date_found and month is not None:
            current_date = f"{year:04d}-{month:02d}"
            lines_since_date = 0
        else:
            lines_since_date += 1

        # Heuristic: reset current date if we go more than 2 lines without a transaction date.
        # This keeps the context for description wrapping or nearby Opening Balance rows
        # while preventing date persistence issues across pages/unrelated sections.
        if lines_since_date > 2:
            current_date = None

        # Find amounts in this line
        amounts = amount_pattern.findall(line_strip)
        if amounts:
            floats = []
            for amt in amounts:
                try:
                    floats.append(float(amt.replace(",", "")))
                except ValueError:
                    pass
            
            if floats and current_date:
                if "BALANCE FORWARD" in line_strip.upper() or "OPENING BALANCE" in line_strip.upper():
                    transactions.append({
                        "type": "balance_forward",
                        "month_key": current_date,
                        "amount": floats[-1]
                    })
                elif len(floats) >= 2:
                    transactions.append({
                        "type": "transaction",
                        "month_key": current_date,
                        "amount": floats[-2],
                        "balance": floats[-1]
                    })

    if not transactions:
        return None, None, None

    # Calculate running balance and group by month
    running_balance = None
    monthly_data = {}

    for tx in transactions:
        mkey = tx["month_key"]
        if mkey not in monthly_data:
            monthly_data[mkey] = {"credits": 0.0, "debits": 0.0}

        if tx["type"] == "balance_forward":
            running_balance = tx["amount"]
        elif tx["type"] == "transaction":
            amount = tx["amount"]
            balance = tx["balance"]
            if running_balance is not None:
                diff = balance - running_balance
                if diff > 0.01:
                    monthly_data[mkey]["credits"] += amount
                elif diff < -0.01:
                    monthly_data[mkey]["debits"] += amount
            running_balance = balance

    # Format output as comma-separated values sorted by month
    sorted_months = sorted(monthly_data.keys())
    if not sorted_months:
        return None, None, None

    credits_list = [f"{monthly_data[m]['credits']:.2f}" for m in sorted_months]
    debits_list = [f"{monthly_data[m]['debits']:.2f}" for m in sorted_months]

    return (
        ", ".join(credits_list),
        ", ".join(debits_list),
        ", ".join(sorted_months)
    )


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
        "month_labels": None,
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

    # Try transaction state-machine parser first
    monthly_credits, monthly_debits, month_labels = _parse_transactions_to_monthly(text)
    if monthly_credits and monthly_debits:
        fields["monthly_credits"] = monthly_credits
        fields["monthly_debits"] = monthly_debits
        fields["month_labels"] = month_labels
    else:
        # Credits / Debits fallback — look for labelled totals (deposits / withdrawals)
        credit_val = _find_labelled_value(
            r"(?:total\s*)?\b(?:credit|credits|cr|deposit|deposits)\b\s*[:.]?\s*(?:₹|Rs\.?|INR)?\s*"
            r"([\d,]+(?:\.\d{1,2})?)",
            text,
        )
        debit_val = _find_labelled_value(
            r"(?:total\s*)?\b(?:debit|debits|dr|withdrawal|withdrawals)\b\s*[:.]?\s*(?:₹|Rs\.?|INR)?\s*"
            r"([\d,]+(?:\.\d{1,2})?)",
            text,
        )
        fields["monthly_credits"] = credit_val
        fields["monthly_debits"] = debit_val
        fields["month_labels"] = None

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
    # ── Original types (kept for backward compatibility) ──
    "identity_proof": _extract_identity_proof,
    "land_record": _extract_land_record,
    "sale_deed": _extract_sale_deed,
    "valuation_report": _extract_valuation_report,
    "bank_statement": _extract_bank_statement,

    # ── Identity & Address Proofs (KYC) ──
    "aadhaar_card": _extract_identity_proof,
    "passport": _extract_identity_proof,
    "voter_id": _extract_identity_proof,
    "driving_licence": _extract_identity_proof,
    "utility_bill": _extract_identity_proof,
    "rent_agreement": _extract_identity_proof,

    # ── Tax & Business Registrations ──
    "pan_card": _extract_identity_proof,
    "tan_certificate": _extract_identity_proof,
    "gst_certificate": _extract_identity_proof,
    "shop_establishment_licence": _extract_identity_proof,

    # ── Income & Financial Proofs ──
    "salary_slip": _extract_bank_statement,
    "form_16": _extract_bank_statement,
    "itr_filing": _extract_bank_statement,
    "balance_sheet": _extract_bank_statement,
    "profit_loss_statement": _extract_bank_statement,

    # ── Property Ownership & Clearance Records ──
    "agreement_to_sell": _extract_sale_deed,
    "title_deed": _extract_sale_deed,
    "mutation_certificate": _extract_land_record,
    "noc_certificate": _extract_identity_proof,
    "encumbrance_certificate": _extract_land_record,

    # ── Corporate Constitution Documents ──
    "partnership_deed": _extract_identity_proof,
    "trust_deed": _extract_identity_proof,
    "memorandum_of_association": _extract_identity_proof,
    "articles_of_association": _extract_identity_proof,

    # ── Banking Instruments & Forms ──
    "cheque": _extract_bank_statement,
    "demand_draft": _extract_bank_statement,
    "deposit_receipt": _extract_bank_statement,
    "account_opening_form": _extract_bank_statement,
    "loan_application_form": _extract_bank_statement,

    # ── Authorizations & Mandates ──
    "board_resolution": _extract_identity_proof,
    "power_of_attorney": _extract_identity_proof,
    "nach_ecs_mandate": _extract_bank_statement,

    # ── Credit & Loan Agreements ──
    "loan_agreement": _extract_sale_deed,
    "hypothecation_deed": _extract_sale_deed,
    "guarantee_letter": _extract_identity_proof,
}


# ---------------------------------------------------------------------------
# QA model extraction — primary layer
# ---------------------------------------------------------------------------

# Schema of expected fields per document type (used as QA questions)
_FIELD_SCHEMAS: dict[str, dict[str, str]] = {
    # ── Original types ──
    "identity_proof": {
        "name": "Full name of the person on the ID document",
        "dob": "Date of birth (in the format as it appears on the document)",
        "address": "Full residential address",
        "id_number": "ID number (Aadhaar / PAN / Passport number)",
    },
    "land_record": {
        "owner_name": "Name of the property/land owner (pattadar/khatedar)",
        "survey_number": "Survey number of the land",
        "property_id": "Property ID or number",
        "area": "Total area with unit (e.g. 5.5 acres, 2400 sq ft)",
        "address": "Location / address of the property",
    },
    "agreement_to_sell": {
        "seller_name": "Name of the seller",
        "buyer_name": "Name of the buyer",
        "property_id": "Property ID or number or survey number",
        "date": "Date of the agreement",
        "amount": "Sale consideration amount",
    },
    "sale_deed": {
        "seller_name": "Name of the seller / vendor",
        "buyer_name": "Name of the buyer / vendee / purchaser",
        "property_id": "Property ID or number",
        "date": "Date of the deed / registration",
        "amount": "Sale consideration amount",
    },
    "valuation_report": {
        "property_id": "Property ID or number",
        "valuation_amount": "Valuation / assessed / fair market value amount",
        "date": "Date of the valuation report",
        "appraiser": "Name of the appraiser / valuer / assessor",
    },
    "bank_statement": {
        "account_holder": "Name of the account holder / customer",
        "account_number": "Bank account number",
        "monthly_credits": "Total credits / deposits amount",
        "monthly_debits": "Total debits / withdrawals amount",
        "month_labels": "Month labels / periods",
    },

    # ── Identity & Address Proofs (KYC) ──
    "aadhaar_card": {
        "name": "Full name on the Aadhaar card",
        "dob": "Date of birth",
        "address": "Full residential address",
        "id_number": "12-digit Aadhaar number",
    },
    "passport": {
        "name": "Full name on the passport",
        "dob": "Date of birth",
        "address": "Address (if present)",
        "id_number": "Passport number",
    },
    "voter_id": {
        "name": "Full name on the voter ID",
        "dob": "Date of birth or age",
        "address": "Residential address",
        "id_number": "EPIC / Voter ID number",
    },
    "driving_licence": {
        "name": "Full name on the licence",
        "dob": "Date of birth",
        "address": "Residential address",
        "id_number": "Driving licence number",
    },
    "utility_bill": {
        "name": "Name of the consumer / account holder",
        "address": "Service address on the bill",
        "id_number": "Consumer / account number",
        "dob": "Bill date or billing period",
    },
    "rent_agreement": {
        "name": "Tenant name",
        "address": "Rental property address",
        "id_number": "Agreement registration number (if any)",
        "dob": "Agreement start date",
    },

    # ── Tax & Business Registrations ──
    "pan_card": {
        "name": "Full name on the PAN card",
        "dob": "Date of birth",
        "address": "Address (if present)",
        "id_number": "PAN number (10-character alphanumeric)",
    },
    "tan_certificate": {
        "name": "Name of the deductor / entity",
        "dob": "Date of issue",
        "address": "Address of the entity",
        "id_number": "TAN number",
    },
    "gst_certificate": {
        "name": "Legal name of the business",
        "dob": "Date of registration",
        "address": "Principal place of business",
        "id_number": "GSTIN number",
    },
    "shop_establishment_licence": {
        "name": "Name of the establishment / owner",
        "dob": "Date of issue / validity period",
        "address": "Address of the establishment",
        "id_number": "Licence / registration number",
    },

    # ── Income & Financial Proofs ──
    "salary_slip": {
        "account_holder": "Employee name",
        "account_number": "Employee ID or PF number",
        "monthly_credits": "Gross salary / net pay amount",
        "monthly_debits": "Total deductions amount",
    },
    "form_16": {
        "account_holder": "Employee name",
        "account_number": "PAN of the employee",
        "monthly_credits": "Total income / gross salary",
        "monthly_debits": "Total tax deducted (TDS)",
    },
    "itr_filing": {
        "account_holder": "Name of the assessee",
        "account_number": "PAN number",
        "monthly_credits": "Total income declared",
        "monthly_debits": "Total tax payable / paid",
    },
    "balance_sheet": {
        "account_holder": "Name of the company / entity",
        "account_number": "CIN or registration number",
        "monthly_credits": "Total assets",
        "monthly_debits": "Total liabilities",
    },
    "profit_loss_statement": {
        "account_holder": "Name of the company / entity",
        "account_number": "CIN or registration number",
        "monthly_credits": "Total revenue / income",
        "monthly_debits": "Total expenses",
    },

    # ── Property Ownership & Clearance Records ──
    "title_deed": {
        "seller_name": "Previous owner / transferor",
        "buyer_name": "Current owner / transferee",
        "property_id": "Property ID / survey number",
        "date": "Date of the deed",
        "amount": "Consideration amount (if any)",
    },
    "mutation_certificate": {
        "owner_name": "Name of the new owner (mutated in favour of)",
        "survey_number": "Survey number / property number",
        "property_id": "Khata number / property ID",
        "area": "Area of the property",
        "address": "Location / village / taluk",
    },
    "noc_certificate": {
        "name": "Name of the applicant / property owner",
        "dob": "Date of issue",
        "address": "Property / project address",
        "id_number": "NOC reference number",
    },
    "encumbrance_certificate": {
        "owner_name": "Name of the property owner",
        "survey_number": "Survey number",
        "property_id": "Property ID / document number",
        "area": "Period covered (from-to)",
        "address": "Sub-registrar office / jurisdiction",
    },

    # ── Corporate Constitution Documents ──
    "partnership_deed": {
        "name": "Name of the partnership firm",
        "dob": "Date of the deed / partnership commencement",
        "address": "Registered office address",
        "id_number": "Firm registration number",
    },
    "trust_deed": {
        "name": "Name of the trust",
        "dob": "Date of creation of the trust",
        "address": "Registered address of the trust",
        "id_number": "Trust registration number",
    },
    "memorandum_of_association": {
        "name": "Name of the company",
        "dob": "Date of incorporation",
        "address": "Registered office address",
        "id_number": "CIN / Company registration number",
    },
    "articles_of_association": {
        "name": "Name of the company",
        "dob": "Date of adoption",
        "address": "Registered office address",
        "id_number": "CIN / Company registration number",
    },

    # ── Banking Instruments & Forms ──
    "cheque": {
        "account_holder": "Name of the drawer / issuer",
        "account_number": "Account number / cheque number",
        "monthly_credits": "Amount on the cheque",
        "monthly_debits": "N/A",
    },
    "demand_draft": {
        "account_holder": "Name of the payee",
        "account_number": "DD number",
        "monthly_credits": "Amount of the DD",
        "monthly_debits": "N/A",
    },
    "deposit_receipt": {
        "account_holder": "Name of the depositor",
        "account_number": "FD/RD account number",
        "monthly_credits": "Deposit amount / maturity value",
        "monthly_debits": "Interest rate",
    },
    "account_opening_form": {
        "account_holder": "Name of the applicant",
        "account_number": "Account number (if assigned)",
        "monthly_credits": "Initial deposit amount",
        "monthly_debits": "N/A",
    },
    "loan_application_form": {
        "account_holder": "Name of the applicant / borrower",
        "account_number": "Application / reference number",
        "monthly_credits": "Loan amount requested",
        "monthly_debits": "Proposed EMI amount",
    },

    # ── Authorizations & Mandates ──
    "board_resolution": {
        "name": "Name of the company",
        "dob": "Date of the resolution",
        "address": "Registered office address",
        "id_number": "Resolution reference number",
    },
    "power_of_attorney": {
        "name": "Name of the principal (grantor)",
        "dob": "Date of execution",
        "address": "Address of the principal",
        "id_number": "PoA registration number",
    },
    "nach_ecs_mandate": {
        "account_holder": "Name of the account holder",
        "account_number": "Bank account number",
        "monthly_credits": "Maximum amount authorised",
        "monthly_debits": "UMRN / mandate reference number",
    },

    # ── Credit & Loan Agreements ──
    "loan_agreement": {
        "seller_name": "Name of the lender / bank",
        "buyer_name": "Name of the borrower",
        "property_id": "Loan account / reference number",
        "date": "Date of the agreement",
        "amount": "Loan amount / sanctioned amount",
    },
    "hypothecation_deed": {
        "seller_name": "Name of the lender / financier",
        "buyer_name": "Name of the borrower",
        "property_id": "Asset / vehicle registration number",
        "date": "Date of the deed",
        "amount": "Loan / hypothecation amount",
    },
    "guarantee_letter": {
        "name": "Name of the guarantor",
        "dob": "Date of the guarantee",
        "address": "Address of the guarantor",
        "id_number": "Guarantee reference number",
    },
}

# Dynamically append "issue date" and "issue_date" to every document schema in _FIELD_SCHEMAS
for _schema in _FIELD_SCHEMAS.values():
    _schema["issue date"] = "The date the document was issued, registered, signed, declared, or approved"
    _schema["issue_date"] = "The date the document was issued, registered, signed, declared, or approved (alias)"



# ---------------------------------------------------------------------------
# Lazy-loaded QA pipeline (deepset/roberta-base-squad2)
# ---------------------------------------------------------------------------
_qa_pipeline = None


def _get_qa_pipeline():
    """Load the QA pipeline on first call and cache it."""
    global _qa_pipeline
    if _qa_pipeline is None:
        import torch
        from transformers import pipeline as hf_pipeline

        device = os.environ.get("TORCH_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(
            "Loading QA model (deepset/roberta-base-squad2) on device '%s' …", device
        )
        _qa_pipeline = hf_pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=device,
        )
        logger.info("QA model loaded successfully.")
    return _qa_pipeline


def _qa_extract_fields(
    text: str, document_type: str
) -> dict[str, Optional[str]]:
    """
    Use deepset/roberta-base-squad2 extractive QA to pull field values
    from raw OCR text.

    For each field in the document type's schema, constructs a natural-
    language question from the field description and runs the QA model.

    Returns a dict of field_name → value (or None if not found / low score).
    Returns an empty dict if the model fails to load.
    """
    try:
        qa = _get_qa_pipeline()
    except Exception as exc:
        logger.error("Failed to load QA model: %s", exc)
        return {}

    schema = _FIELD_SCHEMAS.get(document_type)
    if not schema:
        logger.warning(
            "No QA schema for document type '%s' — skipping", document_type
        )
        return {}

    # Truncate context to 3000 chars for efficiency
    context = text[:3000]

    result: dict[str, Optional[str]] = {}
    for field_name, description in schema.items():
        if field_name == "month_labels":
            result[field_name] = None
            continue
        # Construct a natural language question from the description
        question = f"What is the {description}?"

        try:
            answer = qa(question=question, context=context)
            if answer["score"] > 0.15:
                result[field_name] = answer["answer"].strip()
            else:
                result[field_name] = None
        except Exception as exc:
            logger.debug(
                "QA failed for field '%s': %s", field_name, exc
            )
            result[field_name] = None

    found_count = sum(1 for v in result.values() if v is not None)
    logger.info(
        "QA extracted %d/%d fields for '%s'",
        found_count, len(schema), document_type,
    )
    return result


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

    Workflow (three layers — QA Model → Regex → spaCy NER)
    -------------------------------------------------------
    1. **QA Model** (primary): Use deepset/roberta-base-squad2 to extract
       field values via extractive question answering.
    2. **Regex + spaCy NER** (fallback): For any fields that the QA model
       returned as ``None`` or if it fails entirely, run the existing
       regex patterns and NER pipeline to fill the gaps.
    3. **Merge**: QA results take priority; regex/NER fills blanks.

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
        empty_fields["issue date"] = None
        empty_fields["issue_date"] = None
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

    # --- Layer 1: QA model extraction (primary) ---
    qa_fields = _qa_extract_fields(text, document_type)

    # --- Layer 2+3: Regex + spaCy NER extraction (fallback) ---
    regex_ner_fields = extractor(text)
    
    # Populate the fallback/regex "issue date" and "issue_date" fields
    issue_date_val = extract_issue_date_by_keywords(text)
    regex_ner_fields["issue date"] = issue_date_val
    regex_ner_fields["issue_date"] = issue_date_val

    # --- Merge: QA takes priority, regex/NER fills gaps ---
    if qa_fields:
        fields = {}
        for key in regex_ner_fields:
            qa_val = qa_fields.get(key)
            regex_val = regex_ner_fields.get(key)
            # Prefer QA model's value if available, else fall back to regex/NER
            fields[key] = qa_val if qa_val is not None else regex_val
        logger.info(
            "Merged fields — QA provided %d, regex/NER filled %d gaps",
            sum(1 for k in fields if qa_fields.get(k) is not None),
            sum(1 for k in fields
                if qa_fields.get(k) is None and regex_ner_fields.get(k) is not None),
        )
    else:
        # QA model failed entirely — use regex/NER only
        fields = regex_ner_fields
        logger.info("QA model unavailable — using regex/NER extraction only")

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