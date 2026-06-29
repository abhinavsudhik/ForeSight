"""
Format & Checksum Validation Module for ForeSight.

Validates the format and checksums of key document fields:
- IFSC code (regex + bank lookup)
- PAN number (regex + entity type decode)
- Account number (digit-only + bank-specific length check via IFSC)
- GSTIN (regex + state code + embedded PAN check)
- MICR code (regex + city/bank/branch split)
- Aadhaar number (first-digit check + Verhoeff checksum algorithm)
"""

import re
import logging
from typing import Optional, Union, Dict, List, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Penalty Weights
# ---------------------------------------------------------------------------

# Penalty Weights Rationale:
# - Aadhaar Verhoeff fail (0.90): Near-certain fake. Aadhaar numbers use a mathematical check digit.
#   A randomly fabricated 12-digit number has a ~90% chance of failing this check.
# - IFSC format fail (0.85): High confidence signal. IFSC has a strict standard format.
#   Failure indicates document forgery or extremely sloppy tampering.
# - PAN format fail (0.85): High confidence signal. PAN has a very strict 10-character structure.
#   Failure is highly indicative of synthetic identity or fake PAN card generation.
# - GSTIN fail (0.80): High confidence. GSTIN has a strict structure containing state code, entity type,
#   and an embedded PAN. Failure indicates a malformed business registration document.
# - Account number mismatch (0.70): Moderate confidence. Account number structures vary by bank,
#   and while most follow length rules, some exceptions (e.g. legacy accounts) exist, so we apply a lower penalty.
WEIGHT_AADHAAR_FAIL = 0.90
WEIGHT_IFSC_FAIL = 0.85
WEIGHT_PAN_FAIL = 0.85
WEIGHT_GSTIN_FAIL = 0.80
WEIGHT_ACCOUNT_MISMATCH = 0.70
WEIGHT_MICR_FAIL = 0.60  # Default moderate weight for MICR format errors

# ---------------------------------------------------------------------------
# Verhoeff Tables for Aadhaar Checksum
# ---------------------------------------------------------------------------
# Dihedral group D5 multiplication table
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

# Permutation table
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 1, 4, 6, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

# Inverse table (for reference/completeness, though not strictly required for validation)
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


# ---------------------------------------------------------------------------
# Databases: Banks Info & PAN Entity Codes
# ---------------------------------------------------------------------------
BANK_INFO = {
    "SBIN": {"name": "State Bank of India", "account_lengths": [11]},
    "ICIC": {"name": "ICICI Bank", "account_lengths": [12]},
    "HDFC": {"name": "HDFC Bank", "account_lengths": [14]},
    "BARB": {"name": "Bank of Baroda", "account_lengths": [14]},
    "PUNB": {"name": "Punjab National Bank", "account_lengths": [16]},
    "UTIB": {"name": "Axis Bank", "account_lengths": [15]},
    "KKBK": {"name": "Kotak Mahindra Bank", "account_lengths": [16]},
    "YESB": {"name": "Yes Bank", "account_lengths": [15]},
    "IBKL": {"name": "IDBI Bank", "account_lengths": [13, 14, 15, 16]},
    "CNRB": {"name": "Canara Bank", "account_lengths": [13]},
    "UBIN": {"name": "Union Bank of India", "account_lengths": [15]},
    "IDIB": {"name": "Indian Bank", "account_lengths": [10, 11, 12, 13, 14, 15, 16, 17]},
    "IOBA": {"name": "Indian Overseas Bank", "account_lengths": [15]},
    "INDB": {"name": "IndusInd Bank", "account_lengths": [13, 14, 15, 16]},
    "FDRL": {"name": "Federal Bank", "account_lengths": [14]},
    "CBIN": {"name": "Central Bank of India", "account_lengths": [10, 11, 12, 13, 14, 15]},
    "UCBA": {"name": "UCO Bank", "account_lengths": [14]},
    "BKID": {"name": "Bank of India", "account_lengths": [15]},
    "CITI": {"name": "Citibank", "account_lengths": [10]},
    "HSBC": {"name": "HSBC Bank", "account_lengths": [10, 11, 12, 13, 14]},
}

PAN_ENTITY_TYPES = {
    "C": "Company",
    "P": "Individual",
    "H": "Hindu Undivided Family (HUF)",
    "F": "Firm",
    "A": "Association of Persons (AOP)",
    "T": "Trust",
    "B": "Body of Individuals (BOI)",
    "L": "Local Authority",
    "J": "Artificial Juridical Person",
    "G": "Government"
}


# ---------------------------------------------------------------------------
# FormatValidator Class
# ---------------------------------------------------------------------------
class FormatValidator:
    """
    Validates various financial and identification formats including
    IFSC, PAN, Account Numbers, GSTIN, MICR, and Aadhaar Checksums.
    """

    @staticmethod
    def _clean_string(val: str) -> str:
        """Remove whitespace, hyphens, and cast to uppercase."""
        return re.sub(r"[\s\-]", "", str(val)).upper()

    @classmethod
    def validate_ifsc(cls, ifsc: str) -> dict:
        """
        Validate IFSC format and look up the bank prefix.
        IFSC pattern: 4 chars (bank code), '0' (reserved), 6 chars (branch code).
        """
        cleaned = cls._clean_string(ifsc)
        if not cleaned:
            return {"valid": False, "reason": "IFSC code is empty", "bank_name": "Unknown Bank"}

        pattern = r"^[A-Z]{4}0[A-Z0-9]{6}$"
        if not re.match(pattern, cleaned):
            return {
                "valid": False,
                "reason": f"IFSC '{cleaned}' does not match pattern [A-Z]{{4}}0[A-Z0-9]{{6}}",
                "bank_name": "Unknown Bank"
            }

        bank_prefix = cleaned[:4]
        bank_name = "Unknown Bank"
        if bank_prefix in BANK_INFO:
            bank_name = BANK_INFO[bank_prefix]["name"]

        return {
            "valid": True,
            "reason": None,
            "bank_name": bank_name
        }

    @classmethod
    def validate_pan(cls, pan: str) -> dict:
        """
        Validate PAN number format and decode entity type from position 4.
        PAN pattern: 5 letters, 4 digits, 1 letter.
        """
        cleaned = cls._clean_string(pan)
        if not cleaned:
            return {"valid": False, "reason": "PAN is empty", "entity_type": "Unknown"}

        pattern = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
        if not re.match(pattern, cleaned):
            return {
                "valid": False,
                "reason": f"PAN '{cleaned}' does not match standard pattern [A-Z]{{5}}[0-9]{{4}}[A-Z]",
                "entity_type": "Unknown"
            }

        entity_char = cleaned[3]
        entity_type = PAN_ENTITY_TYPES.get(entity_char, "Unknown")
        if entity_type == "Unknown":
            return {
                "valid": False,
                "reason": f"PAN '{cleaned}' has invalid entity character '{entity_char}' at position 4",
                "entity_type": "Unknown"
            }

        return {
            "valid": True,
            "reason": None,
            "entity_type": entity_type
        }

    @classmethod
    def validate_account_number(cls, account_number: str, ifsc: Optional[str] = None) -> dict:
        """
        Validate Account Number is digit-only, and verify if its length matches
        the bank's standard length requirements (retrieved via the IFSC prefix).
        """
        # Clean account number: remove spaces and hyphens
        cleaned_acc = cls._clean_string(account_number)
        if not cleaned_acc:
            return {"valid": False, "reason": "Account number is empty", "cleaned_account_number": ""}

        # Digit-only check
        if not cleaned_acc.isdigit():
            return {
                "valid": False,
                "reason": f"Account number contains non-digit characters: {account_number}",
                "cleaned_account_number": cleaned_acc
            }

        # Resolve bank parameters from IFSC if provided
        bank_name = "Unknown Bank"
        expected_lengths = None

        if ifsc:
            cleaned_ifsc = cls._clean_string(ifsc)
            bank_prefix = cleaned_ifsc[:4]
            if bank_prefix in BANK_INFO:
                bank_name = BANK_INFO[bank_prefix]["name"]
                expected_lengths = BANK_INFO[bank_prefix]["account_lengths"]

        # Validate length
        acc_len = len(cleaned_acc)
        if expected_lengths:
            if acc_len not in expected_lengths:
                return {
                    "valid": False,
                    "reason": f"Account number length {acc_len} does not match expected length(s) {expected_lengths} for {bank_name}",
                    "cleaned_account_number": cleaned_acc
                }
        else:
            # Fallback range: standard Indian bank accounts range from 9 to 18 digits
            if not (9 <= acc_len <= 18):
                return {
                    "valid": False,
                    "reason": f"Account number length {acc_len} is outside standard Indian bank account length range of 9-18 digits",
                    "cleaned_account_number": cleaned_acc
                }

        return {
            "valid": True,
            "reason": None,
            "cleaned_account_number": cleaned_acc
        }

    @classmethod
    def validate_gstin(cls, gstin: str) -> dict:
        """
        Validate GSTIN code (15 characters) format, state code, and embedded PAN.
        GSTIN pattern: 2-digit state code, 10-char PAN, 1-char entity count,
        1-char default 'Z', 1-char check digit.
        """
        cleaned = cls._clean_string(gstin)
        if not cleaned:
            return {
                "valid": False,
                "reason": "GSTIN is empty",
                "state_code": None,
                "embedded_pan": None,
                "embedded_pan_entity_type": None
            }

        # GSTIN strict format check
        pattern = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
        if not re.match(pattern, cleaned):
            return {
                "valid": False,
                "reason": f"GSTIN '{cleaned}' does not match standard 15-char regex pattern",
                "state_code": None,
                "embedded_pan": None,
                "embedded_pan_entity_type": None
            }

        # 1. State code validation
        try:
            state_code = int(cleaned[:2])
        except ValueError:
            state_code = 0

        # Standard Indian State/UT codes range from 01 to 38
        if not (1 <= state_code <= 38):
            return {
                "valid": False,
                "reason": f"GSTIN '{cleaned}' has invalid state code '{cleaned[:2]}'. Expected 01 to 38.",
                "state_code": state_code,
                "embedded_pan": None,
                "embedded_pan_entity_type": None
            }

        # 2. Embedded PAN validation
        embedded_pan = cleaned[2:12]
        pan_result = cls.validate_pan(embedded_pan)
        if not pan_result["valid"]:
            return {
                "valid": False,
                "reason": f"GSTIN '{cleaned}' contains an invalid embedded PAN '{embedded_pan}': {pan_result['reason']}",
                "state_code": state_code,
                "embedded_pan": embedded_pan,
                "embedded_pan_entity_type": pan_result["entity_type"]
            }

        return {
            "valid": True,
            "reason": None,
            "state_code": state_code,
            "embedded_pan": embedded_pan,
            "embedded_pan_entity_type": pan_result["entity_type"]
        }

    @classmethod
    def validate_micr(cls, micr: str) -> dict:
        """
        Validate MICR code format (9 digits) and split city/bank/branch codes.
        """
        cleaned = cls._clean_string(micr)
        if not cleaned:
            return {
                "valid": False,
                "reason": "MICR is empty",
                "city_code": None,
                "bank_code": None,
                "branch_code": None
            }

        pattern = r"^[0-9]{9}$"
        if not re.match(pattern, cleaned):
            return {
                "valid": False,
                "reason": f"MICR '{cleaned}' does not match 9-digit format",
                "city_code": None,
                "bank_code": None,
                "branch_code": None
            }

        # Split city, bank, branch codes
        city_code = cleaned[0:3]
        bank_code = cleaned[3:6]
        branch_code = cleaned[6:9]

        return {
            "valid": True,
            "reason": None,
            "city_code": city_code,
            "bank_code": bank_code,
            "branch_code": branch_code
        }

    @classmethod
    def validate_aadhaar(cls, aadhaar: str) -> dict:
        """
        Validate Aadhaar number (12 digits) using digit check, leading character,
        and Verhoeff checksum algorithm.
        """
        cleaned = cls._clean_string(aadhaar)
        if not cleaned:
            return {"valid": False, "reason": "Aadhaar is empty"}

        if not cleaned.isdigit() or len(cleaned) != 12:
            return {"valid": False, "reason": f"Aadhaar must be exactly 12 digits, got: {aadhaar}"}

        # First digit check (UIDAI rule: Aadhaar number cannot start with 0 or 1)
        if cleaned[0] in ("0", "1"):
            return {
                "valid": False,
                "reason": f"Aadhaar cannot start with '{cleaned[0]}'. Must start with 2-9."
            }

        # Verhoeff checksum algorithm check
        digits = [int(x) for x in cleaned]
        c = 0
        for i, val in enumerate(reversed(digits)):
            c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][val]]

        if c != 0:
            return {"valid": False, "reason": "Aadhaar Verhoeff checksum validation failed"}

        return {"valid": True, "reason": None}


# ---------------------------------------------------------------------------
# Public Module Runner
# ---------------------------------------------------------------------------
def run_all(extracted_fields: dict) -> dict:
    """
    Run all applicable validators based on the keys present in extracted_fields.

    Parameters
    ----------
    extracted_fields : dict
        A Python dictionary of extracted document fields.

    Returns
    -------
    dict
        The Format & Checksum validation section results matching the risk scoring schema:
        {
            "section": "format_checksum",
            "risk": "clean|medium|high",
            "flags": list[dict],
            "penalty_weight": float,
            "details": dict
        }
    """
    flags = []
    details = {}

    # 1. IFSC Check
    ifsc_val = None
    for k in ["ifsc", "ifsc_code", "ifsc code", "bank_ifsc"]:
        if k in extracted_fields and extracted_fields[k]:
            ifsc_val = str(extracted_fields[k]).strip()
            break

    if ifsc_val:
        res = FormatValidator.validate_ifsc(ifsc_val)
        details["ifsc"] = res
        if not res["valid"]:
            flags.append({
                "check": "ifsc_format",
                "severity": "high",
                "message": res["reason"],
                "penalty_weight": WEIGHT_IFSC_FAIL,
                "evidence": {"field": "ifsc", "value": ifsc_val}
            })

    # 2. PAN Check
    pan_val = None
    for k in ["pan", "pan_number", "pan card", "pan_card"]:
        if k in extracted_fields and extracted_fields[k]:
            pan_val = str(extracted_fields[k]).strip()
            break

    # Fallback to id_number if it looks like a PAN
    if not pan_val and "id_number" in extracted_fields and extracted_fields["id_number"]:
        candidate = FormatValidator._clean_string(extracted_fields["id_number"])
        if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", candidate):
            pan_val = str(extracted_fields["id_number"]).strip()

    if pan_val:
        res = FormatValidator.validate_pan(pan_val)
        details["pan"] = res
        if not res["valid"]:
            flags.append({
                "check": "pan_format",
                "severity": "high",
                "message": res["reason"],
                "penalty_weight": WEIGHT_PAN_FAIL,
                "evidence": {"field": "pan", "value": pan_val}
            })

    # 3. Account Number Check
    acc_val = None
    for k in ["account_number", "account number", "bank_account_number", "acc_no"]:
        if k in extracted_fields and extracted_fields[k]:
            acc_val = str(extracted_fields[k]).strip()
            break

    if acc_val:
        # Pass IFSC if available to check bank-specific lengths
        res = FormatValidator.validate_account_number(acc_val, ifsc=ifsc_val)
        details["account_number"] = res
        if not res["valid"]:
            flags.append({
                "check": "account_number_mismatch",
                "severity": "medium",
                "message": res["reason"],
                "penalty_weight": WEIGHT_ACCOUNT_MISMATCH,
                "evidence": {
                    "field": "account_number",
                    "value": acc_val,
                    "associated_ifsc": ifsc_val
                }
            })

    # 4. GSTIN Check
    gst_val = None
    for k in ["gstin", "gstin_number", "gstin code", "gstin_code", "gst_number"]:
        if k in extracted_fields and extracted_fields[k]:
            gst_val = str(extracted_fields[k]).strip()
            break

    # Fallback to id_number if it looks like a GSTIN
    if not gst_val and "id_number" in extracted_fields and extracted_fields["id_number"]:
        candidate = FormatValidator._clean_string(extracted_fields["id_number"])
        if re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", candidate):
            gst_val = str(extracted_fields["id_number"]).strip()

    if gst_val:
        res = FormatValidator.validate_gstin(gst_val)
        details["gstin"] = res
        if not res["valid"]:
            flags.append({
                "check": "gstin_format",
                "severity": "medium",
                "message": res["reason"],
                "penalty_weight": WEIGHT_GSTIN_FAIL,
                "evidence": {"field": "gstin", "value": gst_val}
            })

    # 5. MICR Check
    micr_val = None
    for k in ["micr", "micr_code", "micr code"]:
        if k in extracted_fields and extracted_fields[k]:
            micr_val = str(extracted_fields[k]).strip()
            break

    if micr_val:
        res = FormatValidator.validate_micr(micr_val)
        details["micr"] = res
        if not res["valid"]:
            flags.append({
                "check": "micr_format",
                "severity": "medium",
                "message": res["reason"],
                "penalty_weight": WEIGHT_MICR_FAIL,
                "evidence": {"field": "micr", "value": micr_val}
            })

    # 6. Aadhaar Check
    aadhaar_val = None
    for k in ["aadhaar", "aadhar", "aadhaar_number", "aadhar_number", "aadhaar card", "aadhar card"]:
        if k in extracted_fields and extracted_fields[k]:
            aadhaar_val = str(extracted_fields[k]).strip()
            break

    # Fallback to id_number if it looks like an Aadhaar
    if not aadhaar_val and "id_number" in extracted_fields and extracted_fields["id_number"]:
        candidate = FormatValidator._clean_string(extracted_fields["id_number"])
        if re.match(r"^[0-9]{12}$", candidate):
            aadhaar_val = str(extracted_fields["id_number"]).strip()

    if aadhaar_val:
        res = FormatValidator.validate_aadhaar(aadhaar_val)
        details["aadhaar"] = res
        if not res["valid"]:
            flags.append({
                "check": "aadhaar_checksum",
                "severity": "high",
                "message": res["reason"],
                "penalty_weight": WEIGHT_AADHAAR_FAIL,
                "evidence": {"field": "aadhaar", "value": aadhaar_val}
            })

    # Determine risk level and worst flag's weight
    max_weight = max([flag["penalty_weight"] for flag in flags], default=0.0)
    
    if max_weight >= 0.85:
        risk = "high"
    elif max_weight > 0.0:
        risk = "medium"
    else:
        risk = "clean"

    return {
        "section": "format_checksum",
        "risk": risk,
        "flags": flags,
        "penalty_weight": max_weight,
        "details": details
    }
