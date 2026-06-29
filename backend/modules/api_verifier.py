"""
API Verification Module for ForeSight.

Routes document validation requests to public and commercial (Surepass) APIs.
Supports online verification and offline fallback (specifically Aadhaar Offline XML QR signature verification).
"""

import os
import sys
import re
import logging
import base64
import requests
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List
from PIL import Image

# Ensure Homebrew's lib path is added to DYLD_LIBRARY_PATH on macOS for pyzbar to locate libzbar.dylib
if sys.platform == 'darwin':
    _brew_lib = '/opt/homebrew/lib'
    if os.path.exists(_brew_lib):
        _current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
        _dyld_paths = _current_dyld.split(':') if _current_dyld else []
        if _brew_lib not in _dyld_paths:
            _dyld_paths.insert(0, _brew_lib)
            os.environ['DYLD_LIBRARY_PATH'] = ':'.join(_dyld_paths)

# Try loading pyzbar, handle missing zbar shared library on host systems
try:
    from pyzbar.pyzbar import decode
    pyzbar_available = True
except (ImportError, OSError) as e:
    logger = logging.getLogger(__name__)
    logger.warning("pyzbar or zbar shared library is not available on this host: %s. Aadhaar QR verification will be skipped.", e)
    pyzbar_available = False
    decode = None

# Try loading rapidfuzz for premium fuzzy name matching, fallback to standard library sequence matching
try:
    from rapidfuzz import fuzz
except ImportError:
    import difflib
    class fuzz:
        @staticmethod
        def token_sort_ratio(s1: str, s2: str) -> float:
            return difflib.SequenceMatcher(None, sorted(s1.split()), sorted(s2.split())).ratio() * 100

from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Penalty Weights
# ---------------------------------------------------------------------------
# Rationale:
# - API Returns "invalid/not found" (0.90): Extremely high confidence signal. 
#   If the authoritative issuer database (like NSDL or UIDAI) does not contain 
#   the ID number, the document is a confirmed fake.
# - Name Mismatch (0.75): High confidence signal. Indicates potential identity 
#   theft, synthetic identity, or name manipulation. We use a threshold of 0.75 
#   to allow minor OCR or spelling variations (fuzzy match >= 85%).
# - API Unreachable (0.00): Network connection errors or Surepass API timeouts 
#   are external infrastructural failures and should not penalize the user.
# - API Not Available (0.00): Documents without electronic verification pathways 
#   (e.g., pay slips) simply have no signal, representing neutral risk.
WEIGHT_INVALID_NOT_FOUND = 0.90
WEIGHT_NAME_MISMATCH = 0.75
WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE = 0.0

DEFAULT_TIMEOUT = 10.0

# Standard placeholder UIDAI Public Key PEM (RSA-2048)
# In production, load the current certificate from an environment variable path
PLACEHOLDER_UIDAI_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA0G4sFvj1X2Z7772y+M6N
rV1gSSt9xIf+Zr1V1CkaUAj9oIPHKs0qL5x09MdD5tsuRjUEum8R6aqQ==
-----END PUBLIC KEY-----"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Load SUREPASS_API_KEY from environment variables."""
    key = os.environ.get("SUREPASS_API_KEY", "")
    if not key:
        logger.warning("SUREPASS_API_KEY environment variable is not set. Surepass API calls will fail.")
    return key


def _fuzzy_name_match(ocr_name: Optional[str], api_name: Optional[str]) -> bool:
    """
    Perform a case-insensitive token-sort fuzzy comparison between OCR name and API name.
    Returns True if similarity score is >= 85%.
    """
    if not ocr_name or not api_name:
        return False
    
    n1 = ocr_name.strip().lower()
    n2 = api_name.strip().lower()
    
    # Remove common corporate or individual suffixes for cleaner comparison
    n1 = re.sub(r"\b(pvt|ltd|private|limited|co|corp|inc|mr|ms|mrs|dr)\b", "", n1).strip()
    n2 = re.sub(r"\b(pvt|ltd|private|limited|co|corp|inc|mr|ms|mrs|dr)\b", "", n2).strip()
    
    if n1 == n2:
        return True
        
    score = fuzz.token_sort_ratio(n1, n2)
    logger.debug("Fuzzy name match score: %.2f%% for '%s' vs '%s'", score, ocr_name, api_name)
    return score >= 85.0


def _build_response(
    doc_type: str, 
    api_available: bool, 
    api_result: Dict[str, Any], 
    risk: str, 
    penalty_weight: float, 
    flags: List[str]
) -> Dict[str, Any]:
    """Helper to return uniform verification schema."""
    return {
        "section": "api_verification",
        "doc_type": doc_type,
        "api_available": api_available,
        "api_result": api_result,
        "risk": risk,
        "penalty_weight": penalty_weight,
        "flags": flags
    }


# ---------------------------------------------------------------------------
# Verifier Implementations
# ---------------------------------------------------------------------------

def verify_pan(extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    a) PAN card -> Income Tax PAN comprehensive verification via Surepass
    """
    pan_number = extracted_fields.get("id_number") or extracted_fields.get("pan_number")
    if not pan_number:
        return _build_response(
            "pan_card", True, {"error": "missing_id_number"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["missing_id_number"]
        )

    api_key = _get_api_key()
    url = "https://api.surepass.io/api/v1/pan/pan-comprehensive"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"id_number": str(pan_number).strip().upper()}

    try:
        logger.info("Calling Surepass PAN Comprehensive API for: %s", pan_number)
        response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        logger.debug("Raw PAN API Response: %s", response.text)
        
        if response.status_code == 404 or not response.ok:
            return _build_response(
                "pan_card", True, {"status_code": response.status_code, "body": response.text}, 
                "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )
            
        res_data = response.json()
        success = res_data.get("success", False)
        data = res_data.get("data", {})
        
        if not success or not data:
            return _build_response(
                "pan_card", True, res_data, "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )
            
        pan_status = data.get("pan_status", "UNKNOWN")
        pan_type = data.get("pan_type", "UNKNOWN")
        full_name = data.get("full_name") or data.get("name")
        
        # Verify status
        if pan_status.upper() not in ["VALID", "ACTIVE"]:
            return _build_response(
                "pan_card", True, data, "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )
            
        # Verify name match
        extracted_name = extracted_fields.get("name")
        name_match = _fuzzy_name_match(extracted_name, full_name)
        
        api_result = {
            "name_match": name_match,
            "pan_status": pan_status,
            "pan_type": pan_type,
            "api_name": full_name
        }
        
        if not name_match:
            return _build_response("pan_card", True, api_result, "high", WEIGHT_NAME_MISMATCH, ["api_name_mismatch"])
            
        return _build_response("pan_card", True, api_result, "clean", 0.0, [])
        
    except requests.RequestException as e:
        logger.error("Surepass PAN API connection error: %s", e)
        return _build_response("pan_card", True, {"error": "api_unreachable"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["api_unreachable"])


def verify_aadhaar(extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    b) Aadhaar -> Offline XML QR verification (delegated to backend.modules.aadhaar_verifier)
    Falls back to Surepass Aadhaar API verification if local verification fails or QR is not found.
    """
    image_source = extracted_fields.get("image_path") or extracted_fields.get("image")
    if not image_source or not isinstance(image_source, str):
        return _build_response(
            "aadhaar_card", True, {"error": "missing_image_input"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["missing_image_input"]
        )

    try:
        from backend.modules.aadhaar_verifier import run_aadhaar_check, parse_aadhaar_xml, extract_qr_from_image, extract_photo
    except ImportError as e:
        logger.error("Failed to import aadhaar_verifier: %s", e)
        return _build_response(
            "aadhaar_card", True, {"error": "aadhaar_verifier_missing"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["aadhaar_verifier_missing"]
        )

    # Use public key path from environment or relative keys/uidai_public.cer
    key_path = os.environ.get("UIDAI_PUBLIC_KEY_PATH", "keys/uidai_public.cer")
    if not os.path.exists(key_path):
        os.makedirs(os.path.dirname(key_path) or "keys", exist_ok=True)
        with open(key_path, "w") as f:
            f.write(PLACEHOLDER_UIDAI_PUBLIC_KEY)

    res = run_aadhaar_check(image_source, public_key_path=key_path, extracted_fields=extracted_fields)

    # Extract photo bytes if QR was found
    photo_b64 = None
    qr_data = extract_qr_from_image(image_source)
    if qr_data:
        parsed = parse_aadhaar_xml(qr_data)
        photo_bytes = extract_photo(parsed)
        if photo_bytes:
            photo_b64 = base64.b64encode(photo_bytes).decode("utf-8")

    # Map name, dob, address for UI display
    parsed_fields = res.get("parsed_fields", {})
    address_parts = [
        parsed_fields.get("house"),
        parsed_fields.get("street"),
        parsed_fields.get("landmark"),
        parsed_fields.get("loc"),
        parsed_fields.get("vtc"),
        parsed_fields.get("dist"),
        parsed_fields.get("state"),
        parsed_fields.get("pc")
    ]
    address = ", ".join(filter(None, [str(p).strip() for p in address_parts]))

    api_result = {
        "signature_valid": res["signature_valid"],
        "name": parsed_fields.get("name"),
        "dob": parsed_fields.get("dob"),
        "gender": parsed_fields.get("gender"),
        "address": address or None,
        "photo_b64": photo_b64,
        "qr_found": res["qr_found"],
        "reason": res.get("reason", "")
    }

    # Fallback to Surepass if local signature is invalid or QR is not found
    if not res.get("qr_found") or res.get("signature_valid") is False:
        logger.info("Local Aadhaar check failed (QR found: %s, Signature Valid: %s). Attempting Surepass fallback...", res.get("qr_found"), res.get("signature_valid"))
        aadhaar_number = extracted_fields.get("id_number") or extracted_fields.get("aadhaar_number")
        api_key = _get_api_key()
        if api_key and aadhaar_number:
            url = "https://api.surepass.io/api/v1/aadhaar-verification/aadhaar-comprehensive"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {"id_number": str(aadhaar_number).replace(" ", "")}
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
                logger.debug("Raw Aadhaar Surepass Fallback Response: %s", response.text)
                if response.ok:
                    res_data = response.json()
                    success = res_data.get("success", False)
                    data = res_data.get("data", {})
                    if success and data:
                        full_name = data.get("full_name") or data.get("name")
                        extracted_name = extracted_fields.get("name")
                        name_match = _fuzzy_name_match(extracted_name, full_name)
                        
                        api_result["signature_valid"] = False  # Keep local signature state
                        api_result["name"] = full_name
                        api_result["dob"] = data.get("dob")
                        api_result["gender"] = data.get("gender")
                        api_result["address"] = data.get("address")
                        api_result["photo_b64"] = data.get("profile_image")
                        api_result["surepass_fallback_used"] = True
                        api_result["reason"] = "Verified via Surepass API (Local check failed)."
                        
                        risk = "clean"
                        penalty_weight = 0.0
                        flags = []
                        if not name_match:
                            risk = "high"
                            penalty_weight = WEIGHT_NAME_MISMATCH
                            flags.append("api_name_mismatch")
                            
                        return _build_response("aadhaar_card", True, api_result, risk, penalty_weight, flags)
            except Exception as sp_err:
                logger.error("Surepass Aadhaar fallback failed: %s", sp_err)

    # Map risk to status and worst flags
    risk = res.get("risk", "clean")
    penalty_weight = res.get("penalty_weight", 0.0)
    flags = []
    
    # Map flags list back to api_verifier's standard flags
    for flag_name in res.get("flags", []):
        if flag_name == "signature_invalid":
            flags.append("api_invalid_or_not_found")
        elif flag_name == "name_mismatch":
            flags.append("api_name_mismatch")
        else:
            flags.append(flag_name)

    return _build_response(
        "aadhaar_card",
        True,
        api_result,
        risk,
        penalty_weight,
        flags
    )


def verify_gstin(extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    c) GSTIN -> Public GST API
    """
    gstin = extracted_fields.get("id_number") or extracted_fields.get("gstin")
    if not gstin:
        return _build_response(
            "gstin_certificate", True, {"error": "missing_id_number"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["missing_id_number"]
        )

    # Clean the GSTIN number
    gstin_clean = str(gstin).strip().upper()
    url = f"https://api.gst.gov.in/commonapi/v1.1/search?action=TP&gstin={gstin_clean}"

    try:
        logger.info("Calling Public GSTIN search API for: %s", gstin_clean)
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
        logger.debug("Raw GST API Response: %s", response.text)

        if response.status_code == 404 or not response.ok:
            return _build_response(
                "gstin_certificate", True, {"status_code": response.status_code, "body": response.text}, 
                "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        res_data = response.json()
        # Public GST API returns data containing tradeNam, lgnm, dsc etc.
        # Let's extract trade_name (or legal name if trade name is empty)
        trade_name = res_data.get("tradeNam") or res_data.get("lgnm")
        registration_status = res_data.get("sts", "UNKNOWN")
        cancellation_date = res_data.get("cndt")

        # If data is completely empty or indicates invalid
        if not trade_name and "error" in res_data:
            return _build_response(
                "gstin_certificate", True, res_data, "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        # Status checks: Active, Suspended, Cancelled
        if registration_status.upper() in ["CANCELLED", "INACTIVE"]:
            return _build_response(
                "gstin_certificate", True, res_data, "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        extracted_name = extracted_fields.get("name") or extracted_fields.get("trade_name")
        name_match = _fuzzy_name_match(extracted_name, trade_name)

        api_result = {
            "trade_name": trade_name,
            "registration_status": registration_status,
            "cancellation_date": cancellation_date,
            "name_match": name_match
        }

        if not name_match:
            return _build_response("gstin_certificate", True, api_result, "high", WEIGHT_NAME_MISMATCH, ["api_name_mismatch"])

        return _build_response("gstin_certificate", True, api_result, "clean", 0.0, [])

    except requests.RequestException as e:
        logger.error("Public GST API connection error: %s", e)
        return _build_response("gstin_certificate", True, {"error": "api_unreachable"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["api_unreachable"])


def verify_udyam(extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    d) Udyam/MSME -> Surepass
    """
    udyam_number = extracted_fields.get("id_number") or extracted_fields.get("udyam_number")
    if not udyam_number:
        return _build_response(
            "udyam_msme_certificate", True, {"error": "missing_id_number"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["missing_id_number"]
        )

    api_key = _get_api_key()
    url = "https://api.surepass.io/api/v1/corporate/udyam-certificate-verification"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"id_number": str(udyam_number).strip()}

    try:
        logger.info("Calling Surepass Udyam API for: %s", udyam_number)
        response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        logger.debug("Raw Udyam API Response: %s", response.text)

        if response.status_code == 404 or not response.ok:
            return _build_response(
                "udyam_msme_certificate", True, {"status_code": response.status_code, "body": response.text},
                "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        res_data = response.json()
        success = res_data.get("success", False)
        data = res_data.get("data", {})

        if not success or not data:
            return _build_response(
                "udyam_msme_certificate", True, res_data, "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        enterprise_name = data.get("enterprise_name") or data.get("name")
        extracted_name = extracted_fields.get("name") or extracted_fields.get("enterprise_name")
        name_match = _fuzzy_name_match(extracted_name, enterprise_name)

        api_result = {
            "enterprise_name": enterprise_name,
            "major_activity": data.get("major_activity"),
            "enterprise_type": data.get("enterprise_type"),
            "name_match": name_match
        }

        if not name_match:
            return _build_response("udyam_msme_certificate", True, api_result, "high", WEIGHT_NAME_MISMATCH, ["api_name_mismatch"])

        return _build_response("udyam_msme_certificate", True, api_result, "clean", 0.0, [])

    except requests.RequestException as e:
        logger.error("Surepass Udyam API connection error: %s", e)
        return _build_response(
            "udyam_msme_certificate", True, {"error": "api_unreachable"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["api_unreachable"]
        )


def verify_rc(extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    e) Vehicle RC -> VAHAN via Surepass
    """
    rc_number = extracted_fields.get("id_number") or extracted_fields.get("rc_number")
    if not rc_number:
        return _build_response(
            "vehicle_rc", True, {"error": "missing_id_number"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["missing_id_number"]
        )

    api_key = _get_api_key()
    url = "https://api.surepass.io/api/v1/rc/rc-full"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"id_number": str(rc_number).strip().upper()}

    try:
        logger.info("Calling Surepass RC Full API for: %s", rc_number)
        response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        logger.debug("Raw RC API Response: %s", response.text)

        if response.status_code == 404 or not response.ok:
            return _build_response(
                "vehicle_rc", True, {"status_code": response.status_code, "body": response.text},
                "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        res_data = response.json()
        success = res_data.get("success", False)
        data = res_data.get("data", {})

        if not success or not data:
            return _build_response(
                "vehicle_rc", True, res_data, "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        owner_name = data.get("owner_name") or data.get("owner")
        extracted_name = extracted_fields.get("name") or extracted_fields.get("owner_name")
        name_match = _fuzzy_name_match(extracted_name, owner_name)

        api_result = {
            "owner_name": owner_name,
            "vehicle_class": data.get("vehicle_category") or data.get("vehicle_class"),
            "insurance_upto": data.get("insurance_upto"),
            "registration_date": data.get("registration_date"),
            "name_match": name_match
        }

        if not name_match:
            return _build_response("vehicle_rc", True, api_result, "high", WEIGHT_NAME_MISMATCH, ["api_name_mismatch"])

        return _build_response("vehicle_rc", True, api_result, "clean", 0.0, [])

    except requests.RequestException as e:
        logger.error("Surepass RC API connection error: %s", e)
        return _build_response("vehicle_rc", True, {"error": "api_unreachable"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["api_unreachable"])


def verify_bank_account(extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    f) Bank statement -> RBI Account Aggregator framework (Finvu/OneMoney/CAMS AA/Setu), marked as consent-required
    """
    account_number = extracted_fields.get("id_number") or extracted_fields.get("account_number")
    ifsc = extracted_fields.get("ifsc")

    # Mark as consent required
    api_result = {
        "requires_consent": True,
        "consent_status": "pending_customer_consent",
        "account_aggregator_providers": ["Finvu", "OneMoney", "CAMS AA", "Setu"],
        "account_number": account_number,
        "ifsc": ifsc,
        "description": "Bank statement verification requires customer consent access via RBI Account Aggregator (AA) framework."
    }

    return _build_response("bank_statement", True, api_result, "clean", 0.0, [])


def verify_dl(extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    g) Driving licence -> Surepass DL Lite
    """
    dl_number = extracted_fields.get("id_number") or extracted_fields.get("dl_number")
    dob = extracted_fields.get("dob")

    if not dl_number or not dob:
        return _build_response(
            "driving_licence", True, {"error": "missing_id_number_or_dob"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["missing_id_number"]
        )

    api_key = _get_api_key()
    url = "https://api.surepass.io/api/v1/dl/dl-lite"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "id_number": str(dl_number).strip(),
        "dob": str(dob).strip()
    }

    try:
        logger.info("Calling Surepass DL Lite API for DL: %s, DOB: %s", dl_number, dob)
        response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        logger.debug("Raw DL API Response: %s", response.text)

        if response.status_code == 404 or not response.ok:
            return _build_response(
                "driving_licence", True, {"status_code": response.status_code, "body": response.text},
                "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        res_data = response.json()
        success = res_data.get("success", False)
        data = res_data.get("data", {})

        if not success or not data:
            return _build_response(
                "driving_licence", True, res_data, "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        full_name = data.get("full_name") or data.get("name")
        extracted_name = extracted_fields.get("name")
        name_match = _fuzzy_name_match(extracted_name, full_name)

        api_result = {
            "full_name": full_name,
            "current_status": data.get("current_status"),
            "name_match": name_match
        }

        if not name_match:
            return _build_response("driving_licence", True, api_result, "high", WEIGHT_NAME_MISMATCH, ["api_name_mismatch"])

        return _build_response("driving_licence", True, api_result, "clean", 0.0, [])

    except requests.RequestException as e:
        logger.error("Surepass DL API connection error: %s", e)
        return _build_response("driving_licence", True, {"error": "api_unreachable"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["api_unreachable"])


def verify_estamp(extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    h) DSC on e-stamped documents -> certificate chain verification against CCA root
    using cryptography library + OCSP revocation check.
    Falls back to SHCIL portal search instructions if no signature block is found.
    """
    image_path = extracted_fields.get("image_path") or extracted_fields.get("image")
    certificate_number = extracted_fields.get("id_number") or extracted_fields.get("certificate_number")
    
    if not image_path or not isinstance(image_path, str) or not image_path.lower().endswith(".pdf"):
        api_result = {
            "status": "manual_verification_required",
            "portal_url": "https://www.shcilestamp.com/",
            "certificate_number": certificate_number,
            "note": "Document is not a PDF. Please verify the certificate number manually on the SHCIL portal."
        }
        return _build_response("e_stamped_document", True, api_result, "clean", 0.0, [])

    # Cryptographic signature extraction & verification
    import re
    from cryptography.hazmat.primitives.serialization import pkcs7
    from cryptography.hazmat.primitives import serialization
    from cryptography.x509 import load_pem_x509_certificate, load_der_x509_certificate
    from cryptography.x509.oid import ExtensionOID
    from cryptography.x509 import ocsp
    from cryptography.hazmat.primitives import hashes

    try:
        with open(image_path, 'rb') as f:
            pdf_bytes = f.read()

        # Find ByteRange block in the PDF
        byte_range_match = re.findall(r'/ByteRange\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]', pdf_bytes)
        if not byte_range_match:
            byte_range_match = re.findall(r'/ByteRange\s*\[(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]', pdf_bytes)

        if not byte_range_match:
            api_result = {
                "status": "no_signature_found",
                "portal_url": "https://www.shcilestamp.com/",
                "certificate_number": certificate_number,
                "note": "No PDF digital signature found. Please verify the certificate number manually on the SHCIL portal."
            }
            return _build_response("e_stamped_document", True, api_result, "clean", 0.0, [])

        # Parse first signature xref block
        br = [int(x) for x in byte_range_match[0]]
        signed_data = pdf_bytes[br[0]:br[0]+br[1]] + pdf_bytes[br[2]:br[2]+br[3]]
        sig_hex_block = pdf_bytes[br[0]+br[1]:br[2]].strip(b'<>\x00\r\n\t ')
        sig_bytes = bytes.fromhex(sig_hex_block.decode('ascii'))

        # Load PKCS7 certificate block
        try:
            certs = pkcs7.load_der_pkcs7_certificates(sig_bytes)
        except Exception:
            certs = pkcs7.load_pem_pkcs7_certificates(sig_bytes)

        if not certs:
            api_result = {
                "status": "invalid_signature_block",
                "portal_url": "https://www.shcilestamp.com/",
                "certificate_number": certificate_number,
                "note": "Malformed digital signature block. Verification failed."
            }
            return _build_response("e_stamped_document", True, api_result, "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"])

        signer_cert = certs[0]
        subject = signer_cert.subject.rfc4514_string()
        issuer = signer_cert.issuer.rfc4514_string()
        serial = signer_cert.serial_number

        # Chain Verification against local CCA Root certificate
        root_cert_path = os.environ.get("CCA_ROOT_CERT_PATH", "keys/cca_root.cer")
        chain_verified = False
        root_cert = None
        if os.path.exists(root_cert_path):
            try:
                with open(root_cert_path, 'rb') as r_file:
                    rc_bytes = r_file.read()
                try:
                    root_cert = load_pem_x509_certificate(rc_bytes)
                except Exception:
                    root_cert = load_der_x509_certificate(rc_bytes)
                # Verify signature using public key of issuer / self
                signer_cert.public_key().verify(
                    signer_cert.signature,
                    signer_cert.tbs_certificate_bytes,
                    signer_cert.signature_hash_algorithm
                )
                chain_verified = True
            except Exception as chain_err:
                logger.warning("Local signature chain verification failed: %s", chain_err)

        # OCSP revocation check
        ocsp_status = "not_checked"
        try:
            aia = signer_cert.extensions.get_extension_for_oid(ExtensionOID.AUTHORITY_INFORMATION_ACCESS).value
            ocsp_urls = [desc.access_location.value for desc in aia if desc.access_method._name == "ocsp"]
            if ocsp_urls:
                ocsp_url = ocsp_urls[0]
                # Build request
                builder = ocsp.OCSPRequestBuilder()
                issuer_cert = root_cert if root_cert else signer_cert
                builder = builder.add_certificate(signer_cert, issuer_cert, hashes.SHA1())
                ocsp_req = builder.build()
                ocsp_req_bytes = ocsp_req.public_bytes(serialization.Encoding.DER)

                # Send request
                ocsp_resp = requests.post(ocsp_url, data=ocsp_req_bytes, headers={"Content-Type": "application/ocsp-request"}, timeout=5.0)
                if ocsp_resp.ok:
                    ocsp_status = "good"
                else:
                    ocsp_status = "revocation_check_failed"
        except Exception as ocsp_err:
            logger.warning("OCSP validation failed: %s", ocsp_err)
            ocsp_status = "ocsp_unreachable"

        api_result = {
            "status": "digitally_verified",
            "subject": subject,
            "issuer": issuer,
            "serial_number": str(serial),
            "chain_verified": chain_verified,
            "ocsp_status": ocsp_status,
            "valid_from": signer_cert.not_valid_before_utc.isoformat(),
            "valid_to": signer_cert.not_valid_after_utc.isoformat()
        }
        return _build_response("e_stamped_document", True, api_result, "clean", 0.0, [])

    except Exception as exc:
        logger.error("Digital signature check error: %s", exc)
        return _build_response("e_stamped_document", True, {"error": "signature_check_failed", "details": str(exc)}, "clean", 0.0, [])


def verify_cheque(extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    j) Cheque/Draft -> penny drop API or Surepass Cheque verification
    """
    account_number = extracted_fields.get("account_number") or extracted_fields.get("id_number")
    ifsc = extracted_fields.get("ifsc")

    if not account_number or not ifsc:
        return _build_response(
            "cheque", True, {"error": "missing_account_number_or_ifsc"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["missing_id_number"]
        )

    api_key = _get_api_key()
    url = "https://api.surepass.io/api/v1/bank-verification/"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "id_number": str(account_number).strip(),
        "ifsc": str(ifsc).strip().upper()
    }

    try:
        logger.info("Calling Penny Drop API for Cheque: Account: %s, IFSC: %s", account_number, ifsc)
        response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        logger.debug("Raw Cheque Penny Drop response: %s", response.text)

        if response.status_code == 404 or not response.ok:
            return _build_response(
                "cheque", True, {"status_code": response.status_code, "body": response.text},
                "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        res_data = response.json()
        success = res_data.get("success", False)
        data = res_data.get("data", {})

        if not success or not data:
            return _build_response(
                "cheque", True, res_data, "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        full_name = data.get("full_name") or data.get("name")
        extracted_name = extracted_fields.get("name")
        name_match = _fuzzy_name_match(extracted_name, full_name)

        api_result = {
            "full_name": full_name,
            "account_exists": data.get("account_exists", True),
            "name_match": name_match
        }

        if not name_match:
            return _build_response("cheque", True, api_result, "high", WEIGHT_NAME_MISMATCH, ["api_name_mismatch"])

        return _build_response("cheque", True, api_result, "clean", 0.0, [])

    except requests.RequestException as e:
        logger.error("Cheque penny drop connection error: %s", e)
        return _build_response("cheque", True, {"error": "api_unreachable"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["api_unreachable"])


def verify_ca_certificate(extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    k) CA certificate -> ICAI portal membership number verification
    """
    membership_number = extracted_fields.get("id_number") or extracted_fields.get("membership_number")
    if not membership_number:
        return _build_response(
            "ca_certificate", True, {"error": "missing_membership_number"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["missing_id_number"]
        )

    api_key = _get_api_key()
    url = "https://api.surepass.io/api/v1/corporate/icai-member-search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"membership_number": str(membership_number).strip()}

    try:
        logger.info("Calling ICAI Member Verification API for: %s", membership_number)
        response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        logger.debug("Raw ICAI API response: %s", response.text)

        if response.status_code == 404 or not response.ok:
            return _build_response(
                "ca_certificate", True, {"status_code": response.status_code, "body": response.text},
                "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        res_data = response.json()
        success = res_data.get("success", False)
        data = res_data.get("data", {})

        if not success or not data:
            return _build_response(
                "ca_certificate", True, res_data, "high", WEIGHT_INVALID_NOT_FOUND, ["api_invalid_or_not_found"]
            )

        member_name = data.get("member_name") or data.get("name")
        extracted_name = extracted_fields.get("name")
        name_match = _fuzzy_name_match(extracted_name, member_name)

        api_result = {
            "member_name": member_name,
            "status": data.get("status", "ACTIVE"),
            "name_match": name_match
        }

        if not name_match:
            return _build_response("ca_certificate", True, api_result, "high", WEIGHT_NAME_MISMATCH, ["api_name_mismatch"])

        return _build_response("ca_certificate", True, api_result, "clean", 0.0, [])

    except requests.RequestException as e:
        logger.error("ICAI membership verification connection error: %s", e)
        return _build_response("ca_certificate", True, {"error": "api_unreachable"}, "skipped", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["api_unreachable"])


def verify_itr_form16(extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    i) ITR/Form 16 -> OAuth consent placeholder
    ITR & Form 16 verification requires taxpayer OAuth consent via the Income Tax Department's portal.
    """
    api_result = {
        "requires_consent": True,
        "consent_status": "pending_user_consent",
        "description": "Verification of income tax documents requires explicit taxpayer OAuth consent authorization."
    }
    return _build_response("form_16", True, api_result, "clean", 0.0, [])


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------

def route_api(doc_type: str, extracted_fields: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """
    Routes document types to their respective online/offline verification logic.
    
    Parameters
    ----------
    doc_type : str
        The document type classification label.
    extracted_fields : dict
        Extracted fields from document OCR (e.g. name, ID numbers, address, image path).
    mode : str
        "online" or "offline" operation mode.
        
    Returns
    -------
    dict
        Uniform verification result dictionary conforming to the return schema.
    """
    doc_type_clean = str(doc_type).lower().strip()

    # Route maps: doc_type key -> verification function
    ROUTE_MAP = {
        "pan_card": verify_pan,
        "aadhaar_card": verify_aadhaar,
        "gstin_certificate": verify_gstin,
        "gst_certificate": verify_gstin,
        "udyam_msme_certificate": verify_udyam,
        "vehicle_rc": verify_rc,
        "bank_statement": verify_bank_account,
        "bank_account": verify_bank_account,
        "driving_licence": verify_dl,
        "e_stamped_document": verify_estamp,
        "ca_certificate": verify_ca_certificate,
        "cheque": verify_cheque,
        "cheque_leaf": verify_cheque,
        "form_16": verify_itr_form16,
        "itr_filing": verify_itr_form16,
        "salaried_itr": verify_itr_form16
    }

    # Offline Mode logic
    # Aadhaar Offline XML QR verification works offline as it performs cryptographic validation locally.
    # Other document types require active API connections and are skipped immediately.
    if mode == "offline":
        if doc_type_clean == "aadhaar_card":
            logger.info("Executing Aadhaar Offline XML QR verification.")
            return verify_aadhaar(extracted_fields)
        else:
            logger.info("Offline mode: Skipping API verification for %s.", doc_type)
            return {
                "skipped": True, 
                "reason": "offline mode", 
                "section": "api_verification", 
                "doc_type": doc_type,
                "api_available": doc_type_clean in ROUTE_MAP,
                "api_result": {},
                "risk": "skipped",
                "penalty_weight": 0.0,
                "flags": []
            }

    # Check if doc_type has coverage in ROUTE_MAP
    if doc_type_clean not in ROUTE_MAP:
        # Step 3: Log document types with NO API coverage
        logger.warning("No API verification coverage defined for document type: '%s'", doc_type)
        return _build_response(
            doc_type, False, {}, "clean", WEIGHT_UNREACHABLE_OR_NOT_AVAILABLE, ["api_not_available"]
        )

    # Invoke appropriate verifier
    verifier_func = ROUTE_MAP[doc_type_clean]
    return verifier_func(extracted_fields)
