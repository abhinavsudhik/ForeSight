import os
import sys
import re
import zlib
import base64
import logging
import cv2
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional

# Ensure Homebrew's lib path is added to DYLD_LIBRARY_PATH on macOS for pyzbar to locate libzbar.dylib
if sys.platform == 'darwin':
    _brew_lib = '/opt/homebrew/lib'
    if os.path.exists(_brew_lib):
        _current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
        _dyld_paths = _current_dyld.split(':') if _current_dyld else []
        if _brew_lib not in _dyld_paths:
            _dyld_paths.insert(0, _brew_lib)
            os.environ['DYLD_LIBRARY_PATH'] = ':'.join(_dyld_paths)

logger = logging.getLogger(__name__)

def extract_qr_from_image(image_path: str) -> Optional[str]:
    """
    Extracts QR code content from the Aadhaar card scanned image.
    Uses OpenCV to load the image and pyzbar to find and decode all QR codes.
    If no QR is found initially, retries on a 2x upscaled version.
    Returns the raw QR data string (decoded using 'latin-1' to preserve binary bytes losslessly),
    or None if no QR code is detected.
    """
    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        logger.error("pyzbar is not installed. QR extraction cannot proceed.")
        return None

    if not os.path.exists(image_path):
        logger.error(f"Image path does not exist: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image via OpenCV from path: {image_path}")
        return None

    # Try detecting QR on the full image first
    decoded_objs = decode(img)
    if not decoded_objs:
        logger.info("No QR code detected on original image. Retrying on a 2x upscaled version...")
        # Retry on a 2x upscaled version (cubic interpolation for clarity)
        h, w = img.shape[:2]
        upscaled_img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        decoded_objs = decode(upscaled_img)

    if not decoded_objs:
        logger.warning("No QR code detected — may be older Aadhaar card pre-2015 or low DPI scan")
        return None

    # Get the raw bytes from the first detected QR code
    qr_bytes = decoded_objs[0].data
    
    # Return as string using latin-1 encoding to preserve all byte values (0-255) 1:1
    return qr_bytes.decode('latin-1')


def parse_aadhaar_xml(qr_data: str) -> dict:
    """
    Parses the extracted Aadhaar QR data.
    Supports:
      a) Secure QR format (post-2017): compressed via zlib, binary fields separated by 0xFF.
      b) Old XML format (pre-2017): plain XML text.
    Returns a dictionary of parsed fields, including photo_bytes and signature_bytes if present.
    """
    if not qr_data:
        return {}

    # Convert latin-1 string back to original raw bytes
    qr_bytes = qr_data.encode('latin-1')

    # Try zlib decompression to check for post-2017 Secure QR format
    try:
        # Try default decompression; handle potential zip wrapper differences
        try:
            decompressed = zlib.decompress(qr_bytes)
        except zlib.error:
            try:
                decompressed = zlib.decompress(qr_bytes, 16 + zlib.MAX_WBITS)
            except zlib.error:
                decompressed = zlib.decompress(qr_bytes, -zlib.MAX_WBITS)

        # Decompressed content starts with version byte
        version = decompressed[0]
        raw_fields = decompressed.split(b'\xff')

        # Robust check to see if the version byte is isolated in raw_fields[0] or prepended
        if len(raw_fields[0]) == 1 and raw_fields[0][0] in (1, 2):
            version = raw_fields[0][0]
            fields = raw_fields[1:]
        elif len(raw_fields[0]) > 1 and raw_fields[0][0] in (1, 2):
            version = raw_fields[0][0]
            fields = [raw_fields[0][1:]] + raw_fields[1:]
        else:
            version = raw_fields[0][0] if len(raw_fields[0]) > 0 else 2
            fields = raw_fields[1:]

        # For version 1 and 2, demographic fields in sequence
        field_names = [
            "email_mobile_present_bit_indicator",
            "reference_id",
            "name",
            "dob",
            "gender",
            "co",
            "dist",
            "landmark",
            "house",
            "loc",
            "pc",
            "po",
            "state",
            "street",
            "sub_dist",
            "vtc",
            "last_4_digits_mobile",
            "email_hash"
        ]

        parsed_fields = {
            "format": "secure_qr",
            "version": version
        }

        # Populate demographic text fields
        for idx, name in enumerate(field_names):
            if idx < len(fields):
                parsed_fields[name] = fields[idx].decode("ISO-8859-1", errors="ignore").strip()
            else:
                parsed_fields[name] = ""

        # Extract last 4 digits of Aadhaar (first 4 chars of reference_id)
        ref_id = parsed_fields.get("reference_id", "")
        parsed_fields["aadhaar_last_4_digits"] = ref_id[:4] if len(ref_id) >= 4 else ref_id

        # The signature bytes are in the very last field
        if len(fields) >= 1:
            parsed_fields["signature_bytes"] = fields[-1]
        else:
            parsed_fields["signature_bytes"] = b""

        # The timestamp is the second-to-last field
        if len(fields) >= 2:
            parsed_fields["timestamp"] = fields[-2].decode("ISO-8859-1", errors="ignore").strip()
        else:
            parsed_fields["timestamp"] = ""

        # The image_bytes (JPEG photo) is the third-to-last field (second-to-last before signature)
        if len(fields) >= 3:
            parsed_fields["photo_bytes"] = fields[-3]
        else:
            parsed_fields["photo_bytes"] = b""

        return parsed_fields

    except Exception:
        # zlib failed; fallback to parsing as plain XML text (older Aadhaar pre-2017)
        try:
            xml_text = qr_bytes.decode('utf-8', errors='ignore').strip()
            # Search for XML root boundary
            xml_match = re.search(r"<OfflinePaperlessKyc.*?>.*?</OfflinePaperlessKyc>", xml_text, re.DOTALL | re.IGNORECASE)
            if xml_match:
                xml_str = xml_match.group(0)
            else:
                xml_str = xml_text

            root = ET.fromstring(xml_str)
            poi = root.find(".//Poi")
            poa = root.find(".//Poa")

            parsed_fields = {
                "format": "xml",
                "uid": root.get("uid") if root is not None else "",
                "name": poi.get("name") if poi is not None else "",
                "dob": poi.get("dob") if poi is not None else "",
                "gender": poi.get("gender") if poi is not None else "",
                "co": poa.get("co") if poa is not None else "",
                "house": poa.get("house") if poa is not None else "",
                "street": poa.get("street") if poa is not None else "",
                "lm": poa.get("lm") if poa is not None else "",
                "loc": poa.get("loc") if poa is not None else "",
                "vtc": poa.get("vtc") if poa is not None else "",
                "subdist": poa.get("subdist") if poa is not None else "",
                "dist": poa.get("dist") if poa is not None else "",
                "state": poa.get("state") if poa is not None else "",
                "pc": poa.get("pc") if poa is not None else "",
                "country": poa.get("country") if poa is not None else "India",
                "raw_xml": xml_str
            }

            # Locate XML signature if present
            sig_elem = root.find(".//SignatureValue")
            if sig_elem is None:
                sig_elem = root.find(".//{http://www.w3.org/2000/09/xmldsig#}SignatureValue")
            
            if sig_elem is not None and sig_elem.text:
                parsed_fields["signature_bytes"] = base64.b64decode(sig_elem.text.strip())
            else:
                parsed_fields["signature_bytes"] = b""

            return parsed_fields

        except Exception as e:
            logger.error(f"Failed to parse Aadhaar QR data as XML: {e}")
            return {}


def verify_signature(qr_data: str, public_key_path: str) -> dict:
    """
    Verifies the cryptographic RSA signature of the Aadhaar QR payload against the local public key certificate.
    Returns:
      {"valid": True} if signature is valid.
      {"valid": False, "reason": str} if signature is invalid or validation fails.
      {"valid": None, "reason": str} if verification cannot be performed (e.g. xmlsec1 not installed for XML format).
    """
    if not qr_data:
        return {"valid": False, "reason": "Empty QR data payload"}

    qr_bytes = qr_data.encode('latin-1')

    # Decompress QR data if zlib-compressed
    is_secure_qr = False
    decompressed = None
    try:
        try:
            decompressed = zlib.decompress(qr_bytes)
            is_secure_qr = True
        except zlib.error:
            try:
                decompressed = zlib.decompress(qr_bytes, 16 + zlib.MAX_WBITS)
                is_secure_qr = True
            except zlib.error:
                decompressed = zlib.decompress(qr_bytes, -zlib.MAX_WBITS)
                is_secure_qr = True
    except Exception:
        pass

    if is_secure_qr and decompressed:
        # Verify Secure QR (binary format) signature
        try:
            from cryptography.x509 import load_pem_x509_certificate, load_der_x509_certificate
            with open(public_key_path, 'rb') as f:
                cert_bytes = f.read()
            try:
                cert = load_pem_x509_certificate(cert_bytes)
            except Exception:
                cert = load_der_x509_certificate(cert_bytes)
            public_key = cert.public_key()
        except Exception as e:
            return {"valid": False, "reason": f"Failed to load certificate / public key from {public_key_path}: {e}"}

        try:
            raw_fields = decompressed.split(b'\xff')
            if len(raw_fields) < 2:
                return {"valid": False, "reason": "Malformed secure QR structure (fewer than 2 fields)"}

            signature_bytes = raw_fields[-1]
            # Reconstruct signed data: join all fields except the last with 0xFF delimiter
            signed_data = b'\xff'.join(raw_fields[:-1])

            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.hazmat.primitives import hashes

            # Try verifying using standard RSA PKCS1v15 padding
            try:
                public_key.verify(
                    signature_bytes,
                    signed_data,
                    padding.PKCS1v15(),
                    hashes.SHA256()
                )
                return {"valid": True}
            except Exception as pkcs_err:
                # Try verifying using RSA PSS padding as fallback
                try:
                    public_key.verify(
                        signature_bytes,
                        signed_data,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
                    return {"valid": True}
                except Exception as pss_err:
                    return {"valid": False, "reason": f"Signature mismatch. PKCS1v15: {pkcs_err}; PSS: {pss_err}"}

        except Exception as e:
            return {"valid": False, "reason": f"RSA signature verification error: {e}"}

    else:
        # Verify old XML format
        try:
            import xmlsec
        except ImportError:
            return {"valid": None, "reason": "xmlsec1 not installed, manual verification required"}

        try:
            xml_text = qr_bytes.decode('utf-8', errors='ignore').strip()
            xml_match = re.search(r"<OfflinePaperlessKyc.*?>.*?</OfflinePaperlessKyc>", xml_text, re.DOTALL | re.IGNORECASE)
            if xml_match:
                xml_str = xml_match.group(0)
            else:
                xml_str = xml_text

            root = ET.fromstring(xml_str)
            signature_node = xmlsec.tree.find_node(root, xmlsec.Node.SIGNATURE)
            if signature_node is None:
                return {"valid": False, "reason": "Signature element not found in XML"}

            # Load the UIDAI public key certificate into xmlsec context
            ctx = xmlsec.SignatureContext()
            try:
                key = xmlsec.Key.from_file(public_key_path, xmlsec.KeyFormat.CERT_PEM)
            except Exception:
                try:
                    key = xmlsec.Key.from_file(public_key_path, xmlsec.KeyFormat.CERT_DER)
                except Exception as xmlsec_load_err:
                    return {"valid": False, "reason": f"Failed to load certificate into xmlsec: {xmlsec_load_err}"}
            ctx.key = key

            # Verify the XML signature
            ctx.verify(signature_node)
            return {"valid": True}
        except Exception as e:
            return {"valid": False, "reason": f"XMLSec signature verification failed: {e}"}


def extract_photo(parsed_fields: dict, output_path: Optional[str] = None) -> Optional[bytes]:
    """
    Extracts the JPEG photo bytes from the parsed fields.
    Saves to output_path if provided, and returns the raw photo bytes.
    """
    if not parsed_fields:
        return None

    photo_bytes = parsed_fields.get("photo_bytes")
    if not photo_bytes:
        return None

    if output_path:
        try:
            dir_name = os.path.dirname(os.path.abspath(output_path))
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(photo_bytes)
            logger.info(f"Photo successfully written to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write photo bytes to disk: {e}")

    return photo_bytes


def run_aadhaar_check(
    image_path: str,
    public_key_path: str = "keys/uidai_public.cer",
    extracted_fields: Optional[dict] = None
) -> dict:
    """
    Main entry point that extracts QR, parses demographics, validates the UIDAI signature,
    cross-checks fields if OCR results are provided, and evaluates risk penalties.
    """
    result = {
        "section": "aadhaar_verification",
        "qr_found": False,
        "signature_valid": None,
        "name_match": None,
        "dob_match": None,
        "parsed_fields": {},
        "risk": "clean",
        "penalty_weight": 0.0,
        "flags": []
    }

    # 1. Extract QR from scanned image
    qr_data = extract_qr_from_image(image_path)
    if qr_data is None:
        result["risk"] = "medium"
        result["penalty_weight"] = 0.40  # Medium — older cards have no QR, not proof of fake
        result["flags"] = ["no_qr_found"]
        result["reason"] = "No QR — cannot verify authenticity. May be pre-2015 card."
        return result

    result["qr_found"] = True

    # 2. Parse Aadhaar QR content
    parsed_fields = parse_aadhaar_xml(qr_data)
    result["parsed_fields"] = {k: v for k, v in parsed_fields.items() if k not in ("photo_bytes", "signature_bytes", "raw_xml")}

    # 3. Verify digital signature
    sig_result = verify_signature(qr_data, public_key_path)
    sig_valid = sig_result.get("valid")
    result["signature_valid"] = sig_valid

    # Handle INVALID signature
    if sig_valid is False:
        result["risk"] = "high"
        result["penalty_weight"] = 0.95  # Near-certain fake — UIDAI signature is RSA-2048, cryptographically unbreakable
        result["flags"] = ["signature_invalid"]
        result["reason"] = f"QR found but signature INVALID: {sig_result.get('reason', 'Signature validation failed')}"
        return result

    # 4. Cross-check fields (if signature is valid or unverified)
    name_match = None
    dob_match = None

    if extracted_fields:
        # Cross-check Name
        ocr_name = extracted_fields.get("name")
        parsed_name = parsed_fields.get("name")
        if ocr_name and parsed_name:
            ocr_name_clean = "".join(str(ocr_name).lower().split())
            parsed_name_clean = "".join(str(parsed_name).lower().split())
            name_match = (ocr_name_clean == parsed_name_clean)
            result["name_match"] = name_match

        # Cross-check DOB
        ocr_dob = extracted_fields.get("dob")
        parsed_dob = parsed_fields.get("dob")
        if ocr_dob and parsed_dob:
            ocr_dob_clean = "".join(str(ocr_dob).lower().split())
            parsed_dob_clean = "".join(str(parsed_dob).lower().split())
            dob_match = (ocr_dob_clean == parsed_dob_clean)
            result["dob_match"] = dob_match

    # Evaluate final risk score based on checks
    if sig_valid is True:
        if name_match is False:
            result["risk"] = "high"
            result["penalty_weight"] = 0.70  # Could be name alias or OCR error — high suspicion but not certain
            result["flags"] = ["name_mismatch"]
            result["reason"] = "Signature valid but name mismatch between QR and document text."
        else:
            result["risk"] = "clean"
            result["penalty_weight"] = 0.0  # Clean verification
            result["reason"] = "Signature verified and document details match."
    else:
        # Signature is unverified (sig_valid is None) because xmlsec1 is missing
        result["flags"] = ["xmlsec1_missing_signature_unverified"]
        if name_match is False:
            result["risk"] = "high"
            result["penalty_weight"] = 0.70
            result["reason"] = "Signature could not be verified (xmlsec1 missing) and name mismatch detected."
        else:
            result["risk"] = "medium"
            result["penalty_weight"] = 0.0
            result["reason"] = "Signature could not be verified (xmlsec1 missing). Demographics match or not provided."

    return result

# ---------------------------------------------------------------------------
# Installation and Requirements:
# ---------------------------------------------------------------------------
# To use this module, make sure you install the following dependencies:
#
# 1. Install pyzbar (requires zbar shared library on the host system):
#    Mac OS:     brew install zbar
#    Ubuntu/Deb: sudo apt-get install libzbar0
#    Python:     pip install pyzbar
#
# 2. Install OpenCV and Cryptography:
#    pip install opencv-python-headless cryptography
#
# 3. (Optional) For older XML signature verification:
#    Linux:      sudo apt-get install libxml2-dev libxmlsec1-dev libxmlsec1-openssl
#    Mac OS:     brew install xmlsec1
#    Python:     pip install xmlsec
# ---------------------------------------------------------------------------
