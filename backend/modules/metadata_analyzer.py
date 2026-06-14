"""
Metadata Analyzer for ForeSight.

Takes a PDF file path, extracts metadata using fitz (PyMuPDF),
and compares it against the document's own content dates.

Fields extracted
────────────────
- Creation Date
- Modification Date
- Author
- Producer (software used to create/edit)

Flagging rules
──────────────
Condition                                                       → Severity
Modification date is after the document's issue date            → High
Producer software is a known PDF editor                         → Medium
Author field is empty or generic ("Unknown", "User")            → Low
Creation and modification dates identical but suspiciously recent → Low
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Union, Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Known PDF editors that could indicate document tampering
_SUSPICIOUS_PRODUCERS = {
    "adobe acrobat",
    "smallpdf",
    "ilovepdf",
    "foxit phantompdf",
    "foxit reader",
    "nitro pdf",
    "pdf-xchange",
    "pdfxchange",
    "sejda",
    "pdfelement",
    "pdf expert",
    "inkscape",
    "gimp",
    "photoshop",
    "canva",
}

# Generic / empty author values that raise concern
_GENERIC_AUTHORS = {
    "unknown",
    "user",
    "admin",
    "administrator",
    "owner",
    "",
    "default",
    "test",
    "temp",
}

# Documents created/modified within this many days are "suspiciously recent"
_RECENT_DAYS_THRESHOLD = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pdf_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse a PDF metadata date string into a datetime object.

    PDF dates follow the format: D:YYYYMMDDHHmmSS+HH'mm'
    Some PDFs use simpler ISO-like formats.
    """
    if not date_str:
        return None

    date_str = str(date_str).strip()

    # Remove PDF date prefix "D:"
    if date_str.startswith("D:"):
        date_str = date_str[2:]

    # Remove timezone info (e.g. +05'30', Z, +0530)
    for tz_pattern in ["+", "-", "Z"]:
        idx = date_str.find(tz_pattern, 8)  # search after YYYYMMDD
        if idx > 0:
            date_str = date_str[:idx]

    # Remove any trailing apostrophes
    date_str = date_str.replace("'", "")

    # Try multiple date formats
    formats = [
        "%Y%m%d%H%M%S",
        "%Y%m%d%H%M",
        "%Y%m%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str[:len(fmt.replace("%", "").replace(
                "Y", "0000").replace("m", "00").replace("d", "00").replace(
                "H", "00").replace("M", "00").replace("S", "00"))],
                fmt
            )
        except (ValueError, IndexError):
            continue

    # Brute-force attempt: take the first 14 chars as YYYYMMDDHHmmSS
    try:
        return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
    except (ValueError, IndexError):
        pass

    try:
        return datetime.strptime(date_str[:8], "%Y%m%d")
    except (ValueError, IndexError):
        pass

    logger.debug("Could not parse PDF date: '%s'", date_str)
    return None


def _is_suspicious_producer(producer: str) -> bool:
    """Check if the producer software is a known PDF editor."""
    if not producer:
        return False
    producer_lower = producer.lower().strip()
    return any(editor in producer_lower for editor in _SUSPICIOUS_PRODUCERS)


def _is_generic_author(author: str) -> bool:
    """Check if the author field is empty or generic."""
    if not author:
        return True
    return author.lower().strip() in _GENERIC_AUTHORS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_metadata(
    pdf_path: str,
    document_issue_date: Optional[Union[str, dict]] = None,
) -> dict:
    """
    Extract PDF metadata and flag suspicious patterns.

    Parameters
    ----------
    pdf_path : str
        Absolute path to the PDF file.
    document_issue_date : str or None
        The issue date found within the document's content (from field
        extractor), used to compare against modification dates.
        Accepts common date formats (DD/MM/YYYY, YYYY-MM-DD, etc.).

    Returns
    -------
    dict
        {
            "metadata": dict    — raw extracted metadata fields,
            "flags": list[dict] — suspicious metadata flags,
            "summary": str      — human-readable summary
        }
    """
    flags: list[dict] = []
    metadata: dict = {
        "creation_date": None,
        "modification_date": None,
        "author": None,
        "producer": None,
        "file_size_kb": None,
        "page_count": None,
    }

    # --- Validate file path ---
    if not pdf_path or not os.path.exists(pdf_path):
        logger.warning("PDF file not found: '%s'", pdf_path)
        return {
            "metadata": metadata,
            "flags": [],
            "summary": "PDF file not found — metadata analysis skipped.",
        }

    # --- Extract metadata using PyMuPDF ---
    try:
        doc = fitz.open(pdf_path)
        pdf_meta = doc.metadata or {}
        page_count = len(doc)
        doc.close()
    except Exception as exc:
        logger.error("Failed to read PDF metadata from '%s': %s", pdf_path, exc)
        return {
            "metadata": metadata,
            "flags": [],
            "summary": f"Failed to read PDF metadata: {exc}",
        }

    # --- Populate metadata dict ---
    raw_creation = pdf_meta.get("creationDate", "")
    raw_modification = pdf_meta.get("modDate", "")
    raw_author = pdf_meta.get("author", "") or ""
    raw_producer = pdf_meta.get("producer", "") or ""

    creation_dt = _parse_pdf_date(raw_creation)
    modification_dt = _parse_pdf_date(raw_modification)

    file_size_kb = round(os.path.getsize(pdf_path) / 1024, 1)

    metadata = {
        "creation_date": creation_dt.strftime("%Y-%m-%d %H:%M:%S") if creation_dt else raw_creation or None,
        "modification_date": modification_dt.strftime("%Y-%m-%d %H:%M:%S") if modification_dt else raw_modification or None,
        "author": raw_author or None,
        "producer": raw_producer or None,
        "file_size_kb": file_size_kb,
        "page_count": page_count,
    }

    logger.info(
        "PDF metadata extracted — author: '%s', producer: '%s', "
        "created: %s, modified: %s",
        raw_author, raw_producer,
        metadata["creation_date"], metadata["modification_date"],
    )

    # --- Flag 1: Modification date after the document's issue date ---
    if modification_dt and document_issue_date:
        # Parse the document issue date
        issue_dt = _parse_document_date(document_issue_date)
        
        display_issue_date = document_issue_date
        if isinstance(document_issue_date, dict):
            display_issue_date = (
                document_issue_date.get("issue date")
                or document_issue_date.get("issue_date")
                or document_issue_date.get("date")
                or document_issue_date.get("dob")
                or "Unknown"
            )

        if issue_dt and modification_dt > issue_dt:
            flags.append({
                "check": "metadata_analysis",
                "severity": "high",
                "message": (
                    f"PDF was modified ({metadata['modification_date']}) after "
                    f"the document's issue date ({display_issue_date}) — "
                    f"possible post-issuance tampering"
                ),
                "evidence": {
                    "modification_date": metadata["modification_date"],
                    "document_issue_date": display_issue_date,
                    "days_after": (modification_dt - issue_dt).days,
                },
            })

    # --- Flag 2: Suspicious producer software ---
    if _is_suspicious_producer(raw_producer):
        flags.append({
            "check": "metadata_analysis",
            "severity": "medium",
            "message": (
                f"PDF was created/edited with '{raw_producer}' — "
                f"a known PDF editing tool that could be used for tampering"
            ),
            "evidence": {
                "producer": raw_producer,
            },
        })

    # --- Flag 3: Generic or empty author ---
    if _is_generic_author(raw_author):
        author_display = f"'{raw_author}'" if raw_author else "empty"
        flags.append({
            "check": "metadata_analysis",
            "severity": "low",
            "message": (
                f"Author field is {author_display} — "
                f"legitimate documents usually have an identifiable author or organization"
            ),
            "evidence": {
                "author": raw_author or "(empty)",
            },
        })

    # --- Flag 4: Creation == Modification and suspiciously recent ---
    if creation_dt and modification_dt:
        dates_identical = abs((creation_dt - modification_dt).total_seconds()) < 60
        is_recent = (datetime.now() - creation_dt) < timedelta(
            days=_RECENT_DAYS_THRESHOLD
        )

        if dates_identical and is_recent:
            flags.append({
                "check": "metadata_analysis",
                "severity": "low",
                "message": (
                    f"Creation and modification dates are identical "
                    f"({metadata['creation_date']}) and suspiciously recent "
                    f"(within {_RECENT_DAYS_THRESHOLD} days) — could indicate "
                    f"a freshly fabricated document"
                ),
                "evidence": {
                    "creation_date": metadata["creation_date"],
                    "modification_date": metadata["modification_date"],
                    "days_old": (datetime.now() - creation_dt).days,
                },
            })

    logger.info("Metadata analysis complete — %d flag(s)", len(flags))

    # --- Build summary ---
    if flags:
        summary = (
            f"Metadata analysis found {len(flags)} concern(s): "
            + "; ".join(f["severity"].upper() + " — " + f["message"]
                        for f in flags)
        )
    else:
        summary = "No suspicious metadata patterns detected."

    return {
        "metadata": metadata,
        "flags": flags,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Internal date parser for document issue dates
# ---------------------------------------------------------------------------

def _parse_document_date(date_val: Union[str, dict, None]) -> Optional[datetime]:
    """
    Parse a date string or dictionary containing issue date from the field extractor into a datetime.
    Supports common Indian and international date formats.
    """
    if not date_val:
        return None

    if isinstance(date_val, dict):
        date_str = (
            date_val.get("issue date")
            or date_val.get("issue_date")
            or date_val.get("date")
            or date_val.get("dob")
        )
    else:
        date_str = date_val

    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip()
    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d/%m/%y",
        "%d-%m-%y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    logger.debug("Could not parse document date: '%s'", date_str)
    return None
