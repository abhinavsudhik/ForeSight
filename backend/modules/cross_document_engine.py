"""
Cross-Document Consistency Engine for ForeSight.

Accept a list of extracted field dictionaries (one per document) and
return a list of inconsistency flags.

Three checks
─────────────
1. **Name Consistency**
   Collect all "name" / "owner_name" / "account_holder" / "seller_name" /
   "buyer_name" fields.  Compare every pair using RapidFuzz (fuzz.ratio).
   If similarity < 85 % → flag as "Name mismatch".

2. **Property ID Consistency**
   Collect all "property_id" / "survey_number" fields.
   Compare across documents — exact match expected.
   Any mismatch → flag as "Property ID mismatch".

3. **Timeline Consistency**
   Collect all dates across documents.
   • valuation date should NOT be after sale deed date
   • land record issue date should NOT be after sale deed date
   Flag impossible timelines.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NAME_SIMILARITY_THRESHOLD = 85  # percent

# Field keys that represent a person's name across document types
_NAME_KEYS = frozenset({
    "name",
    "owner_name",
    "account_holder",
    "seller_name",
    "buyer_name",
})

# Field keys that represent property identifiers
_PROPERTY_ID_KEYS = frozenset({
    "property_id",
    "survey_number",
})

# Date formats to try when parsing extracted date strings
_DATE_FORMATS = [
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


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class InconsistencyFlag:
    """A single cross-document inconsistency."""

    check: str
    """Type of check: 'name_consistency', 'property_id_consistency',
    or 'timeline_consistency'."""

    severity: str
    """'high', 'medium', or 'low'."""

    evidence: dict
    """Supporting evidence — contents depend on the check type."""

    message: str
    """Human-readable explanation of the inconsistency."""

    similarity: Optional[int] = None
    """Fuzzy similarity score (only for name checks)."""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _collect_values(
    documents: list[dict],
    keys: frozenset[str],
) -> list[tuple[str, str, str]]:
    """
    Walk through every document dict and collect (doc_label, field_key, value)
    triples for the requested field keys.

    Parameters
    ----------
    documents : list[dict]
        Each dict has at least a ``"document_type"`` key and a ``"fields"``
        sub-dict produced by the field extractor.
    keys : frozenset[str]
        The field names to collect.

    Returns
    -------
    list[tuple[str, str, str]]
        (doc_label, field_key, value) — only entries with non-empty values.
    """
    results: list[tuple[str, str, str]] = []
    for idx, doc in enumerate(documents):
        doc_type = doc.get("document_type", f"doc_{idx + 1}")
        fields = doc.get("fields", {})
        for key in keys:
            value = fields.get(key)
            if value and str(value).strip():
                label = f"{doc_type} (doc {idx + 1})"
                results.append((label, key, str(value).strip()))
    return results


def _parse_date(date_str: str) -> Optional[datetime]:
    """
    Attempt to parse a date string using common Indian / international
    date formats.  Returns ``None`` on failure.
    """
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    logger.debug("Could not parse date: '%s'", date_str)
    return None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_name_consistency(
    documents: list[dict],
) -> list[InconsistencyFlag]:
    """
    Compare every pair of name-like fields across documents using
    RapidFuzz ``fuzz.ratio``.  Flag pairs below the similarity threshold.
    """
    entries = _collect_values(documents, _NAME_KEYS)
    if len(entries) < 2:
        return []

    flags: list[InconsistencyFlag] = []
    seen: set[tuple[int, int]] = set()

    for i, (label_a, key_a, val_a) in enumerate(entries):
        for j, (label_b, key_b, val_b) in enumerate(entries):
            if i >= j:
                continue
            pair = (i, j)
            if pair in seen:
                continue
            seen.add(pair)

            score = fuzz.ratio(val_a.lower(), val_b.lower())
            if score < _NAME_SIMILARITY_THRESHOLD:
                flags.append(InconsistencyFlag(
                    check="name_consistency",
                    severity="high",
                    evidence={
                        label_a: val_a,
                        label_b: val_b,
                        "field_a": key_a,
                        "field_b": key_b,
                    },
                    similarity=int(score),
                    message=(
                        f"Owner name mismatch across {label_a} and {label_b}: "
                        f"'{val_a}' vs '{val_b}' (similarity {int(score)}%)"
                    ),
                ))

    return flags


def _check_property_id_consistency(
    documents: list[dict],
) -> list[InconsistencyFlag]:
    """
    Collect property_id / survey_number values and ensure they match
    exactly across documents.  Any mismatch → flag.
    """
    entries = _collect_values(documents, _PROPERTY_ID_KEYS)
    if len(entries) < 2:
        return []

    flags: list[InconsistencyFlag] = []
    seen: set[tuple[int, int]] = set()

    for i, (label_a, key_a, val_a) in enumerate(entries):
        for j, (label_b, key_b, val_b) in enumerate(entries):
            if i >= j:
                continue
            pair = (i, j)
            if pair in seen:
                continue
            seen.add(pair)

            # Normalise for comparison (strip, upper-case)
            norm_a = val_a.strip().upper()
            norm_b = val_b.strip().upper()

            if norm_a != norm_b:
                flags.append(InconsistencyFlag(
                    check="property_id_consistency",
                    severity="high",
                    evidence={
                        label_a: val_a,
                        label_b: val_b,
                        "field_a": key_a,
                        "field_b": key_b,
                    },
                    message=(
                        f"Property ID mismatch across {label_a} and {label_b}: "
                        f"'{val_a}' vs '{val_b}'"
                    ),
                ))

    return flags


def _check_timeline_consistency(
    documents: list[dict],
) -> list[InconsistencyFlag]:
    """
    Validate that document dates form a logically possible timeline.

    Rules
    -----
    * Valuation date must NOT be **after** the sale deed date.
    * Land record date must NOT be **after** the sale deed date.
    """
    flags: list[InconsistencyFlag] = []

    # Collect dates keyed by document type
    dated_docs: list[dict] = []
    for idx, doc in enumerate(documents):
        doc_type = doc.get("document_type", f"doc_{idx + 1}")
        fields = doc.get("fields", {})

        # Grab the "date" field (Sale Deed, Valuation Report use "date")
        date_str = fields.get("date")
        # Also check "dob" for identity docs, but DOB is not timeline-relevant
        # across other doc types — skip it.

        parsed = _parse_date(date_str) if date_str else None
        if parsed:
            dated_docs.append({
                "doc_type": doc_type,
                "doc_index": idx,
                "date": parsed,
                "date_str": date_str,
            })

    if len(dated_docs) < 2:
        return []

    # Find the sale deed date (anchor)
    sale_deed_entry = None
    for entry in dated_docs:
        if entry["doc_type"] == "sale_deed":
            sale_deed_entry = entry
            break

    if sale_deed_entry is None:
        # No sale deed to anchor against — cannot check timeline
        return []

    sale_date = sale_deed_entry["date"]

    # Check each other document's date against the sale deed
    for entry in dated_docs:
        if entry["doc_type"] == "sale_deed":
            continue

        if entry["doc_type"] in ("valuation_report", "land_record"):
            if entry["date"] > sale_date:
                flags.append(InconsistencyFlag(
                    check="timeline_consistency",
                    severity="medium",
                    evidence={
                        entry["doc_type"]: entry["date_str"],
                        "sale_deed": sale_deed_entry["date_str"],
                    },
                    message=(
                        f"Impossible timeline: {entry['doc_type']} date "
                        f"({entry['date_str']}) is after the sale deed date "
                        f"({sale_deed_entry['date_str']})"
                    ),
                ))

    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cross_validate(
    documents: list[dict],
) -> list[InconsistencyFlag]:
    """
    Run all cross-document consistency checks and return a flat list of
    inconsistency flags.

    Parameters
    ----------
    documents : list[dict]
        Each element should be a dictionary with at least:

        * ``"document_type"``  — e.g. ``"sale_deed"``, ``"identity_proof"``
        * ``"fields"``         — the dict produced by
          :func:`field_extractor.extract_fields`

        Example::

            {
                "document_type": "identity_proof",
                "fields": {
                    "name": "Abel George Abraham",
                    "dob": "15/03/1985",
                    "address": "Kochi, Kerala",
                    "id_number": "1234 5678 9012",
                },
            }

    Returns
    -------
    list[InconsistencyFlag]
        A list of :class:`InconsistencyFlag` objects.  Empty if all checks
        pass.

    Examples
    --------
    >>> docs = [
    ...     {"document_type": "identity_proof",
    ...      "fields": {"name": "Abel George Abraham"}},
    ...     {"document_type": "land_record",
    ...      "fields": {"owner_name": "Aby George Abraham"}},
    ... ]
    >>> flags = cross_validate(docs)
    >>> flags[0].check
    'name_consistency'
    """
    if not documents or len(documents) < 2:
        logger.info(
            "Cross-validation skipped — need at least 2 documents (got %d)",
            len(documents) if documents else 0,
        )
        return []

    logger.info("Running cross-document validation on %d documents …", len(documents))

    all_flags: list[InconsistencyFlag] = []

    # 1. Name consistency
    name_flags = _check_name_consistency(documents)
    all_flags.extend(name_flags)
    logger.info("Name consistency check: %d flag(s)", len(name_flags))

    # 2. Property ID consistency
    pid_flags = _check_property_id_consistency(documents)
    all_flags.extend(pid_flags)
    logger.info("Property ID consistency check: %d flag(s)", len(pid_flags))

    # 3. Timeline consistency
    time_flags = _check_timeline_consistency(documents)
    all_flags.extend(time_flags)
    logger.info("Timeline consistency check: %d flag(s)", len(time_flags))

    logger.info(
        "Cross-validation complete — %d total inconsistency flag(s)", len(all_flags)
    )
    return all_flags
