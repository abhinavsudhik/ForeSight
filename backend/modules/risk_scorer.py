"""
Risk Scorer for ForeSight.

Collects all flags from every module and produces a single trust score
plus a risk level classification.

Scoring
───────
- Start with a base score of 100.
- Deduct points per flag based on severity:
    high   → −25
    medium → −15
    low    → −5
- Floor the score at 0.

Risk levels
───────────
Score ≥ 80  → "Low Risk"      (green)
Score 55–79 → "Medium Risk"   (orange)
Score 30–54 → "High Risk"     (red)
Score < 30  → "Critical Risk" (darkred)
"""

import logging
from dataclasses import dataclass, asdict
from typing import Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SEVERITY_PENALTIES = {
    "high": 25,
    "medium": 15,
    "low": 5,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_flag(flag) -> dict:
    """
    Convert a flag to a plain dict if it's a dataclass
    (e.g. InconsistencyFlag from cross_document_engine).
    """
    if isinstance(flag, dict):
        return flag
    # Dataclass instances have __dataclass_fields__
    if hasattr(flag, "__dataclass_fields__"):
        return asdict(flag)
    # Fallback: try to convert via __dict__
    if hasattr(flag, "__dict__"):
        return flag.__dict__
    return {"severity": "low", "message": str(flag), "check": "unknown"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_trust_score(
    cross_doc_flags: list,
    metadata_flags: list,
    financial_flags: list,
) -> dict:
    """
    Calculate an overall trust score from all module flags.

    Parameters
    ----------
    cross_doc_flags : list
        Flags from ``cross_document_engine.cross_validate()`` —
        can be ``InconsistencyFlag`` dataclass instances or plain dicts.
    metadata_flags : list
        Flags from ``metadata_analyzer.analyze_metadata()["flags"]``.
    financial_flags : list
        Flags from ``financial_anomaly.detect_financial_anomalies()["flags"]``.

    Returns
    -------
    dict
        {
            "trust_score":  int      — 0 to 100,
            "risk_level":   str      — human-readable risk classification,
            "color":        str      — UI color for the risk level,
            "total_flags":  int      — total number of flags across all modules,
            "high_count":   int      — number of high-severity flags,
            "medium_count": int      — number of medium-severity flags,
            "low_count":    int      — number of low-severity flags,
            "all_flags":    list     — combined normalised flags (for recommendation engine),
        }
    """
    base_score = 100

    # Combine all flags into one list (normalise dataclasses to dicts)
    all_flags = [
        _normalize_flag(f)
        for f in (cross_doc_flags + metadata_flags + financial_flags)
    ]

    # Count by severity
    high_count = 0
    medium_count = 0
    low_count = 0

    # Deduct penalties
    for flag in all_flags:
        severity = flag.get("severity", "low").lower()
        penalty = _SEVERITY_PENALTIES.get(severity, 5)
        base_score -= penalty

        if severity == "high":
            high_count += 1
        elif severity == "medium":
            medium_count += 1
        else:
            low_count += 1

    # Floor at 0
    trust_score = max(0, base_score)

    # Determine risk level and colour
    if trust_score >= 80:
        risk_level = "Low Risk"
        color = "green"
    elif trust_score >= 55:
        risk_level = "Medium Risk"
        color = "orange"
    elif trust_score >= 30:
        risk_level = "High Risk"
        color = "red"
    else:
        risk_level = "Critical Risk"
        color = "darkred"

    logger.info(
        "Trust score: %d/100 → %s | Flags: %d high, %d medium, %d low",
        trust_score, risk_level, high_count, medium_count, low_count,
    )

    return {
        "trust_score": trust_score,
        "risk_level": risk_level,
        "color": color,
        "total_flags": len(all_flags),
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "all_flags": all_flags,
    }
