"""
Risk Scorer for ForeSight.

Collects all flags from every module and produces a single trust score
plus a risk level classification.

Scoring
───────
- Start with a base score of 100.
- Deduct points per flag based on a check-specific penalty table
  (_FLAG_PENALTIES).  If a (check, severity) pair is not in the table,
  fall back to _FALLBACK_PENALTIES.
- Floor the score at 0.

Hard-kill rules
───────────────
1. Name or property ID fraud (high severity) → cap score at 20.
2. Two or more high-severity tampering flags → apply 20% additional reduction.

Risk levels
───────────
Score ≥ 85  → "Low Risk"      (green)
Score 65–84 → "Medium Risk"   (orange)
Score 40–64 → "High Risk"     (red)
Score < 40  → "Critical Risk" (darkred)
"""

import logging
from dataclasses import dataclass, asdict
from typing import Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Check-specific penalty table: (check_name, severity) → penalty points
_FLAG_PENALTIES = {
    # Tier 1 — Identity / Property fraud (hardest signals)
    ("name_consistency", "high"):            45,
    ("property_id_consistency", "high"):     40,
    ("metadata_analysis", "high"):           20,  # only for digital-native PDFs now

    # Tier 2 — Strong signals
    ("timeline_consistency", "medium"):      25,
    ("timeline_consistency", "high"):        30,
    ("financial_anomaly", "high"):           20,

    # Tier 3 — Moderate signals
    ("financial_anomaly", "medium"):         12,
    ("metadata_analysis", "medium"):         8,

    # Tier 4 — Weak / informational signals
    ("metadata_analysis", "low"):            3,
    ("financial_anomaly", "low"):            3,

    # Visual Tampering Flags (moderated weightage to prevent genuine document false alarms while maintaining detection)
    ("tampering_ela", "high"):               8,
    ("tampering_ela", "medium"):             4,
    ("tampering_ela", "low"):                2,

    ("tampering_noise", "high"):             8,
    ("tampering_noise", "medium"):           4,
    ("tampering_noise", "low"):              2,

    ("tampering_noise_cluster", "high"):     10,
    ("tampering_noise_cluster", "medium"):   5,
    ("tampering_noise_cluster", "low"):      2,

    ("tampering_copy_paste", "high"):        8,
    ("tampering_copy_paste", "medium"):      4,
    ("tampering_copy_paste", "low"):         2,

    ("tampering_blur", "high"):              8,
    ("tampering_blur", "medium"):            4,
    ("tampering_blur", "low"):               2,

    ("tampering_artifacts", "high"):         6,
    ("tampering_artifacts", "medium"):       3,
    ("tampering_artifacts", "low"):          2,

    ("tampering_face_region", "high"):       10,
    ("tampering_face_region", "medium"):     5,
    ("tampering_face_region", "low"):        2,
}

_FALLBACK_PENALTIES = {"high": 15, "medium": 8, "low": 3}


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
    tampering_flags: list,
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
    tampering_flags : list
        Flags from ``tamper_detector.detect_tampering()["flags"]``.

    Returns
    -------
    dict
        {
            "trust_score":         int      — 0 to 100,
            "risk_level":          str      — human-readable risk classification,
            "color":               str      — UI color for the risk level,
            "total_flags":         int      — total number of flags across all modules,
            "high_count":          int      — number of high-severity flags,
            "medium_count":        int      — number of medium-severity flags,
            "low_count":           int      — number of low-severity flags,
            "all_flags":           list     — combined normalised flags (for recommendation engine),
            "hard_kill_triggered": bool     — True if a hard-kill rule capped the score,
        }
    """
    base_score = 100

    # Combine all flags into one list (normalise dataclasses to dicts)
    # BUG FIX: tampering_flags were previously omitted from all_flags
    all_flags = [
        _normalize_flag(f)
        for f in (cross_doc_flags + metadata_flags + financial_flags + tampering_flags)
    ]

    # Count by severity
    high_count = 0
    medium_count = 0
    low_count = 0

    # Deduct penalties using check-specific lookup
    deductions_breakdown = []
    for flag in all_flags:
        severity = flag.get("severity", "low").lower()
        check = flag.get("check", "unknown")

        # Look up (check, severity) in the penalty table; fall back if not found
        penalty = _FLAG_PENALTIES.get(
            (check, severity),
            _FALLBACK_PENALTIES.get(severity, 3),
        )
        base_score -= penalty
        flag["penalty"] = penalty

        deductions_breakdown.append({
            "check": check,
            "severity": severity,
            "message": flag.get("message", ""),
            "penalty": penalty,
            "type": "flag_penalty"
        })

        if severity == "high":
            high_count += 1
        elif severity == "medium":
            medium_count += 1
        else:
            low_count += 1

    # Floor at 0
    trust_score = max(0, base_score)

    # --- Hard-kill rules ---

    # Hard kill rule 1: name or property ID fraud → cap at 20
    hard_kill_checks = {"name_consistency", "property_id_consistency"}
    has_hard_kill = any(
        f.get("check") in hard_kill_checks
        for f in all_flags
        if f.get("severity") == "high"
    )
    score_before_cap = trust_score
    if has_hard_kill:
        trust_score = min(trust_score, 20)
    
    cap_reduction = score_before_cap - trust_score
    if cap_reduction > 0:
        deductions_breakdown.append({
            "check": "hard_kill_cap",
            "severity": "high",
            "message": "Critical fraud indicator (identity/property mismatch) capped score at 20",
            "penalty": cap_reduction,
            "type": "hard_kill_cap"
        })

    # Hard kill rule 2: 2+ high-severity tampering flags →
    # apply 3% additional reduction (reverted from 5% to soften impact on genuine scans)
    high_tampering = sum(
        1 for f in all_flags
        if f.get("check", "").startswith("tampering_")
        and f.get("severity") == "high"
    )
    score_before_tamper = trust_score
    if high_tampering >= 2:
        trust_score = int(trust_score * 0.97)
    
    tamper_reduction = score_before_tamper - trust_score
    if tamper_reduction > 0:
        deductions_breakdown.append({
            "check": "tampering_multiplier",
            "severity": "high",
            "message": "Multiple high-severity tampering flags (3% additional reduction)",
            "penalty": tamper_reduction,
            "type": "tampering_multiplier"
        })

    hard_kill_triggered = has_hard_kill  # track for frontend banner

    # Determine risk level and colour
    if trust_score >= 85:
        risk_level = "Low Risk"
        color = "green"
    elif trust_score >= 65:
        risk_level = "Medium Risk"
        color = "orange"
    elif trust_score >= 40:
        risk_level = "High Risk"
        color = "red"
    else:
        risk_level = "Critical Risk"
        color = "darkred"

    logger.info(
        "Trust score: %d/100 → %s | Flags: %d high, %d medium, %d low | Hard kill: %s",
        trust_score, risk_level, high_count, medium_count, low_count,
        hard_kill_triggered,
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
        "hard_kill_triggered": hard_kill_triggered,
        "deductions_breakdown": deductions_breakdown,
    }
