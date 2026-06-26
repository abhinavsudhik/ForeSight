"""
Recommendation Engine for ForeSight.

Takes the final trust score + all combined flags → returns a human-readable
recommendation with a decision, reasoning, and per-flag evidence cards.

Decision thresholds
───────────────────
score >= 85  → "Approve — Low risk application"
score 65-84  → "Manual Review Required — moderate inconsistencies detected"
score 40-64  → "Legal Verification Required — significant fraud indicators"
score < 40   → "Reject — Multiple high severity fraud indicators detected"
"""

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision table
# ---------------------------------------------------------------------------
_DECISIONS = [
    {
        "min": 85, "max": 100,
        "decision": "Approve",
        "reasoning": (
            "Low risk application. All documents appear consistent and "
            "no significant fraud indicators were detected. The application "
            "can proceed through the standard approval workflow."
        ),
    },
    {
        "min": 65, "max": 84,
        "decision": "Manual Review Required",
        "reasoning": (
            "Moderate inconsistencies were detected. An underwriter should "
            "manually review the flagged items before proceeding. These may "
            "be benign data-entry errors or genuine discrepancies requiring "
            "clarification from the applicant."
        ),
    },
    {
        "min": 40, "max": 64,
        "decision": "Legal Verification Required",
        "reasoning": (
            "Significant fraud indicators were found. The application should "
            "not proceed until a legal team has verified the authenticity of "
            "the flagged documents and resolved all high-severity "
            "inconsistencies. Consider filing a SAR if indicators persist."
        ),
    },
    {
        "min": 0, "max": 39,
        "decision": "Reject",
        "reasoning": (
            "Multiple high-severity fraud indicators were detected. The "
            "submitted documents show critical inconsistencies that strongly "
            "suggest document tampering or identity fraud. Immediate rejection "
            "is recommended. Escalate to fraud and compliance team."
        ),
    },
]


# ---------------------------------------------------------------------------
# Per-check recommendation templates
# ---------------------------------------------------------------------------
_CHECK_RECOMMENDATIONS: dict[str, str] = {
    "name_consistency": (
        "Verify applicant identity against original documents. "
        "Request fresh copies of mismatched identity proofs."
    ),
    "property_id_consistency": (
        "Send for legal verification of property ownership. "
        "Cross-check with the local land registry."
    ),
    "timeline_consistency": (
        "Validate document issuance timeline with the issuing authority. "
        "Request re-issued documents if dates are illogical."
    ),
    "metadata_analysis": (
        "Examine the original PDF provenance. Request certified copies "
        "directly from the issuing authority if tampering is suspected."
    ),
    "financial_anomaly": (
        "Request additional bank statements or third-party financial "
        "verification. Investigate the source of unusual transactions."
    ),
}

_DEFAULT_RECOMMENDATION = (
    "Investigate this issue further and request supporting documentation."
)


# ---------------------------------------------------------------------------
# Evidence-card builder
# ---------------------------------------------------------------------------

def _build_evidence_cards(all_flags: list[dict]) -> list[dict]:
    """
    Convert raw flags into user-friendly evidence cards.

    Each card has:
    - issue:          short human-readable title
    - evidence:       the raw evidence dict from the flag
    - severity:       high / medium / low
    - recommendation: actionable next step
    """
    cards: list[dict] = []

    for flag in all_flags:
        check = flag.get("check", "unknown")
        severity = flag.get("severity", "low")
        message = flag.get("message", "Unspecified issue")
        evidence = flag.get("evidence", {})

        # Build a concise issue title from the check name
        issue_title = check.replace("_", " ").title()

        cards.append({
            "issue": f"{issue_title}: {message}",
            "evidence": evidence,
            "severity": severity,
            "recommendation": _CHECK_RECOMMENDATIONS.get(
                check, _DEFAULT_RECOMMENDATION
            ),
        })

    return cards


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_recommendation(score_result: dict) -> dict:
    """
    Generate a human-readable recommendation from the trust score result.

    Parameters
    ----------
    score_result : dict
        The output of ``risk_scorer.calculate_trust_score()``, containing
        at least ``"trust_score"`` (int) and ``"all_flags"`` (list[dict]).

    Returns
    -------
    dict
        {
            "decision":       str   — headline decision,
            "reasoning":      str   — detailed reasoning paragraph,
            "evidence_cards": list  — per-flag evidence cards,
            "trust_score":    int   — echoed back for convenience,
            "risk_level":     str   — echoed back for convenience,
        }
    """
    trust_score = score_result.get("trust_score", 0)
    all_flags = score_result.get("all_flags", [])
    risk_level = score_result.get("risk_level", "Unknown")

    # Pick the matching decision row
    decision_row = _DECISIONS[-1]  # default to Reject
    for row in _DECISIONS:
        if row["min"] <= trust_score <= row["max"]:
            decision_row = row
            break

    # Build per-flag evidence cards
    evidence_cards = _build_evidence_cards(all_flags)

    result = {
        "decision": decision_row["decision"],
        "reasoning": decision_row["reasoning"],
        "evidence_cards": evidence_cards,
        "trust_score": trust_score,
        "risk_level": risk_level,
    }

    logger.info(
        "Recommendation generated: %s (score %d/100, %d evidence cards)",
        result["decision"],
        trust_score,
        len(evidence_cards),
    )

    return result