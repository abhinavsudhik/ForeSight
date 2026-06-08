"""
Financial Anomaly Detector for ForeSight.

Takes the extracted bank statement fields (monthly credits and debits lists),
detects spikes and anomalies.

Logic
─────
- Calculate mean and std of monthly credits
- Any month where credit > mean + 2*std → flag as spike
- Sudden large debit before loan application date → flag
- Months with zero activity surrounded by active months → flag

Output
──────
- List of anomaly flags (same dict structure as other modules)
- A data structure for the chart: month-by-month credits and debits
  with anomaly markers — Streamlit will use this to render the chart
"""

import logging
import statistics
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SPIKE_ZSCORE_THRESHOLD = 2.0       # flag credits above mean + 2*std
_LARGE_DEBIT_MULTIPLIER = 3.0      # flag debits above 3x the mean debit
_MIN_MONTHS_FOR_ANALYSIS = 3       # need at least 3 months of data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_amount(value: str) -> float:
    """
    Convert a string amount like '1,50,000.00' or '50000' to a float.
    Returns 0.0 on failure.
    """
    if not value:
        return 0.0
    try:
        cleaned = str(value).replace(",", "").replace(" ", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def _parse_monthly_data(raw: Optional[str]) -> list[float]:
    """
    Parse a monthly credits/debits string into a list of floats.

    Accepts several formats:
    - Comma-separated: "50000, 60000, 45000"
    - Pipe-separated:  "50000|60000|45000"
    - Newline-separated
    - Single value (returned as single-element list)
    """
    if not raw:
        return []

    raw = str(raw).strip()

    # Try common delimiters
    for delimiter in ["|", "\n", ";"]:
        if delimiter in raw:
            parts = raw.split(delimiter)
            return [_parse_amount(p) for p in parts if p.strip()]

    # Comma-separated — but be careful with Indian number format (1,50,000)
    # Heuristic: if commas appear with spaces, treat as list delimiter
    if ", " in raw:
        parts = raw.split(", ")
        if all(p.strip().replace(",", "").replace(".", "").isdigit()
               for p in parts if p.strip()):
            return [_parse_amount(p) for p in parts if p.strip()]

    # Single value
    val = _parse_amount(raw)
    return [val] if val > 0 else []


# ---------------------------------------------------------------------------
# Anomaly detection functions
# ---------------------------------------------------------------------------

def _detect_credit_spikes(
    credits: list[float],
    month_labels: list[str],
) -> list[dict]:
    """
    Flag months where credit > mean + 2*std.
    Returns a list of anomaly flag dicts.
    """
    if len(credits) < _MIN_MONTHS_FOR_ANALYSIS:
        return []

    mean_credit = statistics.mean(credits)
    std_credit = statistics.stdev(credits) if len(credits) > 1 else 0.0

    if std_credit == 0:
        return []  # all values identical — no spikes

    threshold = mean_credit + (_SPIKE_ZSCORE_THRESHOLD * std_credit)
    flags = []

    for i, amount in enumerate(credits):
        if amount > threshold:
            label = month_labels[i] if i < len(month_labels) else f"Month {i + 1}"
            flags.append({
                "check": "financial_anomaly",
                "severity": "high",
                "message": (
                    f"Unusual credit spike in {label}: "
                    f"₹{amount:,.2f} exceeds threshold of ₹{threshold:,.2f} "
                    f"(mean ₹{mean_credit:,.2f} + 2×std ₹{std_credit:,.2f})"
                ),
                "evidence": {
                    "month": label,
                    "amount": amount,
                    "mean": round(mean_credit, 2),
                    "std": round(std_credit, 2),
                    "threshold": round(threshold, 2),
                },
            })

    return flags


def _detect_large_debits(
    debits: list[float],
    month_labels: list[str],
) -> list[dict]:
    """
    Flag sudden large debits (> 3× mean debit) that could indicate
    fund movement before a loan application.
    """
    if len(debits) < _MIN_MONTHS_FOR_ANALYSIS:
        return []

    mean_debit = statistics.mean(debits)
    if mean_debit == 0:
        return []

    threshold = mean_debit * _LARGE_DEBIT_MULTIPLIER
    flags = []

    for i, amount in enumerate(debits):
        if amount > threshold:
            label = month_labels[i] if i < len(month_labels) else f"Month {i + 1}"
            flags.append({
                "check": "financial_anomaly",
                "severity": "medium",
                "message": (
                    f"Sudden large debit in {label}: "
                    f"₹{amount:,.2f} is {amount / mean_debit:.1f}× the average "
                    f"monthly debit of ₹{mean_debit:,.2f}"
                ),
                "evidence": {
                    "month": label,
                    "amount": amount,
                    "mean_debit": round(mean_debit, 2),
                    "multiplier": round(amount / mean_debit, 1),
                },
            })

    return flags


def _detect_zero_activity_months(
    credits: list[float],
    debits: list[float],
    month_labels: list[str],
) -> list[dict]:
    """
    Flag months with zero activity (both credit and debit are zero)
    that are surrounded by active months — indicates possible account
    manipulation or statement gaps.
    """
    if len(credits) < _MIN_MONTHS_FOR_ANALYSIS:
        return []

    length = min(len(credits), len(debits))
    flags = []

    for i in range(1, length - 1):
        current_total = credits[i] + debits[i]
        prev_total = credits[i - 1] + debits[i - 1]
        next_total = credits[i + 1] + debits[i + 1]

        if current_total == 0 and prev_total > 0 and next_total > 0:
            label = month_labels[i] if i < len(month_labels) else f"Month {i + 1}"
            flags.append({
                "check": "financial_anomaly",
                "severity": "low",
                "message": (
                    f"Zero activity in {label} surrounded by active months — "
                    f"possible statement gap or account manipulation"
                ),
                "evidence": {
                    "month": label,
                    "prev_month_activity": round(prev_total, 2),
                    "next_month_activity": round(next_total, 2),
                },
            })

    return flags


# ---------------------------------------------------------------------------
# Chart data builder
# ---------------------------------------------------------------------------

def _build_chart_data(
    credits: list[float],
    debits: list[float],
    month_labels: list[str],
    anomaly_flags: list[dict],
) -> list[dict]:
    """
    Build a month-by-month data structure for Streamlit chart rendering.

    Each entry:
    {
        "month": "Month 1",
        "credits": 50000.0,
        "debits": 30000.0,
        "is_anomaly": True/False,
        "anomaly_types": ["credit_spike", ...]
    }
    """
    # Collect anomaly month labels for quick lookup
    anomaly_months: dict[str, list[str]] = {}
    for flag in anomaly_flags:
        month = flag.get("evidence", {}).get("month", "")
        if month:
            if month not in anomaly_months:
                anomaly_months[month] = []
            anomaly_months[month].append(flag["severity"])

    length = max(len(credits), len(debits))
    chart_data = []

    for i in range(length):
        label = month_labels[i] if i < len(month_labels) else f"Month {i + 1}"
        credit_val = credits[i] if i < len(credits) else 0.0
        debit_val = debits[i] if i < len(debits) else 0.0

        chart_data.append({
            "month": label,
            "credits": credit_val,
            "debits": debit_val,
            "is_anomaly": label in anomaly_months,
            "anomaly_types": anomaly_months.get(label, []),
        })

    return chart_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_financial_anomalies(
    monthly_credits: Optional[str] = None,
    monthly_debits: Optional[str] = None,
    month_labels: Optional[list[str]] = None,
) -> dict:
    """
    Analyse bank statement credit/debit data for anomalies.

    Parameters
    ----------
    monthly_credits : str or None
        Comma/pipe/newline-separated monthly credit amounts extracted
        from the bank statement.
    monthly_debits : str or None
        Comma/pipe/newline-separated monthly debit amounts extracted
        from the bank statement.
    month_labels : list[str] or None
        Optional list of month names/labels. Auto-generated if not provided.

    Returns
    -------
    dict
        {
            "flags": list[dict]   — anomaly flags (same structure as other modules),
            "chart_data": list[dict] — month-by-month data for Streamlit chart,
            "summary": dict       — aggregate statistics
        }
    """
    credits = _parse_monthly_data(monthly_credits)
    debits = _parse_monthly_data(monthly_debits)

    # Generate default month labels if not provided
    length = max(len(credits), len(debits))
    if not month_labels or len(month_labels) < length:
        month_labels = [f"Month {i + 1}" for i in range(length)]

    # Pad shorter list with zeros
    while len(credits) < length:
        credits.append(0.0)
    while len(debits) < length:
        debits.append(0.0)

    if length < _MIN_MONTHS_FOR_ANALYSIS:
        logger.warning(
            "Insufficient monthly data for financial analysis "
            "(got %d months, need %d)",
            length, _MIN_MONTHS_FOR_ANALYSIS,
        )
        return {
            "flags": [],
            "chart_data": _build_chart_data(credits, debits, month_labels, []),
            "summary": {
                "total_credits": sum(credits),
                "total_debits": sum(debits),
                "months_analysed": length,
                "anomalies_found": 0,
            },
        }

    logger.info("Analysing %d months of financial data …", length)

    # --- Run all anomaly detectors ---
    all_flags: list[dict] = []

    credit_spike_flags = _detect_credit_spikes(credits, month_labels)
    all_flags.extend(credit_spike_flags)

    large_debit_flags = _detect_large_debits(debits, month_labels)
    all_flags.extend(large_debit_flags)

    zero_activity_flags = _detect_zero_activity_months(
        credits, debits, month_labels
    )
    all_flags.extend(zero_activity_flags)

    logger.info("Financial analysis complete — %d anomaly flag(s)", len(all_flags))

    # --- Build chart data ---
    chart_data = _build_chart_data(credits, debits, month_labels, all_flags)

    # --- Summary statistics ---
    summary = {
        "total_credits": round(sum(credits), 2),
        "total_debits": round(sum(debits), 2),
        "avg_monthly_credit": round(statistics.mean(credits), 2),
        "avg_monthly_debit": round(statistics.mean(debits), 2),
        "months_analysed": length,
        "anomalies_found": len(all_flags),
        "credit_spikes": len(credit_spike_flags),
        "large_debits": len(large_debit_flags),
        "zero_activity_months": len(zero_activity_flags),
    }

    return {
        "flags": all_flags,
        "chart_data": chart_data,
        "summary": summary,
    }
