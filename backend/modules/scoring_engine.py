
"""
Scoring Engine for ForeSight.

Computes the final authenticity score and verdict based on multiplicative penalty weights,
and generates a human-readable recommendation block for display in the UI.
Includes an audit trail of all penalty weights and rationales.
"""

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight Rationale & Audit Trail
# ---------------------------------------------------------------------------
WEIGHT_RATIONALE: Dict[str, Dict[str, Any]] = {
    "aadhaar_verhoeff_fail": {
        "weight": 0.90,
        "rationale": "Verhoeff checksum is mathematically defined. A failure means the 12-digit number is not a valid Aadhaar number. Near-certain fabrication.",
        "false_positive_risk": "Very low — only OCR misread could cause this"
    },
    "ifsc_fail": {
        "weight": 0.85,
        "rationale": "IFSC codes follow a strict alphanumeric format (4 letters, 0, 6 characters). Failure indicates a fabricated branch code or synthetic document.",
        "false_positive_risk": "Low — OCR misreading '0' as 'O' or 'I' as '1' can trigger this."
    },
    "pan_fail": {
        "weight": 0.85,
        "rationale": "PAN numbers follow a rigid 10-character structure. Failure is highly indicative of synthetic identity or fake PAN card generation.",
        "false_positive_risk": "Low — OCR character segmentation errors could cause invalid structure."
    },
    "gstin_fail": {
        "weight": 0.80,
        "rationale": "GSTIN has a strict structure containing state code, entity type, and an embedded PAN. Failure indicates a malformed business registration document.",
        "false_positive_risk": "Low — business name or code OCR errors might occasionally trigger format mismatches."
    },
    "account_fail": {
        "weight": 0.70,
        "rationale": "Account numbers must be numeric and fall within expected length bounds for the bank prefix. Discrepancy indicates invalid account details.",
        "false_positive_risk": "Moderate — banks sometimes change account number formats or support legacy structures."
    },
    "micr_fail": {
        "weight": 0.60,
        "rationale": "MICR code format must be exactly 9 digits. Failure suggests manual manipulation or low-quality reproduction.",
        "false_positive_risk": "Moderate — OCR misreading digits or signature block overlapping the MICR band."
    },
    "api_invalid_or_not_found": {
        "weight": 0.90,
        "rationale": "The government or third-party authoritative database returned that the document ID does not exist. Extremely high confidence indicator of fake identity.",
        "false_positive_risk": "Very low — database sync delays are rare, meaning the ID is genuinely invalid."
    },
    "api_name_mismatch": {
        "weight": 0.75,
        "rationale": "The name on the document does not match the name returned by the government API database. Indicates identity theft or synthetic profile usage.",
        "false_positive_risk": "Moderate — variations in name spelling, initials, or married names can cause mismatched fuzzy scores."
    },
    "api_unavailable": {
        "weight": 0.00,
        "rationale": "Document type does not have a public verification API available or missing input fields. Treated as neutral risk.",
        "false_positive_risk": "None — no deduction is made."
    },
    "api_unreachable": {
        "weight": 0.00,
        "rationale": "Network connection failure or timeout to third-party server. Operational issues do not penalize the applicant.",
        "false_positive_risk": "None — no deduction is made."
    },
    "layout_anomaly_high": {
        "weight": 0.80,
        "rationale": "Layout similarity score is less than 0.60, representing a major structural anomaly compared to registered genuine templates.",
        "false_positive_risk": "Low-to-moderate — structural variations in authentic regional documents can sometimes trigger false positives."
    },
    "layout_anomaly_medium": {
        "weight": 0.50,
        "rationale": "Layout similarity is between 0.60 and 0.75, representing a moderate spatial anomaly.",
        "false_positive_risk": "Moderate — scanner skew, rotation, or document folding can reduce similarity score."
    },
    "layout_no_template": {
        "weight": 0.00,
        "rationale": "No registered template exists for this document type. Layout analysis cannot be performed.",
        "false_positive_risk": "None."
    },
    "tampering_halftone": {
        "weight": 0.55,
        "rationale": "FFT halftone check detects repetitive patterns characteristic of inkjet/laser printers. Indicates a printed and scanned photo swap or signature copy.",
        "false_positive_risk": "Moderate — low-quality scans or moire patterns can cause halftone false alarms."
    },
    "tampering_font_weight_high": {
        "weight": 0.60,
        "rationale": "Font weight outlier score >= 3. Significant typographical inconsistencies across text blocks suggest text manipulation.",
        "false_positive_risk": "Low-to-moderate — mixed fonts in authentic documents or poor OCR font weight estimates."
    },
    "tampering_font_weight_medium": {
        "weight": 0.35,
        "rationale": "Font weight outlier score of 1-2. Moderate typographical inconsistencies.",
        "false_positive_risk": "Moderate — variations in scan quality or bold fields in genuine documents."
    },
    "tampering_ela": {
        "weight": 0.00,
        "rationale": "Error Level Analysis is removed from standalone visual tampering risk scoring.",
        "false_positive_risk": "None."
    },
    "tampering_blur_high": {
        "weight": 0.02,
        "rationale": "Detects regions with anomalous sharpness. Paste operations create sharp/blurry boundaries.",
        "false_positive_risk": "High — uneven scan lighting, document folding, or camera focus."
    },
    "tampering_blur_medium": {
        "weight": 0.01,
        "rationale": "Moderate sharp/blurry boundaries.",
        "false_positive_risk": "High — scanner focus and lighting variance."
    },
    "tampering_blur_low": {
        "weight": 0.01,
        "rationale": "Minor sharpness variance.",
        "false_positive_risk": "High — general scanner noise."
    },
    "tampering_noise_high": {
        "weight": 0.04,
        "rationale": "Detects inconsistencies in local pixel noise levels, indicative of copy-paste compositing.",
        "false_positive_risk": "High — compression artifacts and scan noise."
    },
    "tampering_noise_medium": {
        "weight": 0.01,
        "rationale": "Moderate noise variance.",
        "false_positive_risk": "High — digital compression."
    },
    "tampering_noise_low": {
        "weight": 0.01,
        "rationale": "Minor noise fluctuations.",
        "false_positive_risk": "High."
    },
    "tampering_noise_cluster_high": {
        "weight": 0.05,
        "rationale": "Clustered high-frequency noise regions indicate digital splicing or editing.",
        "false_positive_risk": "High — local shadows or scanner dust."
    },
    "tampering_noise_cluster_medium": {
        "weight": 0.02,
        "rationale": "Moderate noise clusters.",
        "false_positive_risk": "High."
    },
    "tampering_noise_cluster_low": {
        "weight": 0.01,
        "rationale": "Minor localized noise clusters.",
        "false_positive_risk": "High."
    },
    "tampering_artifacts_high": {
        "weight": 0.01,
        "rationale": "Detects specific pixel level splicing artifacts or gradient anomalies.",
        "false_positive_risk": "High — low-resolution compression."
    },
    "tampering_artifacts_medium": {
        "weight": 0.01,
        "rationale": "Moderate pixel artifacts.",
        "false_positive_risk": "High."
    },
    "tampering_artifacts_low": {
        "weight": 0.01,
        "rationale": "Minor pixel artifacts.",
        "false_positive_risk": "High."
    },
    "tampering_copy_paste_high": {
        "weight": 0.04,
        "rationale": "Detects duplicated image regions within the document, suggesting cloned stamps or text.",
        "false_positive_risk": "Low — repetitive patterns in document design (e.g. grids) can trigger false positives."
    },
    "tampering_copy_paste_medium": {
        "weight": 0.02,
        "rationale": "Moderate copy-paste indicator matching.",
        "false_positive_risk": "Low-to-moderate."
    },
    "tampering_copy_paste_low": {
        "weight": 0.01,
        "rationale": "Minor duplicated region indicators.",
        "false_positive_risk": "Moderate."
    },
    "tampering_face_region_high": {
        "weight": 0.04,
        "rationale": "Photo area shows characteristics (noise/compression/gradient) inconsistent with background. Indicates photo swap.",
        "false_positive_risk": "Moderate — photo scans naturally have different textures than printed text."
    },
    "tampering_face_region_medium": {
        "weight": 0.02,
        "rationale": "Moderate face region inconsistencies.",
        "false_positive_risk": "Moderate."
    },
    "tampering_face_region_low": {
        "weight": 0.01,
        "rationale": "Minor face region anomalies.",
        "false_positive_risk": "Moderate."
    },
    "background_pattern_disruption": {
        "weight": 0.65,
        "rationale": "Continuous security background pattern (guilloche, microprint, gradient) is disrupted under high-value fields. Suggests local erasure and forgery.",
        "false_positive_risk": "Low — flat or plain white backgrounds are skipped, and noise floor checks prevent false alarms."
    },
    "name_consistency": {
        "weight": 0.45,
        "rationale": "Name mismatch across documents (e.g. bank statement vs PAN card). Suggests synthetic identity or incorrect submission.",
        "false_positive_risk": "Moderate — minor spelling variations, initials, or name ordering."
    },
    "property_id_consistency": {
        "weight": 0.40,
        "rationale": "Mismatch in property IDs or survey numbers across documents.",
        "false_positive_risk": "Low — typos in manual entry or OCR reading errors."
    },
    "timeline_consistency_high": {
        "weight": 0.30,
        "rationale": "Critical logical timeline contradiction (e.g. land record issued after sale deed).",
        "false_positive_risk": "Low — OCR date misread."
    },
    "timeline_consistency_medium": {
        "weight": 0.25,
        "rationale": "Non-critical timeline contradiction or suspicious date sequences.",
        "false_positive_risk": "Low — OCR date misread."
    },
    "financial_anomaly_high": {
        "weight": 0.20,
        "rationale": "High severity financial anomalies such as extreme credit spikes or irregular transactions.",
        "false_positive_risk": "Moderate — unusual but legitimate financial events."
    },
    "financial_anomaly_medium": {
        "weight": 0.12,
        "rationale": "Medium severity financial anomalies like sudden zero activity months.",
        "false_positive_risk": "Moderate — temporary inactive bank accounts."
    },
    "financial_anomaly_low": {
        "weight": 0.03,
        "rationale": "Minor transaction irregularities.",
        "false_positive_risk": "High — typical variance in personal banking."
    },
    "metadata_analysis_high": {
        "weight": 0.20,
        "rationale": "High severity metadata concerns (e.g. PDF edited with known tampering tools, anomalous software producer).",
        "false_positive_risk": "Low — legal edits to document PDFs."
    },
    "metadata_analysis_medium": {
        "weight": 0.08,
        "rationale": "Medium severity metadata issues (e.g. modified date after creation date with suspicious author).",
        "false_positive_risk": "Moderate — normal document scanning and saving operations."
    },
    "metadata_analysis_low": {
        "weight": 0.03,
        "rationale": "Minor metadata mismatches.",
        "false_positive_risk": "High — missing creation dates or generic PDF tools."
    }
}


def normalize_section_name(sec_name: str) -> str:
    """Map any custom or sub-check section name to the standard six sections."""
    if not sec_name:
        return "other"
    sec_name = str(sec_name).lower().strip()
    if sec_name == "layout_anomaly":
        return "cross_doc_consistency"
    return sec_name


def resolve_flag_weight(flag: Any, section_name: str = None) -> Tuple[float, str]:
    """
    Resolve the penalty weight and the rationale key for a given flag.
    
    Checks WEIGHT_RATIONALE first based on check name and severity,
    and falls back to any embedded penalty_weight in the flag.
    """
    if isinstance(flag, str):
        check_name = flag
        severity = "high"
        evidence = None
    elif isinstance(flag, dict):
        check_name = flag.get("check") or flag.get("name") or "unknown"
        severity = flag.get("severity") or "low"
        evidence = flag.get("evidence")
    else:
        check_name = str(flag)
        severity = "low"
        evidence = None

    check_name = check_name.lower().strip()
    severity = str(severity).lower().strip()

    rationale_key = None
    weight = 0.0

    # 1. format_checksum
    if check_name in ["aadhaar_checksum", "aadhaar_fail", "aadhaar_verhoeff_fail"]:
        rationale_key = "aadhaar_verhoeff_fail"
    elif check_name in ["ifsc_format", "ifsc_fail"]:
        rationale_key = "ifsc_fail"
    elif check_name in ["pan_format", "pan_fail"]:
        rationale_key = "pan_fail"
    elif check_name in ["gstin_format", "gstin_fail"]:
        rationale_key = "gstin_fail"
    elif check_name in ["account_format", "account_mismatch", "account_fail"]:
        rationale_key = "account_fail"
    elif check_name in ["micr_format", "micr_fail"]:
        rationale_key = "micr_fail"

    # 2. api_verification
    elif check_name in ["api_invalid_or_not_found", "invalid/not found"]:
        rationale_key = "api_invalid_or_not_found"
    elif check_name in ["api_name_mismatch", "name mismatch"]:
        rationale_key = "api_name_mismatch"
    elif check_name in ["api_unreachable", "api unreachable"]:
        rationale_key = "api_unreachable"
    elif check_name in ["api_not_available", "api_unavailable", "api unavailable (doc type)", "missing_id_number", "offline_mode", "offline_skip"]:
        rationale_key = "api_unavailable"

    # 3. cross_doc_consistency & layout
    elif check_name in ["name_consistency"]:
        rationale_key = "name_consistency"
    elif check_name in ["property_id_consistency"]:
        rationale_key = "property_id_consistency"
    elif check_name in ["timeline_consistency"]:
        if severity == "high":
            rationale_key = "timeline_consistency_high"
        else:
            rationale_key = "timeline_consistency_medium"
    elif check_name in ["layout_anomaly", "layout_checker", "layout"]:
        similarity = None
        if isinstance(flag, dict):
            similarity = flag.get("similarity_score")
            if similarity is None and isinstance(evidence, dict):
                similarity = evidence.get("similarity_score")
        
        if similarity is not None:
            try:
                sim_val = float(similarity)
                if sim_val < 0.60:
                    rationale_key = "layout_anomaly_high"
                elif sim_val < 0.75:
                    rationale_key = "layout_anomaly_medium"
                else:
                    rationale_key = "layout_no_template"
            except (ValueError, TypeError):
                pass
        
        if not rationale_key:
            if severity == "high":
                rationale_key = "layout_anomaly_high"
            elif severity == "medium":
                rationale_key = "layout_anomaly_medium"
            else:
                rationale_key = "layout_no_template"

    # 4. image_tampering
    elif check_name in ["tampering_halftone", "halftone", "fft_halftone"]:
        rationale_key = "tampering_halftone"
    elif check_name in ["tampering_font_weight", "font_weight"]:
        outliers = None
        if isinstance(flag, dict):
            outliers = flag.get("outliers")
            if outliers is None and isinstance(evidence, dict):
                outliers = flag.get("outliers") or flag.get("outlier_count")
        
        if outliers is not None:
            try:
                out_val = int(outliers)
                if out_val >= 3:
                    rationale_key = "tampering_font_weight_high"
                elif out_val >= 1:
                    rationale_key = "tampering_font_weight_medium"
                else:
                    weight = 0.0
                    rationale_key = "tampering_font_weight_medium"
            except (ValueError, TypeError):
                pass
        
        if not rationale_key:
            if severity in ["high", "critical"]:
                rationale_key = "tampering_font_weight_high"
            else:
                rationale_key = "tampering_font_weight_medium"
    elif check_name in ["tampering_ela", "ela"]:
        rationale_key = "tampering_ela"
    elif check_name in ["tampering_blur", "blur"]:
        if severity == "high":
            rationale_key = "tampering_blur_high"
        elif severity == "medium":
            rationale_key = "tampering_blur_medium"
        else:
            rationale_key = "tampering_blur_low"
    elif check_name in ["tampering_noise", "noise"]:
        if severity == "high":
            rationale_key = "tampering_noise_high"
        elif severity == "medium":
            rationale_key = "tampering_noise_medium"
        else:
            rationale_key = "tampering_noise_low"
    elif check_name in ["tampering_noise_cluster", "noise_cluster"]:
        if severity == "high":
            rationale_key = "tampering_noise_cluster_high"
        elif severity == "medium":
            rationale_key = "tampering_noise_cluster_medium"
        else:
            rationale_key = "tampering_noise_cluster_low"
    elif check_name in ["tampering_artifacts", "artifacts", "pixel_artifacts"]:
        if severity == "high":
            rationale_key = "tampering_artifacts_high"
        elif severity == "medium":
            rationale_key = "tampering_artifacts_medium"
        else:
            rationale_key = "tampering_artifacts_low"
    elif check_name in ["tampering_copy_paste", "copy_paste"]:
        if severity == "high":
            rationale_key = "tampering_copy_paste_high"
        elif severity == "medium":
            rationale_key = "tampering_copy_paste_medium"
        else:
            rationale_key = "tampering_copy_paste_low"
    elif check_name in ["tampering_face_region", "face_region"]:
        if severity == "high":
            rationale_key = "tampering_face_region_high"
        elif severity == "medium":
            rationale_key = "tampering_face_region_medium"
        else:
            rationale_key = "tampering_face_region_low"
    elif check_name in ["background_pattern_disruption", "tampering_background_pattern_disruption"]:
        rationale_key = "background_pattern_disruption"

    # 5. financial_anomaly
    elif check_name in ["financial_anomaly", "financial"]:
        if severity == "high":
            rationale_key = "financial_anomaly_high"
        elif severity == "medium":
            rationale_key = "financial_anomaly_medium"
        else:
            rationale_key = "financial_anomaly_low"

    # 6. metadata
    elif check_name in ["metadata_analysis", "metadata"]:
        if severity == "high":
            rationale_key = "metadata_analysis_high"
        elif severity == "medium":
            rationale_key = "metadata_analysis_medium"
        else:
            rationale_key = "metadata_analysis_low"

    # Load weight from rationale
    if rationale_key and rationale_key in WEIGHT_RATIONALE:
        weight = WEIGHT_RATIONALE[rationale_key]["weight"]
    else:
        # Fallback to flag properties if not in dictionary
        if isinstance(flag, dict) and "penalty_weight" in flag:
            weight = float(flag["penalty_weight"])
        else:
            if severity == "high":
                weight = 0.50
            elif severity == "medium":
                weight = 0.25
            else:
                weight = 0.05

    return weight, rationale_key or f"custom_{check_name}"


def compute_final_score(section_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Take a list of section result dicts, compute final authenticity score and verdict.
    
    Formula: score = score * (1 - penalty_weight) for each flag raised.
    """
    # Standard list of sections to return in contributions
    standard_sections = [
        "format_checksum",
        "api_verification",
        "cross_doc_consistency",
        "image_tampering",
        "financial_anomaly"
    ]

    section_flags: Dict[str, List[Dict[str, Any]]] = {s: [] for s in standard_sections}
    section_risks: Dict[str, str] = {s: "clean" for s in standard_sections}
    all_flags_flat: List[Dict[str, Any]] = []

    # Map raw results to standard sections
    for sec in section_results:
        if not isinstance(sec, dict):
            continue
        
        raw_sec_name = sec.get("section")
        sec_name = normalize_section_name(raw_sec_name)
        
        # Ensure the section is registered in standard list
        if sec_name not in section_flags:
            section_flags[sec_name] = []
            section_risks[sec_name] = "clean"

        # Track risk level
        current_risk = str(sec.get("risk", "clean")).lower().strip()
        risk_hierarchy = {"clean": 0, "skipped": 1, "medium": 2, "high": 3, "suspicious": 2, "high_risk": 3}
        if risk_hierarchy.get(current_risk, 0) > risk_hierarchy.get(section_risks[sec_name], 0):
            # Normalize risk names to standard output
            normalized_risk = current_risk
            if current_risk == "suspicious":
                normalized_risk = "medium"
            elif current_risk == "high_risk":
                normalized_risk = "high"
            section_risks[sec_name] = normalized_risk

        # Extract flags
        flags_list = []
        if "flags" in sec and isinstance(sec["flags"], list):
            for f in sec["flags"]:
                flags_list.append(f)
        elif "checks" in sec and isinstance(sec["checks"], list):
            for check in sec["checks"]:
                if isinstance(check, dict) and "flags" in check:
                    for f in check["flags"]:
                        flags_list.append(f)

        # Synthesize a flag if there are no explicit flags but section has a penalty weight/risk
        if not flags_list:
            pw = sec.get("penalty_weight", 0.0)
            if pw > 0.0 or current_risk not in ["clean", "skipped", "neutral"]:
                flags_list.append({
                    "check": sec.get("check") or raw_sec_name or "anomaly",
                    "severity": current_risk if current_risk in ["low", "medium", "high"] else "high",
                    "message": sec.get("summary") or sec.get("message") or f"Anomaly detected in {raw_sec_name or 'section'}",
                    "penalty_weight": pw,
                    # Propagate similarity score or outliers if layout/font weight checks are nested
                    "similarity_score": sec.get("similarity_score"),
                    "outliers": sec.get("outliers") or sec.get("outlier_count")
                })
            elif current_risk == "skipped" or sec.get("skipped"):
                msg = sec.get("reason") or "verification skipped"
                flags_list.append({
                    "check": "api_not_available" if "no api" in msg.lower() else ("api_unavailable" if "unreachable" in msg.lower() or "timeout" in msg.lower() else "offline_mode"),
                    "severity": "low",
                    "message": f"API verification skipped: {msg}",
                    "penalty_weight": 0.0
                })

        for f in flags_list:
            weight, rationale_key = resolve_flag_weight(f, sec_name)
            
            # Format flag for flat output
            check_val = f.get("check") if isinstance(f, dict) else str(f)
            message_val = f.get("message") if isinstance(f, dict) else str(f)
            severity_val = f.get("severity", "medium") if isinstance(f, dict) else "medium"
            evidence_val = f.get("evidence") if isinstance(f, dict) else None

            flat_flag = {
                "section": sec_name,
                "check": check_val,
                "severity": severity_val,
                "message": message_val,
                "penalty_weight": weight,
                "rationale_key": rationale_key,
                "evidence": evidence_val
            }
            section_flags[sec_name].append(flat_flag)
            all_flags_flat.append(flat_flag)

    # Compute authenticity score and section multipliers
    final_score = 1.0
    section_contributions = []

    for sec_name in standard_sections:
        flags = section_flags[sec_name]
        
        # Calculate section multiplier: product of (1 - weight) for all flags in the section
        multiplier = 1.0
        for f in flags:
            multiplier *= (1.0 - f["penalty_weight"])
        
        final_score *= multiplier
        
        max_weight = max([f["penalty_weight"] for f in flags], default=0.0)
        
        section_contributions.append({
            "section": sec_name,
            "risk": section_risks[sec_name],
            "penalty_weight": round(max_weight, 2),
            "score_multiplier": round(multiplier, 4),
            "flags_count": len(flags),
            "flags": flags
        })

    # Floor and Cap final score
    final_score = max(0.0, min(1.0, final_score))
    final_score_rounded = round(final_score, 4)

    # Determine Verdict
    if final_score_rounded >= 0.70:
        verdict = "PASS"
    elif final_score_rounded >= 0.40:
        verdict = "REVIEW"
    else:
        verdict = "REJECT"

    # Sort all flags to find the top 3 worst penalty flags
    sorted_flags = sorted(all_flags_flat, key=lambda x: x["penalty_weight"], reverse=True)
    top_flags = sorted_flags[:3]

    return {
        "final_score": final_score_rounded,
        "verdict": verdict,
        "verdict_thresholds": {
            "PASS": ">= 0.70",
            "REVIEW": "0.40 - 0.69",
            "REJECT": "< 0.40"
        },
        "section_contributions": section_contributions,
        "top_flags": top_flags
    }


def generate_recommendation_text(score_result: Dict[str, Any], section_results: List[Dict[str, Any]]) -> str:
    """
    Generate a human-readable recommendation block for UI display.
    """
    verdict = score_result.get("verdict", "REVIEW")
    score = score_result.get("final_score", 1.0)
    
    # Map contributions by standardized section name
    contrib_map = {}
    for contrib in score_result.get("section_contributions", []):
        contrib_map[contrib["section"]] = contrib
        
    # Map raw results by standardized name for any fields not fully resolved in contributions
    results_map = {}
    for sec in section_results:
        if isinstance(sec, dict):
            sec_name = normalize_section_name(sec.get("section"))
            results_map[sec_name] = sec

    # 1. Critical Flags (non-zero weight flags)
    all_flags = []
    for contrib in contrib_map.values():
        for f in contrib.get("flags", []):
            all_flags.append(f)
            
    critical_flags = [f for f in all_flags if f.get("penalty_weight", 0.0) > 0.0]
    critical_flags_sorted = sorted(critical_flags, key=lambda x: x["penalty_weight"], reverse=True)
    
    critical_block_lines = []
    if critical_flags_sorted:
        for f in critical_flags_sorted:
            check_desc = f.get("message") or f.get("check") or "Flag raised"
            sec_display = str(f.get("section", "unknown")).replace("_", " ").title()
            weight = f.get("penalty_weight", 0.0)
            critical_block_lines.append(f"• {check_desc} — {sec_display} — weight: {weight:.2f}")
        critical_flags_text = "\n".join(critical_block_lines)
    else:
        critical_flags_text = "None"

    # 2. Extract section details for Analysis Summary
    format_sum = "clean — All formats and Verhoeff checksums validated successfully."
    api_sum = "skipped — No API verification performed."
    layout_sum = "no template — Layout checker skipped."
    image_sum = "clean — No visual tampering checks executed."
    financial_sum = "clean — No bank statement anomalies analyzed."

    # Format & Checksum Summary
    if "format_checksum" in contrib_map:
        contrib = contrib_map["format_checksum"]
        risk = contrib.get("risk", "clean")
        flags = contrib.get("flags", [])
        failed_fields = []
        for f in flags:
            if f.get("penalty_weight", 0.0) > 0.0:
                check_name = f.get("check", "").lower()
                field_name = check_name.replace("_format", "").replace("_checksum", "").replace("_fail", "").upper()
                failed_fields.append(field_name)
        
        if failed_fields:
            format_sum = f"{risk} — Failed validations: {', '.join(failed_fields)}"
        else:
            format_sum = f"{risk} — All formats and Verhoeff checksums validated successfully."

    # API Verification Summary
    if "api_verification" in contrib_map:
        contrib = contrib_map["api_verification"]
        risk = contrib.get("risk", "clean")
        flags = contrib.get("flags", [])
        raw_sec = results_map.get("api_verification", {})
        skipped = raw_sec.get("skipped", False) or risk == "skipped" or (not raw_sec.get("api_available", True))
        
        if skipped:
            api_sum = "skipped — API verification skipped (offline mode or no API coverage)"
        elif flags:
            flag_desc = flags[0].get("message", "API verification returned errors")
            api_sum = f"{risk} — {flag_desc}"
        else:
            api_sum = f"{risk} — Identity verified via government/3P database"

    # Layout Analysis Summary
    # Check similarity score in raw results or search in cross-document flags
    layout_sec = results_map.get("layout_anomaly") or results_map.get("cross_doc_consistency")
    layout_flags = []
    if "cross_doc_consistency" in contrib_map:
        layout_flags = [f for f in contrib_map["cross_doc_consistency"]["flags"] if "layout" in f.get("check", "").lower()]

    if layout_sec:
        no_template = layout_sec.get("no_template", False)
        sim_score = layout_sec.get("similarity_score")
        risk = layout_sec.get("risk", "clean")
        
        if no_template:
            layout_sum = "no template — No template registered for this document type"
        elif sim_score is not None:
            layout_sum = f"similarity score: {sim_score:.2f} ({risk})"
        elif layout_flags:
            layout_sum = f"{layout_flags[0].get('severity')} — {layout_flags[0].get('message')}"
        else:
            layout_sum = f"{risk} — Layout verified against registered template"

    # Image Forensics Summary
    if "image_tampering" in contrib_map:
        contrib = contrib_map["image_tampering"]
        risk = contrib.get("risk", "clean")
        flags = contrib.get("flags", [])
        
        if flags:
            indicators = []
            for f in flags:
                name = f.get("check", "").replace("tampering_", "").replace("_", " ")
                if name not in indicators:
                    indicators.append(name)
            image_sum = f"{risk} — {len(flags)} tampering indicator(s) detected: {', '.join(indicators)}"
        else:
            image_sum = "clean — Halftone and font weight analysis clean. No tampering detected."

    # Financial Analysis Summary
    if "financial_anomaly" in contrib_map:
        contrib = contrib_map["financial_anomaly"]
        flags = contrib.get("flags", [])
        risk = contrib.get("risk", "clean")
        
        if flags:
            financial_sum = f"{risk} — {len(flags)} financial anomalies detected: " + "; ".join([f.get("message") for f in flags[:2]])
        else:
            financial_sum = "clean — No bank statement anomalies detected."

    # 3. Recommended Action
    if verdict == "PASS":
        recommended_action = "Standard automated onboarding approved. Proceed with standard approval workflow as no fraud indicators were detected."
    elif verdict == "REVIEW":
        recommended_action = "Manual review required. An underwriter should inspect the flagged inconsistencies, verify candidate identity against original physical documents, and request clarifications if needed."
    else:  # REJECT
        recommended_action = "Critical risk detected. Immediate rejection of the application is recommended. Escalate to the fraud investigation unit and file a suspicious activity report (SAR) if required."

    # 4. Low Confidence Signals (flags with weight 0.0)
    low_confidence_flags = [f for f in all_flags if f.get("penalty_weight", 0.0) == 0.0]
    low_confidence_lines = []
    if low_confidence_flags:
        for f in low_confidence_flags:
            check_desc = f.get("message") or f.get("check") or "Flag raised"
            low_confidence_lines.append(f"• {check_desc}")
        low_confidence_text = "\n".join(low_confidence_lines)
    else:
        low_confidence_text = "None"

    # Assemble the recommendation block
    recommendation_block = f"""VERDICT: {verdict}
AUTHENTICITY SCORE: {score:.2f}

CRITICAL FLAGS (if any):
{critical_flags_text}

ANALYSIS SUMMARY:
• Format & Checksum: {format_sum}
• API Verification: {api_sum}
• Layout Analysis: {layout_sum}
• Image Forensics: {image_sum}
• Financial Analysis: {financial_sum}

RECOMMENDED ACTION:
{recommended_action}

LOW-CONFIDENCE SIGNALS (flagged but not weighted):
{low_confidence_text}"""

    return recommendation_block
