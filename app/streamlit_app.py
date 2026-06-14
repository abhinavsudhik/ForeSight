"""
ForeSight — AI-Powered Document Fraud Detection Dashboard

Streamlit frontend that orchestrates the full analysis pipeline:
  Upload → OCR → Classify → Extract → Cross-Doc → Metadata → Financial → Score → Recommend

Six tabs:
  1. Case Overview        — trust score, risk badge, flag summary
  2. Extracted Data       — per-document field tables
  3. Cross-Document Flags — expandable cards by severity
  4. Metadata Analysis    — PDF metadata + suspicious-pattern flags
  5. Financial Analysis   — Plotly credit/debit charts with anomaly markers
  6. Recommendation       — decision card + evidence cards
"""

import os
import sys
import uuid
import tempfile
import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so backend modules can be imported.
# This MUST come before any `from backend.*` import.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import plotly.graph_objects as go

from backend.modules.tamper_detector import (
    detect_tampering,
    detect_tampering_from_pdf_page,
)

from backend.modules.ocr_engine import extract_text
from backend.modules.document_classifier import classify_document
from backend.modules.field_extractor import extract_fields
from backend.modules.cross_document_engine import cross_validate
from backend.modules.metadata_analyzer import analyze_metadata
from backend.modules.financial_anomaly import detect_financial_anomalies
from backend.modules.risk_scorer import calculate_trust_score
from backend.modules.recommendation_engine import generate_recommendation

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Page config & global styling
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ForeSight — Document Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}

/* ── Score card ─────────────────────────────────────────────────────────── */
.score-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.score-number {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1.1;
}
.score-label {
    font-size: 0.9rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.25rem;
}

/* ── Risk badge ─────────────────────────────────────────────────────────── */
.risk-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Info card ──────────────────────────────────────────────────────────── */
.info-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    margin-bottom: 0.75rem;
}
.info-card h4 {
    margin: 0 0 0.25rem 0;
    color: #8892b0;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
.info-card p {
    margin: 0;
    font-size: 1.35rem;
    font-weight: 600;
    color: #ccd6f6;
}

/* ── Severity dot ──────────────────────────────────────────────────────── */
.severity-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
}
.severity-high   { background: #ff4d4f; box-shadow: 0 0 6px #ff4d4f88; }
.severity-medium { background: #faad14; box-shadow: 0 0 6px #faad1488; }
.severity-low    { background: #52c41a; box-shadow: 0 0 6px #52c41a88; }

/* ── Flag counter badges ───────────────────────────────────────────────── */
.flag-counts {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 1rem;
}
.flag-badge {
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
}

/* ── Decision card ─────────────────────────────────────────────────────── */
.decision-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    margin-bottom: 1.5rem;
}

/* ── Metadata table ────────────────────────────────────────────────────── */
.meta-table {
    width: 100%;
    border-collapse: collapse;
}
.meta-table td {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.meta-table td:first-child {
    color: #8892b0;
    font-weight: 500;
    width: 40%;
}

/* ── Hero header ───────────────────────────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.hero-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.hero-header p {
    color: #8892b0;
    font-size: 1.05rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def _generate_case_id() -> str:
    """Generate a unique case ID like FS-2026-A3F."""
    now = datetime.now()
    short_id = uuid.uuid4().hex[:3].upper()
    return f"FS-{now.year}-{short_id}"


def _severity_color(severity: str) -> str:
    """Map severity string to a CSS colour."""
    return {
        "high": "#ff4d4f",
        "medium": "#faad14",
        "low": "#52c41a",
    }.get(severity.lower(), "#8892b0")


def _severity_emoji(severity: str) -> str:
    """Map severity to a coloured emoji."""
    return {
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢",
    }.get(severity.lower(), "⚪")


def _risk_badge_color(color: str) -> str:
    """Map the risk_scorer colour name to a hex value."""
    return {
        "green": "#52c41a",
        "orange": "#faad14",
        "red": "#ff4d4f",
        "darkred": "#a8071a",
    }.get(color, "#8892b0")


def _get_applicant_name(documents: list[dict]) -> str:
    """Pull the applicant name from the identity-proof document."""
    for doc in documents:
        if doc.get("document_type") == "identity_proof":
            name = doc.get("fields", {}).get("name")
            if name:
                return name
    # Fallback: check any name-like field
    for doc in documents:
        fields = doc.get("fields", {})
        for key in ("name", "owner_name", "account_holder", "seller_name", "buyer_name"):
            val = fields.get(key)
            if val:
                return val
    return "Unknown Applicant"


# ═══════════════════════════════════════════════════════════════════════════
# Processing pipeline
# ═══════════════════════════════════════════════════════════════════════════

def _run_pipeline(uploaded_files) -> dict:
    """
    Execute the full ForeSight analysis pipeline on uploaded files.

    Returns a dict with all results stored in session state.
    """
    case_id = _generate_case_id()
    documents: list[dict] = []
    all_metadata_results: list[dict] = []
    financial_result: dict | None = None

    total_steps = len(uploaded_files) * 3 + 5  # OCR+Classify+Extract per file, + cross/meta/tamper/fin/score
    current_step = 0
    progress = st.progress(0, text="Starting analysis…")

    # ------------------------------------------------------------------
    # Phase 1 — Per-document processing
    # ------------------------------------------------------------------
    for file in uploaded_files:
        filename = file.name

        # Save uploaded file to a temp path for OCR / metadata
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(filename)[1],
        ) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        # Step 1: OCR
        current_step += 1
        progress.progress(
            current_step / total_steps,
            text=f"🔍 Running OCR on {filename}…",
        )
        try:
            raw_text = extract_text(tmp_path)
        except Exception as exc:
            st.warning(f"OCR failed for {filename}: {exc}")
            raw_text = ""

        # Step 2: Classify
        current_step += 1
        progress.progress(
            current_step / total_steps,
            text=f"📑 Classifying {filename}…",
        )
        classification = classify_document(raw_text)

        # Step 3: Extract fields
        current_step += 1
        progress.progress(
            current_step / total_steps,
            text=f"📋 Extracting fields from {filename}…",
        )
        if classification.label == "unknown":
            st.warning(
                f"⚠️ **{filename}**: Could not determine document type "
                f"(no keyword matches). Skipping field extraction."
            )
            doc_record = {
                "filename": filename,
                "tmp_path": tmp_path,
                "document_type": "unknown",
                "classification_confidence": 0.0,
                "fields": {},
                "fields_found": 0,
                "fields_missing": [],
                "raw_text_length": len(raw_text),
            }
        else:
            try:
                extraction = extract_fields(raw_text, classification.label)
                doc_record = {
                    "filename": filename,
                    "tmp_path": tmp_path,
                    "document_type": classification.label,
                    "classification_confidence": classification.confidence,
                    "fields": extraction.fields,
                    "fields_found": extraction.fields_found,
                    "fields_missing": extraction.fields_missing,
                    "raw_text_length": extraction.raw_text_length,
                }
            except ValueError as exc:
                st.warning(f"⚠️ **{filename}**: Field extraction failed — {exc}")
                doc_record = {
                    "filename": filename,
                    "tmp_path": tmp_path,
                    "document_type": classification.label,
                    "classification_confidence": classification.confidence,
                    "fields": {},
                    "fields_found": 0,
                    "fields_missing": [],
                    "raw_text_length": len(raw_text),
                }
        documents.append(doc_record)

    try:
        raw_text = extract_text(tmp_path)
    except Exception as exc:
        st.error(f"OCR ERROR: {exc}")
        raw_text = ""

    # ADD THIS TEMPORARILY
    st.markdown(f"**Debug — {filename}**")
    st.write(f"Characters extracted: {len(raw_text)}")
    st.text_area("Raw text", raw_text[:2000], height=200)

    # ------------------------------------------------------------------
    # Phase 2 — Cross-document validation
    # ------------------------------------------------------------------
    current_step += 1
    progress.progress(
        current_step / total_steps,
        text="🔗 Running cross-document checks…",
    )
    cross_doc_flags = cross_validate(documents)

    # ------------------------------------------------------------------
    # Phase 3 — Metadata analysis (per PDF)
    # ------------------------------------------------------------------
    current_step += 1
    progress.progress(
        current_step / total_steps,
        text="🔒 Analysing PDF metadata…",
    )
    all_metadata_flags: list[dict] = []
    for doc in documents:
        if doc["tmp_path"].lower().endswith(".pdf"):
            # Pass the fields dictionary to let the analyzer fetch the issue date
            meta_result = analyze_metadata(doc["tmp_path"], doc["fields"])
            all_metadata_results.append({
                "filename": doc["filename"],
                "metadata": meta_result["metadata"],
                "flags": meta_result["flags"],
                "summary": meta_result["summary"],
            })
            all_metadata_flags.extend(meta_result["flags"])

    
    # Phase 3b — Visual tampering analysis (per document)
    current_step += 1
    progress.progress(
        current_step / total_steps,
        text="🔬 Running visual tampering analysis…",
    )
    all_tampering_results = []
    for doc in documents:
        path = doc["tmp_path"]
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                tamper_result = detect_tampering_from_pdf_page(path)
            else:
                tamper_result = detect_tampering(path)
            all_tampering_results.append({
                "filename": doc["filename"],
                "result": tamper_result,
            })
        except Exception as exc:
            logger.warning("Tampering analysis failed for %s: %s", doc["filename"], exc)

    # ------------------------------------------------------------------
    # Phase 4 — Financial anomaly detection (bank statement only)
    # ------------------------------------------------------------------
    current_step += 1
    progress.progress(
        current_step / total_steps,
        text="📊 Detecting financial anomalies…",
    )
    financial_flags: list[dict] = []
    for doc in documents:
        if doc["document_type"] == "bank_statement":
            financial_result = detect_financial_anomalies(
                monthly_credits=doc["fields"].get("monthly_credits"),
                monthly_debits=doc["fields"].get("monthly_debits"),
            )
            financial_flags = financial_result.get("flags", [])
            break  # only one bank statement expected

    # ------------------------------------------------------------------
    # Phase 5 — Trust score & recommendation
    # ------------------------------------------------------------------
    current_step += 1

    tampering_flags = []
    for tr in all_tampering_results:
        tampering_flags.extend(tr["result"].get("flags", []))

    progress.progress(
        current_step / total_steps,
        text="⚖️ Calculating trust score…",
    )
    
    score_result = calculate_trust_score(
        cross_doc_flags=cross_doc_flags,
        metadata_flags=all_metadata_flags,
        financial_flags=financial_flags,
        tampering_flags=tampering_flags,
    )
    recommendation = generate_recommendation(score_result)

    progress.progress(1.0, text="✅ Analysis complete!")

    return {
        "case_id": case_id,
        "documents": documents,
        "cross_doc_flags": cross_doc_flags,
        "metadata_results": all_metadata_results,
        "financial_result": financial_result,
        "score_result": score_result,
        "recommendation": recommendation,
        "tampering_results": all_tampering_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tab renderers
# ═══════════════════════════════════════════════════════════════════════════

def _render_tab_case_overview(results: dict):
    """Tab 1 — Case Overview."""
    score = results["score_result"]
    recommendation = results["recommendation"]
    documents = results["documents"]

    applicant_name = _get_applicant_name(documents)
    badge_color = _risk_badge_color(score["color"])
    score_color = badge_color

    # ── Top row: Case ID + Applicant ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h4>Case ID</h4>
            <p>{results['case_id']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h4>Applicant</h4>
            <p>{applicant_name}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # ── Score + Badge + Decision ──
    col_score, col_badge, col_decision = st.columns([1, 1, 1.5])

    with col_score:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-number" style="color: {score_color};">{score['trust_score']}</div>
            <div class="score-label">Trust Score / 100</div>
        </div>
        """, unsafe_allow_html=True)

    with col_badge:
        st.markdown(f"""
        <div class="score-card">
            <div style="margin-bottom: 0.75rem;">
                <span class="risk-badge" style="background: {badge_color}22; color: {badge_color}; border: 2px solid {badge_color};">
                    {score['risk_level']}
                </span>
            </div>
            <div class="score-label" style="margin-top: 1rem;">Risk Classification</div>
        </div>
        """, unsafe_allow_html=True)

    with col_decision:
        st.markdown(f"""
        <div class="score-card">
            <p style="font-size: 1.5rem; font-weight: 700; color: {badge_color}; margin: 0;">
                {recommendation['decision']}
            </p>
            <div class="score-label" style="margin-top: 0.75rem;">Underwriter Decision</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    # ── Flag counts ──
    col_h, col_m, col_l, col_t = st.columns(4)
    with col_h:
        st.markdown(f"""
        <div class="info-card" style="text-align: center; border-left: 3px solid #ff4d4f;">
            <h4>🔴 High</h4>
            <p>{score['high_count']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_m:
        st.markdown(f"""
        <div class="info-card" style="text-align: center; border-left: 3px solid #faad14;">
            <h4>🟡 Medium</h4>
            <p>{score['medium_count']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_l:
        st.markdown(f"""
        <div class="info-card" style="text-align: center; border-left: 3px solid #52c41a;">
            <h4>🟢 Low</h4>
            <p>{score['low_count']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_t:
        st.markdown(f"""
        <div class="info-card" style="text-align: center; border-left: 3px solid #667eea;">
            <h4>📋 Total</h4>
            <p>{score['total_flags']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Documents processed ──
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    st.markdown("#### 📄 Documents Processed")
    for doc in documents:
        label = doc["document_type"].replace("_", " ").title()
        confidence = doc.get("classification_confidence", 0)
        st.markdown(
            f"- **{doc['filename']}** → `{label}` "
            f"(confidence: {confidence:.0%}, "
            f"{doc['fields_found']} fields extracted)"
        )


def _render_tab_extracted_data(results: dict):
    """Tab 2 — Extracted Data."""
    documents = results["documents"]

    if not documents:
        st.info("No documents to display.")
        return

    for doc in documents:
        label = doc["document_type"].replace("_", " ").title()
        st.markdown(f"### 📄 {label} — `{doc['filename']}`")

        fields = doc.get("fields", {})
        if not fields:
            st.warning("No fields could be extracted from this document.")
            continue

        # Build a clean display table
        table_data = []
        for key, value in fields.items():
            display_key = key.replace("_", " ").title()
            display_val = value if value else "—"
            table_data.append({"Field": display_key, "Value": display_val})

        st.table(table_data)

        # Show extraction diagnostics
        found = doc.get("fields_found", 0)
        missing = doc.get("fields_missing", [])
        total = found + len(missing)

        if missing:
            st.caption(
                f"✅ {found}/{total} fields extracted  ·  "
                f"❌ Missing: {', '.join(m.replace('_', ' ').title() for m in missing)}"
            )
        else:
            st.caption(f"✅ All {total} fields extracted successfully")

        st.divider()


def _render_tab_tampering(results: dict):
    tampering_results = results.get("tampering_results", [])
    
    if not tampering_results:
        st.info("No tampering analysis results available.")
        return
    
    for item in tampering_results:
        filename = item["filename"]
        result = item["result"]
        risk_level = result.get("risk_level", "unknown")
        
        risk_colors = {
            "clean": "#52c41a",
            "low_suspicion": "#faad14",
            "suspicious": "#ff7a00",
            "high_risk": "#ff4d4f",
            "unknown": "#8892b0",
        }
        color = risk_colors.get(risk_level, "#8892b0")
        
        st.markdown(f"### 🔬 {filename}")
        st.markdown(
            f"**Overall Assessment:** "
            f"<span style='color:{color}; font-weight:700;'>"
            f"{risk_level.replace('_', ' ').title()}</span>",
            unsafe_allow_html=True,
        )
        st.caption(result.get("summary", ""))
        
        checks = result.get("checks", [])
        if not checks:
            st.warning("No check results available.")
            continue
        
        # Render each check's heatmap in a grid
        cols = st.columns(min(len(checks), 3))
        for i, check in enumerate(checks):
            col = cols[i % 3]
            with col:
                st.markdown(f"**{check['label']}**")
                heatmap = check.get("heatmap_b64")
                if heatmap:
                    st.image(
                        f"data:image/png;base64,{heatmap}",
                        width=300,
                    )
                else:
                    st.caption("No heatmap available.")
                st.caption(check.get("description", ""))
                
                flags = check.get("flags", [])
                for flag in flags:
                    sev = flag.get("severity", "low")
                    if sev == "high":
                        st.error(f"🔴 {flag['message']}")
                    elif sev == "medium":
                        st.warning(f"🟡 {flag['message']}")
                    else:
                        st.info(f"🟢 {flag['message']}")
        
        st.divider()


def _render_tab_cross_doc_flags(results: dict):
    """Tab 3 — Cross-Document Flags."""
    from dataclasses import asdict

    cross_doc_flags = results["cross_doc_flags"]

    if not cross_doc_flags:
        st.success("✅ No cross-document inconsistencies detected — all documents are consistent.")
        return

    st.markdown(f"### ⚠️ {len(cross_doc_flags)} Inconsistency Flag(s) Detected")
    st.markdown("")

    for flag in cross_doc_flags:
        # Normalise to dict if dataclass
        if hasattr(flag, "__dataclass_fields__"):
            flag_dict = asdict(flag)
        else:
            flag_dict = flag

        severity = flag_dict.get("severity", "low")
        check = flag_dict.get("check", "unknown")
        message = flag_dict.get("message", "")
        evidence = flag_dict.get("evidence", {})
        emoji = _severity_emoji(severity)
        color = _severity_color(severity)

        check_title = check.replace("_", " ").title()

        with st.expander(
            f"{emoji} {check_title} — **{severity.upper()}**",
            expanded=(severity == "high"),
        ):
            st.markdown(f"<p style='color: {color}; font-weight: 600;'>{message}</p>",
                        unsafe_allow_html=True)

            if flag_dict.get("similarity") is not None:
                st.markdown(f"**Similarity Score:** {flag_dict['similarity']}%")

            st.markdown("**Evidence:**")
            st.json(evidence)


def _render_tab_metadata(results: dict):
    """Tab 4 — Metadata Analysis."""
    metadata_results = results["metadata_results"]

    if not metadata_results:
        st.info("No PDF documents were processed — metadata analysis was skipped.")
        return

    for meta in metadata_results:
        st.markdown(f"### 📎 {meta['filename']}")

        # Metadata table
        md = meta.get("metadata", {})
        meta_rows = ""
        display_keys = {
            "creation_date": "Creation Date",
            "modification_date": "Modification Date",
            "author": "Author",
            "producer": "Producer Software",
            "file_size_kb": "File Size (KB)",
            "page_count": "Page Count",
        }
        for key, label in display_keys.items():
            val = md.get(key, "—") or "—"
            meta_rows += f"<tr><td>{label}</td><td>{val}</td></tr>"

        st.markdown(f"""
        <table class="meta-table">
            {meta_rows}
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

        # Flags
        flags = meta.get("flags", [])
        if flags:
            for flag in flags:
                severity = flag.get("severity", "low")
                msg = flag.get("message", "")
                emoji = _severity_emoji(severity)
                if severity == "high":
                    st.error(f"{emoji} {msg}")
                elif severity == "medium":
                    st.warning(f"{emoji} {msg}")
                else:
                    st.info(f"{emoji} {msg}")
        else:
            st.success("✅ No suspicious metadata patterns detected.")

        st.divider()


def _render_tab_financial(results: dict):
    """Tab 5 — Financial Analysis."""
    financial_result = results.get("financial_result")

    if not financial_result:
        st.info("No bank statement was uploaded — financial analysis was skipped.")
        return

    chart_data = financial_result.get("chart_data", [])
    flags = financial_result.get("flags", [])
    summary = financial_result.get("summary", {})

    # ── Summary cards ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Credits", f"₹{summary.get('total_credits', 0):,.2f}")
    with col2:
        st.metric("Total Debits", f"₹{summary.get('total_debits', 0):,.2f}")
    with col3:
        st.metric("Months Analysed", summary.get("months_analysed", 0))
    with col4:
        st.metric("Anomalies Found", summary.get("anomalies_found", 0))

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    if not chart_data:
        st.warning("Insufficient data for chart rendering.")
        return

    # ── Plotly chart ──
    months = [d["month"] for d in chart_data]
    credits = [d["credits"] for d in chart_data]
    debits = [d["debits"] for d in chart_data]
    anomaly_mask = [d["is_anomaly"] for d in chart_data]

    fig = go.Figure()

    # Credits line
    fig.add_trace(go.Scatter(
        x=months,
        y=credits,
        mode="lines+markers",
        name="Credits",
        line=dict(color="#667eea", width=3),
        marker=dict(size=8, color="#667eea"),
    ))

    # Debits line
    fig.add_trace(go.Scatter(
        x=months,
        y=debits,
        mode="lines+markers",
        name="Debits",
        line=dict(color="#764ba2", width=3),
        marker=dict(size=8, color="#764ba2"),
    ))

    # Anomaly markers — big red dots
    anomaly_months = [m for m, a in zip(months, anomaly_mask) if a]
    anomaly_credits = [c for c, a in zip(credits, anomaly_mask) if a]
    anomaly_debits = [d for d, a in zip(debits, anomaly_mask) if a]

    if anomaly_months:
        # Mark anomalies on whichever line is higher for that month
        anomaly_values = [
            max(c, d) for c, d in zip(anomaly_credits, anomaly_debits)
        ]
        fig.add_trace(go.Scatter(
            x=anomaly_months,
            y=anomaly_values,
            mode="markers",
            name="⚠ Anomaly",
            marker=dict(
                size=16,
                color="#ff4d4f",
                symbol="diamond",
                line=dict(width=2, color="#fff"),
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,46,0.5)",
        font=dict(family="Inter", color="#ccd6f6"),
        title=dict(text="Monthly Credits & Debits", font=dict(size=18)),
        xaxis=dict(title="Month", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Amount (₹)", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=60, r=30, t=50, b=60),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Anomaly flags ──
    if flags:
        st.markdown("### ⚠️ Financial Anomalies")
        for flag in flags:
            severity = flag.get("severity", "low")
            msg = flag.get("message", "")
            emoji = _severity_emoji(severity)
            if severity == "high":
                st.error(f"{emoji} {msg}")
            elif severity == "medium":
                st.warning(f"{emoji} {msg}")
            else:
                st.info(f"{emoji} {msg}")
    else:
        st.success("✅ No financial anomalies detected.")


def _render_tab_recommendation(results: dict):
    """Tab 6 — Recommendation."""
    recommendation = results["recommendation"]
    score = results["score_result"]
    badge_color = _risk_badge_color(score["color"])

    # ── Decision card ──
    st.markdown(f"""
    <div class="decision-card">
        <p style="font-size: 0.85rem; color: #8892b0; text-transform: uppercase;
                  letter-spacing: 2px; margin-bottom: 0.5rem;">Underwriter Recommendation</p>
        <h2 style="color: {badge_color}; margin: 0 0 1rem 0; font-size: 2rem;">
            {recommendation['decision']}
        </h2>
        <p style="color: #ccd6f6; font-size: 1.05rem; line-height: 1.7;">
            {recommendation['reasoning']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Evidence cards ──
    evidence_cards = recommendation.get("evidence_cards", [])

    if evidence_cards:
        st.markdown("### 📋 Evidence Summary")
        st.markdown(f"*{len(evidence_cards)} item(s) requiring attention:*")
        st.markdown("")

        for card in evidence_cards:
            severity = card.get("severity", "low")
            emoji = _severity_emoji(severity)
            color = _severity_color(severity)

            with st.expander(
                f"{emoji} {card['issue']}",
                expanded=(severity == "high"),
            ):
                st.markdown(f"**Severity:** "
                            f"<span style='color: {color}; font-weight: 700;'>"
                            f"{severity.upper()}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"**Recommended Action:** {card['recommendation']}")
                st.markdown("**Evidence:**")
                st.json(card["evidence"])
    else:
        st.success("✅ No issues to report — the application looks clean.")


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — File upload
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <p style="font-size: 1.8rem; font-weight: 800; margin: 0;
                  background: linear-gradient(135deg, #667eea, #764ba2);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🔍 ForeSight
        </p>
        <p style="font-size: 0.85rem; color: #8892b0; margin-top: 0.25rem;">
            AI-Powered Document Fraud Detection
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
        help="Upload identity proofs, land records, sale deeds, valuation reports, and bank statements.",
    )

    st.markdown("")

    analyze_btn = st.button(
        "🚀 Analyze Documents",
        use_container_width=True,
        type="primary",
        disabled=not uploaded_files,
    )

    st.divider()

    st.markdown("""
    <div style="padding: 0.75rem; background: rgba(102,126,234,0.08);
                border-radius: 8px; border-left: 3px solid #667eea;">
        <p style="font-size: 0.8rem; margin: 0; line-height: 1.6;">
            <strong>Supported documents:</strong><br>
            • Identity Proof (Aadhaar / PAN / Passport)<br>
            • Land Records<br>
            • Sale Deed<br>
            • Valuation Report<br>
            • Bank Statement
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "results" in st.session_state:
        st.markdown("")
        if st.button("🗑️ Clear Results", use_container_width=True):
            del st.session_state["results"]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Main — Processing trigger
# ═══════════════════════════════════════════════════════════════════════════

if analyze_btn and uploaded_files:
    with st.spinner("Processing documents…"):
        results = _run_pipeline(uploaded_files)
        st.session_state["results"] = results


# ═══════════════════════════════════════════════════════════════════════════
# Main — Dashboard rendering
# ═══════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    # Landing state
    st.markdown("""
    <div class="hero-header">
        <h1>ForeSight</h1>
        <p>AI-Powered Document Fraud Detection for Loan Underwriting</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <p style="font-size: 2rem; margin-bottom: 0.5rem;">📄</p>
            <h4 style="font-size: 0.9rem !important;">Upload Documents</h4>
            <p style="font-size: 0.85rem !important; color: #8892b0 !important; font-weight: 400 !important;">
                Upload PDFs or images of identity proofs, land records, sale deeds, and bank statements
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <p style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</p>
            <h4 style="font-size: 0.9rem !important;">AI Analysis</h4>
            <p style="font-size: 0.85rem !important; color: #8892b0 !important; font-weight: 400 !important;">
                OCR extraction, field parsing, cross-document validation, metadata and financial checks
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <p style="font-size: 2rem; margin-bottom: 0.5rem;">✅</p>
            <h4 style="font-size: 0.9rem !important;">Risk Assessment</h4>
            <p style="font-size: 0.85rem !important; color: #8892b0 !important; font-weight: 400 !important;">
                Trust score, risk classification, and actionable underwriter recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info("👈 Upload documents using the sidebar and click **Analyze Documents** to begin.")
    st.stop()


# ── Render the 6 tabs ──
results = st.session_state["results"]

tab1, tab2, tab3, tab4, tab5, tab6 , tab7 = st.tabs([
    "📊 Case Overview",
    "📋 Extracted Data",
    "🔗 Cross-Doc Flags",
    "🔒 Metadata",
    "🔬 Tampering",
    "💰 Financial",
    "✅ Recommendation",
])

with tab1:
    _render_tab_case_overview(results)

with tab2:
    _render_tab_extracted_data(results)

with tab3:
    _render_tab_cross_doc_flags(results)

with tab4:
    _render_tab_metadata(results)

with tab5:
    _render_tab_financial(results)

with tab6:
    _render_tab_recommendation(results)

with tab7:
    _render_tab_tampering(results)
