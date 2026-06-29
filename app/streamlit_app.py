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

# Ensure Homebrew's lib path is added to DYLD_LIBRARY_PATH on macOS for pyzbar to locate libzbar.dylib
if sys.platform == 'darwin':
    # Force PyTorch and Surya OCR to use CPU to prevent segmentation faults (exit code 139)
    # caused by PyTorch initializing MPS inside spawned child processes on macOS.
    if 'TORCH_DEVICE' not in os.environ:
        os.environ['TORCH_DEVICE'] = 'cpu'
    
    # Disable MPS backend globally in PyTorch for Apple Silicon to prevent segfaults
    try:
        import torch
        torch.backends.mps.is_available = lambda: False
        # Limit PyTorch CPU operations to a single thread to prevent OpenMP sync crashes on macOS
        torch.set_num_threads(1)
    except ImportError:
        pass

    # Prevent OpenMP library conflicts from causing crashes
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Disable Tokenizers parallelism to prevent thread conflicts during forks/spawns
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    _brew_lib = '/opt/homebrew/lib'
    if os.path.exists(_brew_lib):
        _current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
        _dyld_paths = _current_dyld.split(':') if _current_dyld else []
        if _brew_lib not in _dyld_paths:
            _dyld_paths.insert(0, _brew_lib)
            os.environ['DYLD_LIBRARY_PATH'] = ':'.join(_dyld_paths)

# ---------------------------------------------------------------------------
# Force HuggingFace libraries and tokenizers to run in offline mode.
# This prevents network calls to huggingface.co for already downloaded models.
# ---------------------------------------------------------------------------
if os.environ.get("FORESIGHT_ONLINE") == "1":
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
else:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


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

from backend.modules.ocr_engine import extract_text, OCRResult
from backend.modules.document_classifier import classify_document
from backend.modules.field_extractor import extract_fields
from backend.modules.cross_document_engine import cross_validate
from backend.modules.metadata_analyzer import analyze_metadata
from backend.modules.financial_anomaly import detect_financial_anomalies
from backend.modules.risk_scorer import calculate_trust_score
from backend.modules.recommendation_engine import generate_recommendation

from backend.modules.validators import run_all as run_format_validators
from backend.modules.api_verifier import route_api
from backend.modules import scoring_engine

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Page config & global styling
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ForeSight — Document Fraud Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-surface:    #1c2128;
    --border:        rgba(0, 229, 255, 0.08);
    --border-hover:  rgba(0, 229, 255, 0.18);
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --accent-cyan:   #00e5ff;
    --accent-blue:   #58a6ff;
    --alert-red:     #dc3545;
    --alert-yellow:  #faad14;
    --alert-green:   #3fb950;
}

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Restore icon font family for Material Symbols/Icons —
   The [class*="st-"] rule above overrides the font on ALL Streamlit elements,
   including icon spans that render Material Symbols ligatures
   (e.g. "double_arrow_right", "arrow_right"). We must restore
   the icon font on every element Streamlit uses to display icons. */
[class*="material-symbols-"],
[class*="material-icons-"],
.material-icons,
.material-symbols-outlined,
.material-symbols-rounded,
.material-symbols-sharp,
/* Streamlit icon elements (wildcard to cover version differences) */
[data-testid*="Icon"] *,
[data-testid*="icon"] *,
[data-testid*="Icon"],
[data-testid*="icon"],
[data-testid*="Collapse"] *,
[data-testid*="collapse"] *,
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] *,
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stSidebarCollapsedControl"] *,
[data-testid="baseButton-headerNoPadding"] *,
button[kind="headerNoPadding"] *,
/* Catch-all: any span inside a Streamlit button in the header area */
header[data-testid="stHeader"] span,
header[data-testid="stHeader"] button span {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
    font-style: normal !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}

/* Hide Streamlit branding and default header controls but keep sidebar collapsed control visible */
#MainMenu, footer {
    visibility: hidden;
}
.stAppDeployButton {
    visibility: hidden;
}
[data-testid="stDecoration"] {
    display: none;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-primary);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Score card ─────────────────────────────────────────────────────────── */
.score-card {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 1.75rem;
    text-align: center;
    border: 1px solid var(--border);
    transition: border-color 0.2s ease;
}
.score-card:hover {
    border-color: var(--border-hover);
}
.score-number {
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
}
.score-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-top: 0.5rem;
    font-weight: 500;
}

/* ── Risk badge ─────────────────────────────────────────────────────────── */
.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Info card ──────────────────────────────────────────────────────────── */
.info-card {
    background: var(--bg-secondary);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    border: 1px solid var(--border);
    margin-bottom: 0.75rem;
    transition: border-color 0.2s ease;
}
.info-card:hover {
    border-color: var(--border-hover);
}
.info-card h4 {
    margin: 0 0 0.25rem 0;
    color: var(--text-secondary);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 500;
}
.info-card p {
    margin: 0;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ── Severity dot ──────────────────────────────────────────────────────── */
.severity-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.severity-high   { background: var(--alert-red);    box-shadow: 0 0 6px rgba(220,53,69,0.4); }
.severity-medium { background: var(--alert-yellow); box-shadow: 0 0 6px rgba(250,173,20,0.4); }
.severity-low    { background: var(--alert-green);   box-shadow: 0 0 6px rgba(63,185,80,0.4); }

/* ── Flag counter badges ───────────────────────────────────────────────── */
.flag-counts {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    margin-top: 1rem;
}
.flag-badge {
    padding: 0.35rem 0.75rem;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Decision card ─────────────────────────────────────────────────────── */
.decision-card {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 1.75rem;
    border: 1px solid var(--border);
    margin-bottom: 1.5rem;
}

/* ── Metadata table ────────────────────────────────────────────────────── */
.meta-table {
    width: 100%;
    border-collapse: collapse;
}
.meta-table td {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
}
.meta-table td:first-child {
    color: var(--text-secondary);
    font-weight: 500;
    width: 40%;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.meta-table td:last-child {
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

/* ── Hero header ───────────────────────────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 3rem 0 1.5rem 0;
}
.hero-header h1 {
    font-size: 2.25rem;
    font-weight: 700;
    color: var(--accent-cyan);
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    font-weight: 400;
}

/* ── Streamlit overrides for minimalism ────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    padding: 0.6rem 1.2rem;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    border-bottom: 2px solid transparent;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}
.stExpander {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    background: var(--bg-secondary) !important;
}
div[data-testid="stMetric"] {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--accent-cyan) !important;
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
        "high": "#dc3545",
        "medium": "#faad14",
        "low": "#e6edf3",
    }.get(severity.lower(), "#8b949e")


def _severity_label_md(severity: str) -> str:
    """Map severity to professional colored text in Markdown."""
    sev = severity.lower()
    if sev == "high":
        return ":red[[HIGH SEVERITY]]"
    elif sev == "medium":
        return ":orange[[MEDIUM SEVERITY]]"
    elif sev == "low":
        return ":gray[[LOW SEVERITY]]"
    elif sev == "accepted":
        return ":green[[ACCEPTED]]"
    return f"[{severity.upper()}]"


def _severity_label_html(severity: str) -> str:
    """Map severity to professional colored text in HTML."""
    sev = severity.lower()
    if sev == "high":
        return "<span style='color: #dc3545; font-weight: bold;'>[HIGH SEVERITY]</span>"
    elif sev == "medium":
        return "<span style='color: #faad14; font-weight: bold;'>[MEDIUM SEVERITY]</span>"
    elif sev == "low":
        return "<span style='color: #e6edf3; font-weight: bold;'>[LOW SEVERITY]</span>"
    elif sev == "accepted":
        return "<span style='color: #3fb950; font-weight: bold;'>[ACCEPTED]</span>"
    return f"<span style='color: #8b949e; font-weight: bold;'>[{severity.upper()}]</span>"


def _risk_badge_color(color: str) -> str:
    """Map the risk_scorer colour name to a hex value."""
    return {
        "green": "#3fb950",
        "orange": "#faad14",
        "red": "#dc3545",
        "darkred": "#a8071a",
    }.get(color, "#8b949e")


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

def _run_pipeline(uploaded_files, mode: str = "offline") -> dict:
    """
    Execute the full ForeSight analysis pipeline on uploaded files.

    Returns a dict with all results stored in session state.
    """
    case_id = _generate_case_id()
    documents: list[dict] = []
    all_metadata_results: list[dict] = []
    financial_result: dict | None = None

    total_steps = len(uploaded_files) * 5 + 5  # OCR+Classify+Extract+Format+API per file, + cross/meta/tamper/fin/score
    current_step = 0
    progress = st.progress(0, text="Starting analysis...")

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
            text=f"Running OCR on {filename}...",
        )
        try:
            ocr_result: OCRResult = extract_text(tmp_path)
            raw_text = ocr_result.text

            if ocr_result.status == "failed":
                st.error(
                    f"**{filename}**: OCR was unable to extract text from this document.  \n"
                    f"This is likely due to a noisy, blurry, or low-quality scan. "
                    f"Please try uploading a clearer copy of the document.  \n"
                    f"{'  \\n'.join(ocr_result.diagnostics)}"
                )
            elif ocr_result.status == "partial":
                st.warning(
                    f"**{filename}**: OCR partially succeeded — "
                    f"{ocr_result.pages_succeeded}/{ocr_result.pages_total} pages "
                    f"processed ({ocr_result.total_chars} chars).  \n"
                    f"{'  \n'.join(ocr_result.diagnostics)}"
                )
            else:
                logger.info(
                    "OCR succeeded for %s: %d chars in %.1fs",
                    filename, ocr_result.total_chars, ocr_result.elapsed_seconds,
                )
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"**{filename}**: {exc}")
            raw_text = ""
            ocr_result = OCRResult(status="failed", diagnostics=[str(exc)])
        except Exception as exc:
            st.warning(f"OCR failed for {filename}: {exc}")
            raw_text = ""
            ocr_result = OCRResult(status="failed", diagnostics=[str(exc)])

        # Step 2: Classify
        current_step += 1
        progress.progress(
            current_step / total_steps,
            text=f"Classifying {filename}...",
        )
        classification = classify_document(raw_text)

        # Step 3: Extract fields
        current_step += 1
        progress.progress(
            current_step / total_steps,
            text=f"Extracting fields from {filename}...",
        )
        if classification.label == "unknown":
            st.warning(
                f"**{filename}**: Could not determine document type "
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
                "ocr_status": ocr_result.status,
                "ocr_diagnostics": ocr_result.diagnostics,
                "ocr_method": ocr_result.method,
                "word_boxes": ocr_result.word_boxes,
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
                    "ocr_status": ocr_result.status,
                    "ocr_diagnostics": ocr_result.diagnostics,
                    "ocr_method": ocr_result.method,
                    "word_boxes": ocr_result.word_boxes,
                }
            except ValueError as exc:
                st.warning(f"**{filename}**: Field extraction failed — {exc}")
                doc_record = {
                    "filename": filename,
                    "tmp_path": tmp_path,
                    "document_type": classification.label,
                    "classification_confidence": classification.confidence,
                    "fields": {},
                    "fields_found": 0,
                    "fields_missing": [],
                    "raw_text_length": len(raw_text),
                    "ocr_status": ocr_result.status,
                    "ocr_diagnostics": ocr_result.diagnostics,
                    "ocr_method": ocr_result.method,
                    "word_boxes": ocr_result.word_boxes,
                }

        # Step 4: Format & Checksum Validation
        current_step += 1
        progress.progress(
            current_step / total_steps,
            text=f"Running format validation on {filename}...",
        )
        format_res = run_format_validators(doc_record.get("fields", {}))
        doc_record["format_checksum"] = format_res

        # Step 5: API Verification
        current_step += 1
        progress.progress(
            current_step / total_steps,
            text=f"Running API verification on {filename}...",
        )
        api_fields = dict(doc_record.get("fields", {}))
        api_fields["image_path"] = tmp_path
        api_res = route_api(doc_record["document_type"], api_fields, mode=mode)
        doc_record["api_verification"] = api_res

        documents.append(doc_record)

    # ------------------------------------------------------------------
    # Phase 2 — Cross-document validation
    # ------------------------------------------------------------------
    current_step += 1
    progress.progress(
        current_step / total_steps,
        text="Running cross-document checks...",
    )
    cross_doc_flags = cross_validate(documents)

    # ------------------------------------------------------------------
    # Phase 3 — Metadata analysis (deprecated and removed)
    all_metadata_flags: list[dict] = []

    
    # Phase 3b — Visual tampering analysis (per document)
    current_step += 1
    progress.progress(
        current_step / total_steps,
        text="Running visual tampering analysis...",
    )
    all_tampering_results = []
    for doc in documents:
        path = doc["tmp_path"]
        ext = os.path.splitext(path)[1].lower()
        ocr_word_tuples = []
        for w in doc.get("word_boxes", []):
            b = w["box"]
            ocr_word_tuples.append((b[0], b[1], b[2], b[3], w["text"]))
        try:
            if ext == ".pdf":
                tamper_result = detect_tampering_from_pdf_page(
                    path,
                    ocr_word_boxes=ocr_word_tuples,
                    document_type=doc.get("document_type")
                )
            else:
                tamper_result = detect_tampering(
                    path,
                    ocr_word_boxes=ocr_word_tuples,
                    document_type=doc.get("document_type")
                )
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
        text="Detecting financial anomalies...",
    )
    financial_flags: list[dict] = []
    for doc in documents:
        if doc["document_type"] == "bank_statement":
            financial_result = detect_financial_anomalies(
                monthly_credits=doc["fields"].get("monthly_credits"),
                monthly_debits=doc["fields"].get("monthly_debits"),
                month_labels=doc["fields"].get("month_labels"),
            )
            financial_flags = financial_result.get("flags", [])
            break  # only one bank statement expected

    # ------------------------------------------------------------------
    # Phase 5 — Trust score & recommendation
    # ------------------------------------------------------------------
    current_step += 1
    progress.progress(
        current_step / total_steps,
        text="Calculating trust score...",
    )

    # 1. Format/Checksum
    format_flags = []
    worst_format_risk = "clean"
    highest_format_pw = 0.0
    for doc in documents:
        f_res = doc.get("format_checksum")
        if f_res:
            format_flags.extend(f_res.get("flags", []))
            if f_res.get("penalty_weight", 0.0) > highest_format_pw:
                highest_format_pw = f_res["penalty_weight"]
            if f_res.get("risk") == "high":
                worst_format_risk = "high"
            elif f_res.get("risk") == "medium" and worst_format_risk != "high":
                worst_format_risk = "medium"
    
    format_checksum_sec = {
        "section": "format_checksum",
        "risk": worst_format_risk,
        "flags": format_flags,
        "penalty_weight": highest_format_pw
    }

    # 2. API Verification
    api_flags = []
    worst_api_risk = "clean"
    highest_api_pw = 0.0
    for doc in documents:
        a_res = doc.get("api_verification")
        if a_res:
            for f in a_res.get("flags", []):
                severity = "high"
                if f == "api_name_mismatch":
                    severity = "high"
                elif f in ["missing_id_number", "api_unreachable", "api_not_available"]:
                    severity = "low"
                api_flags.append({
                    "check": f,
                    "severity": severity,
                    "message": f"API verification flag: {f}",
                    "evidence": a_res.get("api_result")
                })
            if a_res.get("penalty_weight", 0.0) > highest_api_pw:
                highest_api_pw = a_res["penalty_weight"]
            if a_res.get("risk") == "high":
                worst_api_risk = "high"
            elif a_res.get("risk") == "medium" and worst_api_risk != "high":
                worst_api_risk = "medium"

    api_verification_sec = {
        "section": "api_verification",
        "risk": worst_api_risk,
        "flags": api_flags,
        "penalty_weight": highest_api_pw
    }

    # 3. Cross-Doc Consistency
    cross_doc_flags_mapped = []
    worst_cross_risk = "clean"
    highest_cross_pw = 0.0
    
    from dataclasses import asdict
    for f in cross_doc_flags:
        if hasattr(f, "__dataclass_fields__"):
            f_dict = asdict(f)
        else:
            f_dict = f
        severity = f_dict.get("severity", "low")
        check = f_dict.get("check", "unknown")
        pw = 0.0
        if check == "name_consistency":
            pw = 0.45
        elif check == "property_id_consistency":
            pw = 0.40
        elif check == "timeline_consistency":
            pw = 0.30 if severity == "high" else 0.25
        
        if pw > highest_cross_pw:
            highest_cross_pw = pw
        if severity == "high":
            worst_cross_risk = "high"
        elif severity == "medium" and worst_cross_risk != "high":
            worst_cross_risk = "medium"
            
        cross_doc_flags_mapped.append({
            "check": check,
            "severity": severity,
            "message": f_dict.get("message", ""),
            "penalty_weight": pw,
            "evidence": f_dict.get("evidence", {})
        })
        
    cross_doc_sec = {
        "section": "cross_doc_consistency",
        "risk": worst_cross_risk,
        "flags": cross_doc_flags_mapped,
        "penalty_weight": highest_cross_pw
    }

    # 4. Image Tampering
    tampering_flags = []
    worst_tamp_risk = "clean"
    highest_tamp_pw = 0.0
    for tr in all_tampering_results:
        res = tr["result"]
        for f in res.get("flags", []):
            tampering_flags.append(f)
        if res.get("penalty_weight", 0.0) > highest_tamp_pw:
            highest_tamp_pw = res["penalty_weight"]
        if res.get("risk_level") == "high_risk":
            worst_tamp_risk = "high"
        elif res.get("risk_level") in ["suspicious", "low_suspicion"] and worst_tamp_risk != "high":
            worst_tamp_risk = "medium"
            
    image_tampering_sec = {
        "section": "image_tampering",
        "risk": worst_tamp_risk,
        "flags": tampering_flags,
        "penalty_weight": highest_tamp_pw
    }

    # 5. Financial Anomaly
    financial_flags = []
    worst_fin_risk = "clean"
    highest_fin_pw = 0.0
    if financial_result:
        financial_flags = financial_result.get("flags", [])
        for f in financial_flags:
            severity = f.get("severity", "low").lower()
            pw = 0.20 if severity == "high" else (0.12 if severity == "medium" else 0.03)
            if pw > highest_fin_pw:
                highest_fin_pw = pw
            if severity == "high":
                worst_fin_risk = "high"
            elif severity == "medium" and worst_fin_risk != "high":
                worst_fin_risk = "medium"
                
    financial_sec = {
        "section": "financial_anomaly",
        "risk": worst_fin_risk,
        "flags": financial_flags,
        "penalty_weight": highest_fin_pw
    }

    section_results = [
        format_checksum_sec,
        api_verification_sec,
        cross_doc_sec,
        image_tampering_sec,
        financial_sec
    ]

    score_res = scoring_engine.compute_final_score(section_results)
    
    # Map back to UI risk scorer structure
    trust_score = int(score_res["final_score"] * 100)
    
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
        
    all_flags_flat = []
    for contrib in score_res.get("section_contributions", []):
        for f in contrib.get("flags", []):
            f_mapped = dict(f)
            f_mapped["penalty"] = int(f.get("penalty_weight", 0.0) * 100)
            all_flags_flat.append(f_mapped)

    score_result = {
        "trust_score": trust_score,
        "risk_level": risk_level,
        "color": color,
        "total_flags": len(all_flags_flat),
        "high_count": sum(1 for f in all_flags_flat if f.get("severity") == "high"),
        "medium_count": sum(1 for f in all_flags_flat if f.get("severity") == "medium"),
        "low_count": sum(1 for f in all_flags_flat if f.get("severity") == "low"),
        "all_flags": all_flags_flat,
        "hard_kill_triggered": score_res["final_score"] <= 0.20 and any(f.get("check") in ["name_consistency", "property_id_consistency"] for f in all_flags_flat if f.get("severity") == "high"),
        "base_deductions": 100 - trust_score,
        "hard_kill_reduction": 0,
        "tampering_reduction": 0,
    }
    
    recommendation = generate_recommendation(score_result)
    
    progress.progress(1.0, text="Analysis complete!")

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

    # ── Hard-kill warning banner ──
    if results["score_result"].get("hard_kill_triggered"):
        st.markdown("""
        <div style="background: rgba(220,53,69,0.12); border: 1px solid #dc3545;
                    border-radius: 6px; padding: 1rem 1.25rem; margin-top: 1rem;">
            <p style="color: #dc3545; font-weight: 700; margin: 0; font-size: 0.9rem;">
                HARD KILL RULE TRIGGERED
            </p>
            <p style="color: #e6edf3; margin: 0.4rem 0 0 0; font-size: 0.85rem;">
                A critical fraud indicator (identity or property mismatch) was
                detected. Score has been capped regardless of other signals.
                This application requires immediate escalation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Flag counts ──
    col_h, col_m, col_l, col_t = st.columns(4)
    with col_h:
        st.markdown(f"""
        <div class="info-card" style="text-align: center; border-left: 3px solid #dc3545;">
            <h4>High Severity</h4>
            <p>{score['high_count']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_m:
        st.markdown(f"""
        <div class="info-card" style="text-align: center; border-left: 3px solid #faad14;">
            <h4>Medium Severity</h4>
            <p>{score['medium_count']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_l:
        st.markdown(f"""
        <div class="info-card" style="text-align: center; border-left: 3px solid #e6edf3;">
            <h4>Low Severity</h4>
            <p>{score['low_count']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_t:
        st.markdown(f"""
        <div class="info-card" style="text-align: center; border-left: 3px solid #00e5ff;">
            <h4>Total Flags</h4>
            <p>{score['total_flags']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Documents processed ──
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    st.markdown("#### Documents Processed")
    for doc in documents:
        label = doc["document_type"].replace("_", " ").title()
        confidence = doc.get("classification_confidence", 0)
        ocr_status = doc.get("ocr_status", "success")
        ocr_badge = {
            "success": _severity_label_md("accepted"),
            "partial": _severity_label_md("medium"),
            "failed": _severity_label_md("high"),
        }.get(ocr_status, f"[{ocr_status.upper()}]")
        st.markdown(
            f"- {ocr_badge} **{doc['filename']}** → `{label}` "
            f"(confidence: {confidence:.0%}, "
            f"{doc['fields_found']} fields extracted, "
            f"OCR: {ocr_status})"
        )

    # ── Score Impact Breakdown ──
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("#### Score Impact Breakdown")
    
    breakdown_rows = ""
    # Base Score
    breakdown_rows += (
        "<tr>"
        "<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem;'>Base Score</td>"
        f"<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem;'>{_severity_label_html('accepted')}</td>"
        "<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); color:#3fb950; font-weight:bold; font-family:monospace; font-size: 0.9rem;'>+100</td>"
        "</tr>"
    )
    
    for flag in score.get("all_flags", []):
        check_name = flag.get("check", "unknown").replace("_", " ").title()
        severity = flag.get("severity", "low")
        penalty = flag.get("penalty", 0)
        msg = flag.get("message", "")
        if penalty > 0:
            breakdown_rows += (
                "<tr>"
                f"<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem;'>Deduction: {check_name}<br><small style='color:#8b949e;'>{msg}</small></td>"
                f"<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem;'>{_severity_label_html(severity)}</td>"
                f"<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); color:#dc3545; font-weight:bold; font-family:monospace; font-size: 0.9rem;'>-{penalty}</td>"
                "</tr>"
            )
            
    if score.get("hard_kill_reduction", 0) > 0:
        breakdown_rows += (
            "<tr>"
            "<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem;'>Capped score adjustment (Critical Identity/Property Mismatch)</td>"
            f"<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem;'>{_severity_label_html('high')}</td>"
            f"<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); color:#dc3545; font-weight:bold; font-family:monospace; font-size: 0.9rem;'>-{score['hard_kill_reduction']}</td>"
            "</tr>"
        )
        
    if score.get("tampering_reduction", 0) > 0:
        breakdown_rows += (
            "<tr>"
            "<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem;'>Compounding Tampering safety factor (2+ visual tampering indicators)</td>"
            f"<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem;'>{_severity_label_html('high')}</td>"
            f"<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); color:#dc3545; font-weight:bold; font-family:monospace; font-size: 0.9rem;'>-{score['tampering_reduction']}</td>"
            "</tr>"
        )
        
    # Final Trust Score Row
    final_color = _risk_badge_color(score["color"])
    breakdown_rows += (
        "<tr style='background: rgba(255,255,255,0.02); font-weight:bold; border-top: 2px solid var(--border);'>"
        "<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem;'>Final Trust Score</td>"
        "<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.9rem;'>—</td>"
        f"<td style='padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); color:{final_color}; font-size:1.15rem; font-family:monospace;'>{score['trust_score']}</td>"
        "</tr>"
    )
    
    st.markdown(f"""
    <table style="width:100%; border-collapse:collapse; margin-top:0.5rem; border: 1px solid var(--border);">
        <thead>
            <tr style="border-bottom: 2px solid var(--border); text-align:left; background: rgba(255,255,255,0.02);">
                <th style="padding:0.5rem 1rem; color:var(--text-secondary); font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Pattern / Indicator</th>
                <th style="padding:0.5rem 1rem; color:var(--text-secondary); font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Severity</th>
                <th style="padding:0.5rem 1rem; color:var(--text-secondary); font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Impact</th>
            </tr>
        </thead>
        <tbody>
            {breakdown_rows}
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
    st.divider()
    _render_tab_extracted_data(results)


def _render_tab_extracted_data(results: dict):
    """Tab 2 — Extracted Data."""
    documents = results["documents"]

    if not documents:
        st.info("No documents to display.")
        return

    for doc in documents:
        label = doc["document_type"].replace("_", " ").title()
        st.markdown(f"### {label} — `{doc['filename']}`")

        # Show OCR status warning if applicable
        ocr_status = doc.get("ocr_status", "success")
        if ocr_status == "failed":
            st.error(
                f"**OCR failed** — unable to extract text from this document. "
                f"This is likely due to a noisy, blurry, or low-quality scan. "
                f"Please try uploading a clearer copy.  \n"
                + "  \n".join(doc.get("ocr_diagnostics", []))
            )
        elif ocr_status == "partial":
            st.warning(
                f"**Partial OCR** — some pages could not be processed.  \n"
                + "  \n".join(doc.get("ocr_diagnostics", []))
            )

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
                f"{_severity_label_md('accepted')} {found}/{total} fields extracted  ·  "
                f"{_severity_label_md('high')} Missing: {', '.join(m.replace('_', ' ').title() for m in missing)}"
            )
        else:
            st.caption(f"{_severity_label_md('accepted')} All {total} fields extracted successfully")

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
            "clean": "#3fb950",
            "low_suspicion": "#faad14",
            "suspicious": "#ff7a00",
            "high_risk": "#dc3545",
            "unknown": "#8b949e",
        }
        color = risk_colors.get(risk_level, "#8b949e")
        
        st.markdown(f"### {filename}")
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
        
        # Render each check with its heatmap on the left and explanation/severity on the right
        for idx, check in enumerate(checks):
            with st.container():
                col_left, col_right = st.columns([1.2, 1.5])
                with col_left:
                    heatmap = check.get("heatmap_b64")
                    if heatmap:
                        st.image(
                            f"data:image/png;base64,{heatmap}",
                            width='stretch',
                        )
                    else:
                        st.caption("No heatmap available for this check.")
                with col_right:
                    st.markdown(f"#### {check['label']}")
                    st.markdown(f"**Description:** {check.get('description', '')}")
                    
                    flags = check.get("flags", [])
                    if flags:
                        st.markdown("**Anomalies Detected:**")
                        for flag in flags:
                            sev = flag.get("severity", "low")
                            if sev == "high":
                                st.error(f"{_severity_label_md('high')} {flag['message']}")
                            elif sev == "medium":
                                st.warning(f"{_severity_label_md('medium')} {flag['message']}")
                            else:
                                st.info(f"{_severity_label_md('low')} {flag['message']}")
                    else:
                        st.success("✅ No anomalies detected.")
            
            if idx < len(checks) - 1:
                st.divider()
        
        st.divider()


def _render_tab_cross_doc_flags(results: dict):
    """Tab 3 — Cross-Document Flags."""
    from dataclasses import asdict

    cross_doc_flags = results["cross_doc_flags"]

    if not cross_doc_flags:
        st.success("No cross-document inconsistencies detected — all documents are consistent.")
    else:
        st.markdown(f"### {len(cross_doc_flags)} Inconsistency Flag(s) Detected")
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
            color = _severity_color(severity)

            check_title = check.replace("_", " ").title()

            with st.expander(
                f"{_severity_label_md(severity)} {check_title}",
                expanded=(severity == "high"),
            ):
                st.markdown(f"<p style='color: {color}; font-weight: 600;'>{message}</p>",
                            unsafe_allow_html=True)

                if flag_dict.get("similarity") is not None:
                    st.markdown(f"**Similarity Score:** {flag_dict['similarity']}%")

                st.markdown("**Evidence:**")
                st.json(evidence)







def _render_tab_financial(results: dict):
    """Tab 5 — Financial Analysis."""
    financial_result = results.get("financial_result")

    if not financial_result:
        st.info("No bank statement was uploaded — financial analysis was skipped.")
        return

    # Data quality warning for insufficient per-month data
    data_quality = financial_result.get("data_quality", "ok")
    if data_quality == "insufficient_for_monthly_analysis":
        st.warning(
            "Only aggregate totals were extracted from the bank statement — "
            "per-month anomaly analysis requires at least 3 months of data. "
            "Upload a multi-month bank statement for full analysis."
        )

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
        line=dict(color="#00e5ff", width=2),
        marker=dict(size=6, color="#00e5ff"),
    ))

    # Debits line
    fig.add_trace(go.Scatter(
        x=months,
        y=debits,
        mode="lines+markers",
        name="Debits",
        line=dict(color="#58a6ff", width=2),
        marker=dict(size=6, color="#58a6ff"),
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
            name="Anomaly",
            marker=dict(
                size=14,
                color="#dc3545",
                symbol="diamond",
                line=dict(width=1, color="#e6edf3"),
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.6)",
        font=dict(family="Inter", color="#e6edf3"),
        title=dict(text="Monthly Credits & Debits", font=dict(size=14, color="#8b949e")),
        xaxis=dict(title="Month", gridcolor="rgba(0,229,255,0.04)"),
        yaxis=dict(title="Amount (₹)", gridcolor="rgba(0,229,255,0.04)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=60, r=30, t=50, b=60),
        height=420,
    )

    st.plotly_chart(fig, width='stretch')

    # ── Anomaly flags ──
    if flags:
        st.markdown("### Financial Anomalies")
        for flag in flags:
            severity = flag.get("severity", "low")
            msg = flag.get("message", "")
            if severity == "high":
                st.error(f"{_severity_label_md('high')} {msg}")
            elif severity == "medium":
                st.warning(f"{_severity_label_md('medium')} {msg}")
            else:
                st.info(f"{_severity_label_md('low')} {msg}")
    else:
        st.success("No financial anomalies detected.")


def _render_tab_recommendation(results: dict):
    """Tab 6 — Recommendation."""
    recommendation = results["recommendation"]
    score = results["score_result"]
    badge_color = _risk_badge_color(score["color"])

    # ── Decision card ──
    st.markdown(f"""
    <div class="decision-card">
        <p style="font-size: 0.7rem; color: #8b949e; text-transform: uppercase;
                  letter-spacing: 2.5px; margin-bottom: 0.5rem;">Underwriter Recommendation</p>
        <h2 style="color: {badge_color}; margin: 0 0 1rem 0; font-size: 1.75rem; font-weight: 700;">
            {recommendation['decision']}
        </h2>
        <p style="color: #e6edf3; font-size: 0.95rem; line-height: 1.7;">
            {recommendation['reasoning']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Evidence cards ──
    evidence_cards = recommendation.get("evidence_cards", [])

    if evidence_cards:
        st.markdown("### Evidence Summary")
        st.markdown(f"*{len(evidence_cards)} item(s) requiring attention:*")
        st.markdown("")

        for card in evidence_cards:
            severity = card.get("severity", "low")
            color = _severity_color(severity)

            with st.expander(
                f"{_severity_label_md(severity)} {card['issue']}",
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
        st.success("No issues to report — the application looks clean.")


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — File upload
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <p style="font-size: 1.5rem; font-weight: 700; margin: 0;
                  color: #00e5ff; letter-spacing: -0.5px;">
            ForeSight
        </p>
        <p style="font-size: 0.75rem; color: #8b949e; margin-top: 0.25rem;
                  text-transform: uppercase; letter-spacing: 2px;">
            Document Fraud Detection
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

    # API Verification Mode Selector
    verification_mode = st.selectbox(
        "API Verification Mode",
        ["Offline (Local Checksums & Aadhaar QR)", "Online (Surepass Live APIs)"],
        index=0,
        help="Online mode validates documents (PAN, GSTIN, DL, etc.) against official government databases using Surepass APIs, requiring SUREPASS_API_KEY. Offline mode uses local mathematical checksums and parses Aadhaar secure QR codes locally."
    )
    mode = "online" if "Online" in verification_mode else "offline"

    st.markdown("")

    analyze_btn = st.button(
        "Analyze Documents",
        width='stretch',
        type="primary",
        disabled=not uploaded_files,
    )

    st.divider()

    st.markdown("""
    <div style="padding: 0.75rem; background: rgba(0,229,255,0.04);
                border-radius: 6px; border-left: 2px solid rgba(0,229,255,0.3);">
        <p style="font-size: 0.75rem; margin: 0; line-height: 1.7; color: #8b949e;">
            <strong style="color: #e6edf3;">Supported documents</strong><br>
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
        if st.button("Clear Results", width='stretch'):
            del st.session_state["results"]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Main — Processing trigger
# ═══════════════════════════════════════════════════════════════════════════

if analyze_btn and uploaded_files:
    with st.spinner("Processing documents…"):
        results = _run_pipeline(uploaded_files, mode=mode)
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
            <div style="height: 1rem;"></div>
            <h4 style="font-size: 0.8rem !important; color: #e6edf3 !important;">Upload</h4>
            <p style="font-size: 0.78rem !important; color: #8b949e !important; font-weight: 400 !important;">
                PDFs or images — identity proofs, land records, sale deeds, bank statements
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div style="height: 1rem;"></div>
            <h4 style="font-size: 0.8rem !important; color: #e6edf3 !important;">Analyze</h4>
            <p style="font-size: 0.78rem !important; color: #8b949e !important; font-weight: 400 !important;">
                OCR, field extraction, cross-document validation, metadata & financial checks
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div style="height: 1rem;"></div>
            <h4 style="font-size: 0.8rem !important; color: #e6edf3 !important;">Assess</h4>
            <p style="font-size: 0.78rem !important; color: #8b949e !important; font-weight: 400 !important;">
                Trust score, risk classification, and actionable underwriter recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info("Upload documents using the sidebar and click **Analyze Documents** to begin.")
    st.stop()


def _render_tab_format_checksums(results: dict):
    """Tab 7 — Format & Checksums validation."""
    st.markdown("### Format & Checksum Verification")
    st.markdown("Validates key document fields using official formats, regular expressions, and mathematical checksums.")
    st.markdown("")

    documents = results.get("documents", [])
    has_checksums = False

    for doc in documents:
        f_res = doc.get("format_checksum")
        if not f_res:
            continue
        
        has_checksums = True
        st.markdown(f"#### {doc['filename']} ({doc['document_type'].replace('_', ' ').title()})")
        
        details = f_res.get("details", {})
        if not details:
            st.info("No formatting checks applicable for this document type.")
            st.markdown("")
            continue
            
        rows = []
        for field, check_info in details.items():
            status = check_info.get("status", "skipped")
            val = check_info.get("value") or "-"
            msg = check_info.get("message") or "-"
            
            # Map status to colored badge
            if status == "valid":
                badge = "🟢 Valid"
            elif status == "invalid":
                badge = "🔴 Invalid"
            else:
                badge = "⚪ Skipped"
                
            rows.append({
                "Field": field.replace("_", " ").title(),
                "Extracted Value": val,
                "Status": badge,
                "Details": msg
            })
            
        if rows:
            st.table(rows)
        st.markdown("")

    if not has_checksums:
        st.info("No documents with format validation checks were uploaded.")


def _render_tab_api_verification(results: dict):
    """Tab 8 — Government API Verification."""
    st.markdown("### Identity & Entity API Verification")
    st.markdown("Cross-verifies document details against government registers (Surepass Online API or Offline QR signature checks).")
    st.markdown("")

    documents = results.get("documents", [])
    has_api = False

    for doc in documents:
        api_res = doc.get("api_verification")
        if not api_res:
            continue

        has_api = True
        st.markdown(f"#### {doc['filename']} ({doc['document_type'].replace('_', ' ').title()})")

        risk = api_res.get("risk", "clean")
        api_result = api_res.get("api_result", {})
        
        if risk == "clean":
            st.success("Verification Status: Approved & Verified")
        elif risk == "medium":
            st.warning("Verification Status: Warning / Discrepancy Found")
        elif risk == "high":
            st.error("Verification Status: Verification Failed")
        else:
            st.info("Verification Status: Skipped")

        if not api_result or "error" in api_result:
            err_msg = api_result.get("error", "No verification results available.")
            st.caption(f"Verification Details: {err_msg.replace('_', ' ').title()}")
            st.markdown("")
            continue

        # Aadhaar Card layout
        if doc["document_type"] == "aadhaar_card":
            # Display layout columns with photo if present
            col1, col2 = st.columns([1, 3])
            with col1:
                photo_b64 = api_result.get("photo_b64")
                if photo_b64:
                    st.markdown(f"""
                    <div style="border: 2px solid #30363d; border-radius: 6px; padding: 4px; background: #0d1117; display: inline-block;">
                        <img src="data:image/jpeg;base64,{photo_b64}" style="width: 140px; border-radius: 4px;" />
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.caption("No photo extracted")
            with col2:
                st.markdown(f"**Verified Name:** {api_result.get('name') or '-'}")
                st.markdown(f"**Date of Birth:** {api_result.get('dob') or '-'}")
                st.markdown(f"**Gender:** {api_result.get('gender') or '-'}")
                st.markdown(f"**Address:** {api_result.get('address') or '-'}")
                
                sig_valid = api_result.get("signature_valid")
                if sig_valid:
                    st.markdown("🔑 **Signature Validation:** 🟢 Cryptographically Signed & Verified by UIDAI")
                elif sig_valid is False:
                    st.markdown("🔑 **Signature Validation:** 🔴 Cryptographic Signature Verification Failed")
                else:
                    st.markdown("🔑 **Signature Validation:** ⚪ Unknown/No Signature")

        else:
            # Standard government database lookup result (PAN, GSTIN, DL, etc.)
            st.markdown("**API Response Data:**")
            st.json(api_result)

        st.markdown("")

    if not has_api:
        st.info("No documents with API verification checks were processed.")


# ── Render the tabs ──
results = st.session_state["results"]

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Format & Checksum Validation",
    "API Verification",
    "Cross Document Consistency",
    "Image Tampering",
    "Financial Analysis",
    "Recommendations",
])

with tab1:
    _render_tab_case_overview(results)

with tab2:
    _render_tab_format_checksums(results)

with tab3:
    _render_tab_api_verification(results)

with tab4:
    _render_tab_cross_doc_flags(results)

with tab5:
    _render_tab_tampering(results)

with tab6:
    _render_tab_financial(results)

with tab7:
    _render_tab_recommendation(results)
