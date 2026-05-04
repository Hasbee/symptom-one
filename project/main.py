"""SymptomOne - Medical Assessment System (Streamlit UI)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

INPUT_DIR = PROJECT_ROOT / "data" / "input"

FIELD_LABELS = {
    "patient_age_years": "Age (years)",
    "symptom_duration_hours": "Symptom duration (hours)",
    "fever_present": "Fever present (0/1)",
    "neck_stiffness": "Neck stiffness (0/1)",
    "body_temperature_celsius": "Temperature (°C)",
    "heart_rate_bpm": "Heart rate (bpm)",
    "blood_pressure_systolic_mmhg": "BP systolic (mmHg)",
    "blood_pressure_diastolic_mmhg": "BP diastolic (mmHg)",
    "respiratory_rate_breaths_per_minute": "Respiratory rate",
    "oxygen_saturation_percent": "SpO₂ (%)",
    "comorbidities_count": "Comorbidities count",
    "symptom_description": "Symptom description",
    "symptom_type": "Symptom type",
}


@st.cache_data(ttl=600, show_spinner=False)
def load_model_evaluation() -> Dict[str, Any]:
    """Cached ML evaluation metrics (optional sidebar)."""
    try:
        from ml.evaluation import evaluate_all_models

        return evaluate_all_models()
    except Exception as exc:
        return {"status": "error", "_exception": str(exc)}


def get_available_json_files() -> List[Path]:
    if not INPUT_DIR.is_dir():
        return []
    return sorted(INPUT_DIR.glob("user_query_*.json"))


def get_severity_color(level: Optional[str]) -> str:
    if not level:
        return "#6b7280"
    m = {
        "Low": "#22c55e",
        "Moderate": "#eab308",
        "High": "#f97316",
        "Critical": "#ef4444",
    }
    return m.get(str(level).strip(), "#6b7280")


def format_clinical_field(key: str) -> str:
    return FIELD_LABELS.get(key, key.replace("_", " ").title())


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem; max-width: 1100px; }
            h1 { font-weight: 600; letter-spacing: -0.02em; }
            div[data-testid="stMetricValue"] { font-size: 1.35rem; }
            .risk-pill {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-weight: 600;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_assessment(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    from workflow import workflow_instance

    return workflow_instance.run_workflow(patient_data)


def main() -> None:
    st.set_page_config(
        page_title="SymptomOne",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.title("SymptomOne")
    st.caption("Clinical assessment workflow — educational demo; not a substitute for professional care.")

    if "phase" not in st.session_state:
        st.session_state.phase = "input"
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    with st.sidebar:
        st.subheader("Model evaluation")
        ev = load_model_evaluation()
        if ev.get("_exception"):
            st.warning(ev["_exception"])
        elif ev.get("status") in ("ok", "success"):
            st.success("Models evaluated on holdout data.")
            models = ev.get("models", {})
            if models:
                for model_name, model_info in models.items():
                    st.caption(model_name.replace("_", " ").title())
                    metrics = model_info.get("metrics", {})
                    if metrics:
                        # Display metrics as a table
                        st.table(
                            {k.replace("_", " ").title(): [v] for k, v in metrics.items()}
                        )
            else:
                # Fallback for old format
                for name in ("severity", "risk_score"):
                    b = ev.get(name, {})
                    if b.get("metrics"):
                        st.caption(name.replace("_", " ").title())
                        st.json(b["metrics"])
        else:
            st.info(str(ev.get("status", "partial")))

        st.divider()
        st.caption(f"Input samples: `{INPUT_DIR.name}/`")

    if st.session_state.phase == "results" and st.session_state.last_result:
        result: Dict[str, Any] = st.session_state.last_result
        if "error" in result:
            st.error(result["error"])
            if st.button("← Back to input"):
                st.session_state.phase = "input"
                st.session_state.last_result = None
                st.rerun()
            return

        ui = result.get("ui_output") or {}
        sev = ui.get("severity_level") or result.get("severity_level")
        risk = ui.get("risk_score")
        if risk is None:
            risk = result.get("risk_score", 0.0)
        rlevel = ui.get("risk_level", "—")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Session", str(ui.get("session_id") or result.get("session_id", "—"))[:12] + "…")
        with col_b:
            st.markdown(
                f'<p style="color:{get_severity_color(sev)};font-weight:600;">Severity: {sev or "—"}</p>',
                unsafe_allow_html=True,
            )
        with col_c:
            st.metric("Risk score", f"{float(risk):.2f}")
        with col_d:
            st.metric("Risk band", rlevel)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Symptom analysis", "Clinical data", "Diagnosis & treatment", "Clinical guidance"]
        )

        with tab1:
            st.json(ui.get("symptom_classification") or result.get("symptom_classification", {}))

        with tab2:
            ext = result.get("extracted_data") or {}
            rows = [{"Field": format_clinical_field(k), "Value": v} for k, v in ext.items()]
            st.dataframe(rows, use_container_width=True, hide_index=True)

        with tab3:
            st.subheader("Differential diagnoses")
            st.json(ui.get("differential_diagnoses") or {})
            st.subheader("Treatment plan")
            st.json(ui.get("treatment_plan") or {})

        with tab4:
            st.markdown(ui.get("health_advice") or result.get("health_advice") or "_No advice text._")
            st.divider()
            st.caption("Validation")
            st.write(ui.get("validation_status") or result.get("validation_status"))
            errs = ui.get("validation_errors") or result.get("validation_errors") or []
            if errs:
                for e in errs:
                    st.warning(e)

        report = result.get("report_json") or {}
        path = result.get("report_file_path")
        dl = json.dumps(report, indent=2)
        st.download_button(
            "Download report JSON",
            data=dl,
            file_name=f"{result.get('session_id', 'report')}.json",
            mime="application/json",
        )
        if path:
            st.caption(f"Saved: `{path}`")

        if st.button("← New assessment"):
            st.session_state.phase = "input"
            st.session_state.last_result = None
            st.rerun()
        return

    # --- Input phase ---
    st.subheader("Patient input")
    files = get_available_json_files()
    labels = [p.name for p in files]
    choice = st.selectbox("Load sample JSON", options=["—"] + labels, index=0)

    default_text = ""
    if choice != "—":
        try:
            default_text = files[labels.index(choice)].read_text(encoding="utf-8")
        except OSError:
            default_text = ""

    raw = st.text_area(
        "Patient JSON (11 clinical fields required; extra keys allowed)",
        value=default_text,
        height=220,
        placeholder='{"patient_age_years": 35, ...}',
    )

    c1, c2 = st.columns(2)
    with c1:
        run = st.button("Run assessment", type="primary", use_container_width=True)
    with c2:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    if run:
        try:
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return
        if not isinstance(data, dict):
            st.error("Root JSON value must be an object.")
            return

        with st.spinner("Running workflow…"):
            out = run_assessment(data)
        st.session_state.last_result = out
        st.session_state.phase = "results"
        st.rerun()


if __name__ == "__main__":
    main()


"""SymptomOne - Medical Assessment System (Professional UI)"""

# TODO: Implement Streamlit UI application with professional styling and workflow integration
# PURPOSE: Create comprehensive medical assessment interface with patient case selection, data display, and results visualization
# INCLUDES: Page configuration, CSS styling, cached functions (load_model_evaluation, get_available_json_files, get_severity_color, format_clinical_field)
# INCLUDES: Two-phase UI (INPUT/RESULTS), sidebar metrics, patient data visualization, 4-tab results display, report download functionality
# RETURNS: Interactive Streamlit application with real-time workflow execution and result display

#test commit
