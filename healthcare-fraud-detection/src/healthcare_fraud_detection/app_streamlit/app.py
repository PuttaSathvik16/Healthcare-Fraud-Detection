"""
Healthcare fraud risk review — run from the project root::

    streamlit run src/healthcare_fraud_detection/app_streamlit/app.py

Requires ``models/fraud_model.pkl`` (or legacy ``model.joblib``). Set ``OPENAI_API_KEY`` for narratives.
"""

from __future__ import annotations

import os
from datetime import date

import streamlit as st

from healthcare_fraud_detection.api.claims import claims_to_dataframe
from healthcare_fraud_detection.api.schemas import ClaimInput
from healthcare_fraud_detection.models.inference import (
    load_predictor_from_disk,
    top_global_feature_importances,
)
from healthcare_fraud_detection.utils.explanations import generate_fraud_explanation

_PAGE_STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>
    :root {
        --hfd-navy: #0f2744;
        --hfd-navy-light: #1a3a5c;
        --hfd-accent: #0d6efd;
        --hfd-accent-soft: #e8f1ff;
        --hfd-surface: #f4f6f9;
        --hfd-card: #ffffff;
        --hfd-border: #dce3ed;
        --hfd-text: #1a2332;
        --hfd-muted: #5c6b7f;
        --hfd-success: #0d9488;
        --hfd-warn: #d97706;
        --hfd-danger: #c2410c;
        --hfd-radius: 10px;
        --hfd-shadow: 0 1px 2px rgba(15, 39, 68, 0.06), 0 4px 12px rgba(15, 39, 68, 0.04);
    }
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent; }
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1080px;
    }
    .hfd-topbar {
        background: linear-gradient(135deg, var(--hfd-navy) 0%, var(--hfd-navy-light) 100%);
        color: #fff;
        padding: 1.35rem 1.75rem;
        border-radius: var(--hfd-radius);
        margin-bottom: 1.75rem;
        box-shadow: var(--hfd-shadow);
    }
    .hfd-topbar h1 {
        font-size: 1.65rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.03em !important;
        color: #fff !important;
        margin: 0 0 0.35rem 0 !important;
        border: none !important;
        padding: 0 !important;
    }
    .hfd-topbar p {
        margin: 0;
        opacity: 0.88;
        font-size: 0.95rem;
        font-weight: 400;
        line-height: 1.45;
    }
    .hfd-panel {
        background: var(--hfd-card);
        border: 1px solid var(--hfd-border);
        border-radius: var(--hfd-radius);
        padding: 1.35rem 1.5rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--hfd-shadow);
    }
    .hfd-panel-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--hfd-muted);
        margin: 0 0 1rem 0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--hfd-border);
        border-radius: var(--hfd-radius);
        background: var(--hfd-surface);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
    }
    .hfd-result {
        background: var(--hfd-card);
        border: 1px solid var(--hfd-border);
        border-radius: var(--hfd-radius);
        padding: 1.5rem 1.75rem;
        margin: 1.5rem 0;
        box-shadow: var(--hfd-shadow);
    }
    .hfd-score-row {
        display: flex;
        align-items: center;
        gap: 2rem;
        flex-wrap: wrap;
    }
    .hfd-score-big {
        font-size: 3.25rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: var(--hfd-navy);
        line-height: 1;
    }
    .hfd-score-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--hfd-muted);
        margin-bottom: 0.35rem;
    }
    .hfd-badge {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .hfd-badge-low { background: #ecfdf5; color: #0f766e; border: 1px solid #99f6e4; }
    .hfd-badge-mid { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
    .hfd-badge-high { background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }
    .hfd-explain {
        background: linear-gradient(180deg, #fafbfd 0%, #ffffff 100%);
        border: 1px solid var(--hfd-border);
        border-left: 4px solid var(--hfd-accent);
        border-radius: var(--hfd-radius);
        padding: 1.35rem 1.5rem;
        margin-top: 0.5rem;
    }
    .hfd-explain-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--hfd-muted);
        margin-bottom: 0.75rem;
    }
    .hfd-footnote {
        font-size: 0.78rem;
        color: var(--hfd-muted);
        margin-top: 2.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid var(--hfd-border);
        line-height: 1.5;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.75rem !important;
        font-weight: 700 !important;
        color: var(--hfd-navy) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: var(--hfd-muted) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--hfd-navy) !important;
        border: none !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        padding: 0.65rem 1.25rem !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(15, 39, 68, 0.12);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--hfd-navy-light) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafbfc 0%, #f0f3f7 100%);
        border-right: 1px solid var(--hfd-border);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    .hfd-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        margin: 0.5rem 0;
        color: var(--hfd-text);
    }
    .hfd-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .hfd-dot-ok { background: #10b981; }
    .hfd-dot-warn { background: #f59e0b; }
</style>
"""


@st.cache_resource
def _load_scoring_assets():
    pred, raw = load_predictor_from_disk()
    highlights = top_global_feature_importances(raw, k=8)
    return pred, highlights


def _build_claim_from_form(
    procedure_code: str,
    claim_amount: float,
    claim_date: date,
    age: int,
    gender: str,
    chronic_condition: str,
    specialization: str,
    region: str,
    avg_claim_amount_per_provider: float,
    total_claims_per_provider: int,
    total_claims_per_patient: int,
    claim_amount_deviation_from_provider_avg: float,
    claim_amount_zscore: float,
    claims_last_7_days: int,
    claims_last_30_days: int,
    is_duplicate_claim: int,
) -> ClaimInput:
    return ClaimInput(
        procedure_code=procedure_code.strip(),
        claim_amount=float(claim_amount),
        claim_date=claim_date.isoformat(),
        age=int(age),
        gender=gender.strip(),
        chronic_condition=chronic_condition.strip() or "none",
        specialization=specialization.strip(),
        region=region.strip(),
        avg_claim_amount_per_provider=float(avg_claim_amount_per_provider),
        total_claims_per_provider=int(total_claims_per_provider),
        total_claims_per_patient=int(total_claims_per_patient),
        claim_amount_deviation_from_provider_avg=float(claim_amount_deviation_from_provider_avg),
        claim_amount_zscore=float(claim_amount_zscore),
        claims_last_7_days=int(claims_last_7_days),
        claims_last_30_days=int(claims_last_30_days),
        is_duplicate_claim=int(is_duplicate_claim),
    )


def _tier_badge(score: float) -> tuple[str, str]:
    if score >= 0.7:
        return "Elevated priority", "hfd-badge hfd-badge-high"
    if score >= 0.4:
        return "Moderate review", "hfd-badge hfd-badge-mid"
    return "Within normal range", "hfd-badge hfd-badge-low"


def main() -> None:
    st.set_page_config(
        page_title="Fraud Risk Console | Healthcare",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_PAGE_STYLE, unsafe_allow_html=True)

    api_key_set = bool(os.environ.get("OPENAI_API_KEY"))

    try:
        predictor, highlights = _load_scoring_assets()
        model_ok = True
    except FileNotFoundError:
        predictor, highlights = None, None
        model_ok = False

    with st.sidebar:
        st.markdown("### Console")
        st.caption("Operational fraud risk assessment")
        st.divider()
        if model_ok:
            st.markdown(
                '<div class="hfd-status"><span class="hfd-dot hfd-dot-ok"></span>Scoring engine ready</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="hfd-status"><span class="hfd-dot hfd-dot-warn"></span>Model not loaded</div>',
                unsafe_allow_html=True,
            )
        if api_key_set:
            st.markdown(
                '<div class="hfd-status"><span class="hfd-dot hfd-dot-ok"></span>Narrative (OpenAI) on</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="hfd-status"><span class="hfd-dot hfd-dot-warn"></span>Narrative disabled</div>',
                unsafe_allow_html=True,
            )
        st.divider()
        with st.expander("Operational notes"):
            st.caption(
                "Scores are model estimates for workflow triage—not legal findings. "
                "Align thresholds with your SIU policy."
            )
        st.divider()
        st.caption("Healthcare Fraud Detection")

    st.markdown(
        """
        <div class="hfd-topbar">
            <h1>Fraud risk console</h1>
            <p>Evaluate a single claim against trained risk signals. Use results to prioritize specialist review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_ok:
        st.error(
            "**Scoring engine unavailable.** Train a model from the project root, then refresh:\n\n"
            "`python -m healthcare_fraud_detection.models.train`"
        )
        st.stop()

    with st.form("claim_form", clear_on_submit=False):
        st.markdown('<p class="hfd-panel-title">Claim & beneficiary</p>', unsafe_allow_html=True)
        st.markdown('<div class="hfd-panel">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            procedure_code = st.text_input("Procedure code", placeholder="e.g. 99213", help="CPT or equivalent")
            claim_amount = st.number_input("Claim amount (USD)", min_value=0.0, value=150.0, step=10.0, format="%.2f")
        with c2:
            claim_date = st.date_input("Service / claim date", value=date.today())
            age = st.number_input("Patient age", min_value=0, max_value=120, value=45)
        with c3:
            gender = st.selectbox("Gender", ["M", "F", "NB", "U"], index=0)
            chronic_condition = st.text_input("Chronic condition", value="none", help="e.g. diabetes, hypertension, none")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<p class="hfd-panel-title">Provider & market</p>', unsafe_allow_html=True)
        st.markdown('<div class="hfd-panel">', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            specialization = st.text_input("Clinical specialization", value="internal_medicine")
        with p2:
            region = st.selectbox(
                "Geographic region",
                ["NORTHEAST", "SOUTH", "MIDWEST", "WEST", "PACIFIC"],
                index=2,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Advanced utilization & billing signals", expanded=False):
            st.caption(
                "Populate when your data warehouse exposes aggregates. Leave at zero for a claim-line–only review."
            )
            a1, a2, a3 = st.columns(3)
            with a1:
                avg_claim_amount_per_provider = st.number_input("Avg paid (provider)", value=0.0, format="%.2f")
                total_claims_per_provider = st.number_input("Claim volume (provider)", value=0, step=1)
            with a2:
                total_claims_per_patient = st.number_input("Claim volume (patient)", value=0, step=1)
                claim_amount_deviation_from_provider_avg = st.number_input("Δ vs provider average ($)", value=0.0, format="%.2f")
            with a3:
                claim_amount_zscore = st.number_input("Amount z-score (provider)", value=0.0, format="%.2f")
                claims_last_7_days = st.number_input("Patient claims (7 days)", value=0, step=1)
                claims_last_30_days = st.number_input("Patient claims (30 days)", value=0, step=1)
                is_duplicate_claim = st.selectbox("Potential duplicate line", [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")

        st.markdown("<br/>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run assessment", type="primary", use_container_width=True)

    if not submitted:
        st.markdown(
            '<p class="hfd-footnote">Outputs support triage only. Document decisions per your compliance program.</p>',
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        claim = _build_claim_from_form(
            procedure_code=procedure_code,
            claim_amount=claim_amount,
            claim_date=claim_date,
            age=age,
            gender=gender,
            chronic_condition=chronic_condition,
            specialization=specialization,
            region=region,
            avg_claim_amount_per_provider=avg_claim_amount_per_provider,
            total_claims_per_provider=total_claims_per_provider,
            total_claims_per_patient=total_claims_per_patient,
            claim_amount_deviation_from_provider_avg=claim_amount_deviation_from_provider_avg,
            claim_amount_zscore=claim_amount_zscore,
            claims_last_7_days=claims_last_7_days,
            claims_last_30_days=claims_last_30_days,
            is_duplicate_claim=is_duplicate_claim,
        )
    except Exception as exc:
        st.error(f"Input validation failed: {exc}")
        st.stop()

    assert predictor is not None
    df = claims_to_dataframe([claim])
    with st.spinner("Computing risk estimate…"):
        score = float(predictor.predict_proba(df)[0])

    pct = round(score * 100, 1)
    tier_text, tier_class = _tier_badge(score)

    st.markdown('<div class="hfd-result">', unsafe_allow_html=True)
    st.markdown('<p class="hfd-score-label">Model risk score</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown(f'<div class="hfd-score-big">{pct}%</div>', unsafe_allow_html=True)
        st.progress(min(max(score, 0.0), 1.0))
    with col_b:
        st.markdown(f'<p style="margin:0.5rem 0 0.75rem 0;"><span class="{tier_class}">{tier_text}</span></p>', unsafe_allow_html=True)
        if score >= 0.7:
            st.markdown(
                "<p style='color:#5c6b7f; margin:0; line-height:1.55;'>Several indicators align with "
                "historically escalated claims. Recommend structured review against billing and authorization policy.</p>",
                unsafe_allow_html=True,
            )
        elif score >= 0.4:
            st.markdown(
                "<p style='color:#5c6b7f; margin:0; line-height:1.55;'>Mixed signals versus peer patterns. "
                "A targeted desk review or documentation check may be sufficient.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p style='color:#5c6b7f; margin:0; line-height:1.55;'>Relative to similar claims, this line "
                "presents limited anomaly—routine monitoring may apply.</p>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<p class="hfd-panel-title" style="margin-top:1.5rem;">Narrative summary</p>', unsafe_allow_html=True)
    if not api_key_set:
        st.markdown(
            '<div class="hfd-explain"><div class="hfd-explain-title">Configuration</div>'
            '<p style="margin:0; color:#5c6b7f;">Set environment variable <code>OPENAI_API_KEY</code> and restart '
            "the app to generate executive-ready language from the same inputs.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Generating narrative…"):
            try:
                narrative = generate_fraud_explanation(
                    score,
                    claim.model_dump(),
                    highlights,
                )
            except Exception as exc:
                st.error(f"Narrative service error: {exc}")
                st.stop()
        st.markdown(
            f'<div class="hfd-explain"><div class="hfd-explain-title">Summary for business readers</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(narrative)

    st.markdown(
        '<p class="hfd-footnote">Model output is not a determination of fraud or abuse. '
        "Use alongside clinical, contractual, and regulatory context.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
