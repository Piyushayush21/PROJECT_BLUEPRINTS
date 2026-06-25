"""
SFEWS — Startup Failure Early Warning System
Streamlit Application Entry Point

Run locally:
    streamlit run app/Home.py

Deploy: push repo to GitHub, connect on share.streamlit.io, set main file
to app/Home.py. See README.md "Deployment" section for full steps.
"""

import streamlit as st
import pandas as pd
import json
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SFEWS — Startup Failure Early Warning System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens — intentional palette, not Streamlit defaults.
# Theme: "mission control for company health" — dark slate base, amber/red
# alert colors reserved ONLY for risk signal (so they stay meaningful),
# a calm teal for "healthy", and a single signature accent (signal-violet)
# used for the brand mark and primary actions only.
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#0E1117",
    "surface": "#161B22",
    "surface_alt": "#1C2330",
    "border": "#2A323D",
    "text": "#E6E8EB",
    "text_dim": "#8B94A3",
    "healthy": "#2DD4BF",
    "watch": "#F5B544",
    "risk": "#F2545B",
    "accent": "#7C6CF6",
}

CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: {PALETTE['bg']}; }}
    section[data-testid="stSidebar"] {{ background-color: {PALETTE['surface']}; border-right: 1px solid {PALETTE['border']}; }}
    h1, h2, h3 {{ font-family: 'IBM Plex Mono', 'SF Mono', monospace; letter-spacing: -0.01em; }}
    .sfews-eyebrow {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em;
        text-transform: uppercase; color: {PALETTE['accent']}; margin-bottom: 0.2rem;
    }}
    .sfews-card {{
        background-color: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
        border-radius: 6px; padding: 1.1rem 1.3rem; margin-bottom: 0.8rem;
    }}
    .sfews-metric-label {{ color: {PALETTE['text_dim']}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }}
    .risk-badge {{
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 600;
    }}
    .risk-low {{ background-color: rgba(45, 212, 191, 0.15); color: {PALETTE['healthy']}; }}
    .risk-mid {{ background-color: rgba(245, 181, 68, 0.15); color: {PALETTE['watch']}; }}
    .risk-high {{ background-color: rgba(242, 84, 91, 0.15); color: {PALETTE['risk']}; }}
    div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    base = Path(__file__).parent.parent
    panel = pd.read_csv(base / "data/raw/startup_monthly_panel.csv")
    master = pd.read_csv(base / "data/raw/startup_master.csv")
    features = pd.read_csv(base / "data/processed/sfews_features.csv")
    return panel, master, features


@st.cache_resource
def load_model():
    base = Path(__file__).parent.parent
    model = joblib.load(base / "src/models/sfews_calibrated_model.joblib")
    raw_model = joblib.load(base / "src/models/sfews_raw_best_model.joblib")
    with open(base / "src/models/feature_columns.json") as f:
        meta = json.load(f)
    return model, raw_model, meta


def risk_badge_html(score: float) -> str:
    if score < 0.15:
        return f'<span class="risk-badge risk-low">LOW RISK · {score:.0%}</span>'
    elif score < 0.45:
        return f'<span class="risk-badge risk-mid">WATCH · {score:.0%}</span>'
    else:
        return f'<span class="risk-badge risk-high">HIGH RISK · {score:.0%}</span>'


# ---------------------------------------------------------------------------
# Landing page content
# ---------------------------------------------------------------------------
st.markdown('<div class="sfews-eyebrow">SFEWS · v1.0</div>', unsafe_allow_html=True)
st.title("🛰️ Startup Failure Early Warning System")
st.markdown(
    "A 6-month-ahead failure risk model trained on a 4,200-company, "
    "**140K-row monthly operating panel** — with calibrated probabilities, "
    "SHAP-based explanations, and a portfolio-level monitoring view."
)

panel, master, features = load_data()
model, raw_model, meta = load_model()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Companies Tracked", f"{master['company_id'].nunique():,}")
with col2:
    st.metric("Historical Failure Rate", f"{master['failed'].mean():.1%}")
with col3:
    with open(Path(__file__).parent.parent / "reports/test_metrics.json") as f:
        test_metrics = json.load(f)
    st.metric("Held-out Test ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
with col4:
    st.metric("Held-out Test PR-AUC", f"{test_metrics['pr_auc']:.3f}")

st.divider()
st.markdown("### 👉 Try it yourself")
st.caption("Type in a company's current metrics and get an instant risk score with a plain-English explanation — no historical data required.")
if st.button("🧮 Open Quick Risk Assessment", type="primary", use_container_width=True):
    st.switch_page("pages/5_Quick_Risk_Assessment.py")

st.divider()

st.markdown("### Navigate")
nav_cols = st.columns(5)
pages = [
    ("🧮 Quick Risk Assessment", "pages/5_Quick_Risk_Assessment.py", "Type in a company's metrics and get an instant risk score + explanation."),
    ("📊 Portfolio Monitor", "pages/1_Portfolio_Monitor.py", "Score every company, sort by risk, filter by sector/region."),
    ("🔍 Company Drilldown", "pages/2_Company_Drilldown.py", "Single-company risk trajectory + SHAP waterfall explanation."),
    ("🧪 What-If Simulator", "pages/3_What_If_Simulator.py", "Adjust a company's metrics live and see risk score respond."),
    ("📈 Model Performance", "pages/4_Model_Performance.py", "Validation methodology, metrics, calibration, fairness checks."),
]
for c, (label, path, desc) in zip(nav_cols, pages):
    with c:
        st.markdown(f"**{label}**")
        st.caption(desc)

st.divider()
with st.expander("ℹ️ About this dataset — please read before drawing conclusions"):
    st.markdown("""
    **This dataset is synthetic, not scraped from Crunchbase/PitchBook.**

    Public startup datasets are small, static (one row per company, no monthly
    operating history), and survivorship-biased toward companies that raised
    *any* funding round (so they already passed one filter). None of them support
    a genuine *early warning* framing because they have no time-series signal.

    This project instead simulates an 18–42 month monthly operating panel per
    company using a structural model calibrated against publicly reported
    industry statistics (CB Insights post-mortems, Startup Genome Report,
    First Round Capital benchmark data) for failure rates by sector, typical
    burn multiples, churn distributions, and founder-team risk factors.

    **Why this is the right call, not a shortcut:** it's disclosed openly, it
    enables the actual longitudinal ML problem this system is meant to solve,
    and it avoids quietly overfitting to one overused public Kaggle CSV that
    every other portfolio project on the internet already uses.

    See `src/data/generate_dataset.py` for the full generative model and the
    `reports/model_card.md` for calibration sources and limitations.
    """)

st.caption("Built with scikit-learn, XGBoost, SHAP, and Streamlit · See GitHub repo for full source.")
