"""SFEWS — Company Drilldown: trajectory over time + SHAP-based local explanation."""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))

from src.explainability.shap_explain import get_shap_explainer, explain_single_company

st.set_page_config(page_title="Company Drilldown — SFEWS", page_icon="🔍", layout="wide")


@st.cache_data
def load_data():
    features = pd.read_csv(BASE / "data/processed/sfews_features.csv")
    panel = pd.read_csv(BASE / "data/raw/startup_monthly_panel.csv")
    master = pd.read_csv(BASE / "data/raw/startup_master.csv")
    return features, panel, master


@st.cache_resource
def load_models():
    calibrated = joblib.load(BASE / "src/models/sfews_calibrated_model.joblib")
    raw = joblib.load(BASE / "src/models/sfews_raw_best_model.joblib")
    with open(BASE / "src/models/feature_columns.json") as f:
        meta = json.load(f)
    return calibrated, raw, meta


st.title("🔍 Company Drilldown")

features, panel, master = load_data()
calibrated, raw_model, meta = load_models()
feature_cols, num_cols, cat_cols = meta["feature_cols"], meta["num_cols"], meta["cat_cols"]

all_ids = sorted(features["company_id"].unique().tolist())
default_idx = all_ids.index("SU-10000") if "SU-10000" in all_ids else 0
company_id = st.selectbox("Select Company ID", all_ids, index=default_idx)

company_features = features[features["company_id"] == company_id].sort_values("month_index")
company_panel = panel[panel["company_id"] == company_id].sort_values("month_index")
company_master = master[master["company_id"] == company_id].iloc[0]

if company_features.empty:
    st.warning("No feature rows for this company (likely too short a history). Try another ID.")
    st.stop()

# ---- header summary ----
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sector", company_master["sector"])
c2.metric("Region", company_master["region"])
c3.metric("Funding Stage", company_master["initial_funding_stage"])
c4.metric("Founded", int(company_master["founded_year"]))
status = "Failed" if company_master["failed"] else ("Acquired" if company_master["acquired"] else "Active/Survived")
c5.metric("Outcome", status)

st.divider()

# ---- risk trajectory over time ----
X_all_months = company_features[feature_cols]
proba_over_time = calibrated.predict_proba(X_all_months)[:, 1]
traj_df = pd.DataFrame({
    "month_index": company_features["month_index"].values,
    "risk_score": proba_over_time,
})

st.markdown("### Risk Score Over Time")
st.line_chart(traj_df.set_index("month_index")["risk_score"], height=280)
if company_master["failed"]:
    st.caption(f"⚠️ This company failed at month {int(company_master['fail_month'])} "
               f"(panel data ends there). Note how the risk score trended in the months prior.")

st.divider()

# ---- pick a point in time for detailed explanation ----
st.markdown("### Explain a Specific Month")
month_options = company_features["month_index"].tolist()
selected_month = st.select_slider("Month index", options=month_options, value=month_options[-1])

row = company_features[company_features["month_index"] == selected_month]
current_score = float(calibrated.predict_proba(row[feature_cols])[:, 1][0])

score_col, narrative_col = st.columns([1, 2])
with score_col:
    st.metric("Risk Score at this month", f"{current_score:.1%}")
    panel_row = company_panel[company_panel["month_index"] == selected_month].iloc[0]
    st.markdown(f"""
    **Snapshot at month {selected_month}:**
    - MRR: ${panel_row['mrr']:,.0f}
    - Runway: {panel_row['runway_months']:.1f} months
    - Churn rate: {panel_row['churn_rate']:.1%}
    - Headcount: {int(panel_row['headcount'])}
    - LTV:CAC: {panel_row['ltv_cac_ratio']:.2f}
    """)

with narrative_col:
    with st.spinner("Computing SHAP explanation..."):
        background = features[feature_cols].sample(min(1500, len(features)), random_state=42)
        explainer, _, _ = get_shap_explainer(raw_model, background)
        explanation = explain_single_company(raw_model, explainer, feature_cols, num_cols, cat_cols, row)

    st.markdown("**Plain-English Explanation**")
    st.info(explanation["narrative"])

    st.markdown("**Top Risk Drivers**")
    risk_df = pd.DataFrame(explanation["top_risk_drivers"])
    if not risk_df.empty:
        risk_df["feature"] = risk_df["feature"].str.replace("_", " ")
        st.bar_chart(risk_df.set_index("feature")["shap_value"], height=200)

    st.markdown("**Top Protective Factors**")
    prot_df = pd.DataFrame(explanation["top_protective_factors"])
    if not prot_df.empty:
        prot_df["feature"] = prot_df["feature"].str.replace("_", " ")
        st.bar_chart(prot_df.set_index("feature")["shap_value"], height=200)

st.caption(
    "SHAP values are computed against the raw (uncalibrated) tree model for "
    "structural fidelity; the displayed risk score uses the calibrated model. "
    "See model card for why these can differ slightly."
)
