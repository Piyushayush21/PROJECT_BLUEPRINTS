"""SFEWS — What-If Simulator: adjust a company's metrics live, see risk score respond."""

import streamlit as st
import pandas as pd
import json
import joblib
from pathlib import Path

st.set_page_config(page_title="What-If Simulator — SFEWS", page_icon="🧪", layout="wide")

BASE = Path(__file__).parent.parent.parent


@st.cache_data
def load_data():
    return pd.read_csv(BASE / "data/processed/sfews_features.csv")


@st.cache_resource
def load_model():
    model = joblib.load(BASE / "src/models/sfews_calibrated_model.joblib")
    with open(BASE / "src/models/feature_columns.json") as f:
        meta = json.load(f)
    return model, meta


st.title("🧪 What-If Simulator")
st.caption(
    "Pick a starting company, then move the sliders to see how the risk score "
    "responds to operating changes — useful for founders asking 'what would it "
    "take to get out of the danger zone?' or investors stress-testing a thesis."
)

features = load_data()
model, meta = load_model()
feature_cols = meta["feature_cols"]

all_ids = sorted(features["company_id"].unique().tolist())
company_id = st.selectbox("Start from company", all_ids)
company_rows = features[features["company_id"] == company_id].sort_values("month_index")
base_row = company_rows.tail(1).copy().reset_index(drop=True)

baseline_score = float(model.predict_proba(base_row[feature_cols])[:, 1][0])

st.divider()
st.markdown("### Adjust Key Levers")

sim_row = base_row.copy()

slider_specs = [
    ("runway_months_latest", "Cash runway (months)", 0.0, 36.0, 0.5),
    ("churn_rate_latest", "Monthly churn rate", 0.0, 0.6, 0.01),
    ("mrr_slope_6m", "6-month MRR trend (slope)", -50000.0, 50000.0, 500.0),
    ("ltv_cac_ratio_latest", "LTV:CAC ratio", 0.0, 8.0, 0.1),
    ("attrition_rate_latest", "Employee attrition rate", 0.0, 0.5, 0.01),
    ("consecutive_low_runway_months", "Consecutive months runway < 6mo", 0, 20, 1),
    ("glassdoor_rating_latest", "Glassdoor rating", 1.0, 5.0, 0.1),
]

cols = st.columns(2)
for i, (col_name, label, lo, hi, step) in enumerate(slider_specs):
    current_val = float(base_row[col_name].iloc[0])
    current_val = min(max(current_val, lo), hi)
    with cols[i % 2]:
        new_val = st.slider(label, lo, hi, current_val, step=step, key=f"slider_{col_name}")
        sim_row[col_name] = new_val

new_score = float(model.predict_proba(sim_row[feature_cols])[:, 1][0])
delta = new_score - baseline_score

st.divider()
r1, r2, r3 = st.columns(3)
r1.metric("Baseline Risk Score", f"{baseline_score:.1%}")
r2.metric("Simulated Risk Score", f"{new_score:.1%}", delta=f"{delta:+.1%}", delta_color="inverse")
with r3:
    if new_score < 0.15:
        st.success("✅ Simulated scenario: Low Risk")
    elif new_score < 0.45:
        st.warning("⚠️ Simulated scenario: Watch Zone")
    else:
        st.error("🔴 Simulated scenario: High Risk")

st.caption(
    "Note: sliders move one feature at a time without re-deriving correlated features "
    "(e.g. raising runway doesn't automatically reduce burn_multiple). This is a "
    "**ceteris-paribus sensitivity tool**, not a causal simulator — it shows what the "
    "model has learned to associate with risk, not a guaranteed real-world causal effect. "
    "This distinction is intentional and discussed in the model card."
)
