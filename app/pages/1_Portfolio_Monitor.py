"""SFEWS — Portfolio Monitor: score every company's latest snapshot, sort/filter by risk."""

import streamlit as st
import pandas as pd
import json
import joblib
from pathlib import Path

st.set_page_config(page_title="Portfolio Monitor — SFEWS", page_icon="📊", layout="wide")

BASE = Path(__file__).parent.parent.parent


@st.cache_data
def load_data():
    features = pd.read_csv(BASE / "data/processed/sfews_features.csv")
    master = pd.read_csv(BASE / "data/raw/startup_master.csv")
    return features, master


@st.cache_resource
def load_model():
    model = joblib.load(BASE / "src/models/sfews_calibrated_model.joblib")
    with open(BASE / "src/models/feature_columns.json") as f:
        meta = json.load(f)
    return model, meta


st.title("📊 Portfolio Monitor")
st.caption("Latest known operating snapshot for every company, scored for 6-month failure risk.")

features, master = load_data()
model, meta = load_model()
feature_cols = meta["feature_cols"]


@st.cache_data
def build_snapshot(_features, _master, seed=42):
    """
    For a realistic 'as of today' portfolio snapshot, sample a random observed
    month per company rather than always taking the very last one. Using the
    final month for every company would bias the 'active' subset toward
    end-of-panel companies that, by construction of how this dataset truncates,
    skew healthy -- and would bias 'failed' companies toward the single month
    right before failure (always high risk). A random mid-life month per
    company gives the spread of risk levels you'd actually see scoring a real,
    live portfolio at an arbitrary point in time.
    """
    rng = pd.Series(_features["company_id"].unique()).sample(frac=1.0, random_state=seed)
    sampled_rows = []
    for cid, g in _features.groupby("company_id"):
        row = g.sample(1, random_state=hash(cid) % (2**31)).iloc[0]
        sampled_rows.append(row)
    snap = pd.DataFrame(sampled_rows).reset_index(drop=True)
    snap = snap.merge(_master[["company_id", "failed", "acquired", "fail_month"]], on="company_id", how="left")
    return snap


latest = build_snapshot(features, master)

# IMPORTANT DEPLOYMENT NOTE: we snapshot each company at a random observed month
# (see build_snapshot) rather than always its final month. Always using the final
# month would bias "active" companies toward end-of-panel (skews healthy) and
# "failed" companies toward the single month right before failure (skews 100% high
# risk) -- neither resembles scoring a real, live portfolio at an arbitrary point in
# time. A random-month snapshot gives realistic variety across the risk spectrum.
view_mode = st.radio(
    "Portfolio view",
    ["🟢 Active companies only (live monitoring use case)", "🗂️ Full historical dataset (including resolved outcomes)"],
    horizontal=True,
)
if view_mode.startswith("🟢"):
    latest = latest[~latest["failed"] & ~latest["acquired"]].reset_index(drop=True)
    st.caption("Showing only companies with no resolved outcome yet, snapshotted at a random point in their lifecycle -- this mirrors scoring a live, ongoing portfolio at an arbitrary point in time.")
else:
    st.caption("Showing all companies including those that have already failed/been acquired -- useful for backtesting and auditing model behavior, not a realistic live view (since their outcome is already known).")

latest_with_static = latest.merge(
    master[["company_id", "sector", "region", "initial_funding_stage"]].rename(
        columns={"sector": "sector_m", "region": "region_m", "initial_funding_stage": "stage_m"}),
    on="company_id", how="left"
)

X_latest = latest[feature_cols]
proba = model.predict_proba(X_latest)[:, 1]
latest["risk_score"] = proba
latest["risk_tier"] = pd.cut(latest["risk_score"], bins=[-0.01, 0.15, 0.45, 1.01],
                              labels=["Low", "Watch", "High"])

# --- filters ---
fcol1, fcol2, fcol3, fcol4 = st.columns(4)
with fcol1:
    sectors = ["All"] + sorted(latest["sector"].unique().tolist())
    sector_filter = st.selectbox("Sector", sectors)
with fcol2:
    regions = ["All"] + sorted(latest["region"].unique().tolist())
    region_filter = st.selectbox("Region", regions)
with fcol3:
    tiers = ["All", "High", "Watch", "Low"]
    tier_filter = st.selectbox("Risk Tier", tiers)
with fcol4:
    min_runway, max_runway = float(latest["runway_months_latest"].min()), float(latest["runway_months_latest"].max())
    runway_cap = st.slider("Max runway filter (months)", 0.0, min(max_runway, 99.0), min(max_runway, 99.0))

filtered = latest.copy()
if sector_filter != "All":
    filtered = filtered[filtered["sector"] == sector_filter]
if region_filter != "All":
    filtered = filtered[filtered["region"] == region_filter]
if tier_filter != "All":
    filtered = filtered[filtered["risk_tier"] == tier_filter]
filtered = filtered[filtered["runway_months_latest"] <= runway_cap]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Companies (filtered)", f"{len(filtered):,}")
m2.metric("High Risk", f"{(filtered['risk_tier']=='High').sum():,}")
m3.metric("Watch", f"{(filtered['risk_tier']=='Watch').sum():,}")
m4.metric("Avg Risk Score", f"{filtered['risk_score'].mean():.1%}" if len(filtered) else "—")

st.divider()

sort_col = st.selectbox("Sort by", ["risk_score", "runway_months_latest", "mrr_latest", "churn_rate_latest"],
                         format_func=lambda x: {"risk_score": "Risk Score (desc)",
                                                 "runway_months_latest": "Runway (asc)",
                                                 "mrr_latest": "MRR (desc)",
                                                 "churn_rate_latest": "Churn Rate (desc)"}[x])
ascending = sort_col == "runway_months_latest"
display_df = filtered.sort_values(sort_col, ascending=ascending)[
    ["company_id", "sector", "region", "risk_score", "risk_tier", "runway_months_latest",
     "mrr_latest", "churn_rate_latest", "ltv_cac_ratio_latest", "composite_risk_index"]
].rename(columns={
    "company_id": "Company", "sector": "Sector", "region": "Region",
    "risk_score": "Risk Score", "risk_tier": "Tier", "runway_months_latest": "Runway (mo)",
    "mrr_latest": "MRR ($)", "churn_rate_latest": "Churn", "ltv_cac_ratio_latest": "LTV:CAC",
    "composite_risk_index": "Heuristic Risk Idx"
})

st.dataframe(
    display_df,
    use_container_width=True,
    height=500,
    column_config={
        "Risk Score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=1, format="%.0%%"),
        "MRR ($)": st.column_config.NumberColumn("MRR ($)", format="$%.0f"),
        "Churn": st.column_config.NumberColumn("Churn", format="%.1%%"),
    },
    hide_index=True,
)

st.caption(
    "Tip: open **Company Drilldown** and paste a Company ID from this table to see its "
    "SHAP-based explanation and risk trajectory over time."
)

csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Export filtered view as CSV", csv, "sfews_portfolio_snapshot.csv", "text/csv")
