"""
SFEWS — Quick Risk Assessment: manual single-entry prediction form.

Uses a SEPARATE, simpler model (sfews_snapshot_model.joblib) trained only on
point-in-time features -- no rolling 3/6-month trend or volatility features,
since those require historical data a person can't type into a form in one
sitting. See src/models/train_snapshot_model.py docstring for the full
rationale. This is disclosed in the UI below, not hidden.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))

st.set_page_config(page_title="Quick Risk Assessment — SFEWS", page_icon="🧮", layout="wide")

PALETTE = {"healthy": "#2DD4BF", "watch": "#F5B544", "risk": "#F2545B", "accent": "#7C6CF6"}


@st.cache_resource
def load_snapshot_model():
    calibrated = joblib.load(BASE / "src/models/sfews_snapshot_model.joblib")
    raw = joblib.load(BASE / "src/models/sfews_snapshot_raw_model.joblib")
    with open(BASE / "src/models/snapshot_feature_columns.json") as f:
        meta = json.load(f)
    return calibrated, raw, meta


@st.cache_data
def load_background():
    df = pd.read_csv(BASE / "data/processed/sfews_features.csv")
    return df


@st.cache_resource
def get_explainer(_raw_pipe, _background_df, feature_cols):
    import shap
    pre = _raw_pipe.named_steps["pre"]
    clf = _raw_pipe.named_steps["clf"]
    sample = _background_df[feature_cols].sample(min(300, len(_background_df)), random_state=42)
    X_t = pre.transform(sample)
    explainer = shap.TreeExplainer(clf)
    num_cols = pre.transformers_[0][2]
    cat_encoder = pre.named_transformers_["cat"]
    cat_cols_out = list(cat_encoder.get_feature_names_out(pre.transformers_[1][2]))
    feature_names_out = list(num_cols) + cat_cols_out
    return explainer, feature_names_out


FEATURE_LABELS = {
    "runway_months_latest": "cash runway",
    "churn_rate_latest": "customer churn rate",
    "ltv_cac_ratio_latest": "LTV:CAC ratio",
    "attrition_rate_latest": "employee attrition rate",
    "founder_conflict_ever": "history of founder conflict",
    "glassdoor_rating_latest": "Glassdoor rating",
    "burn_multiple": "burn multiple",
    "growth_rate_mom_latest": "month-over-month growth rate",
    "nps_score_latest": "Net Promoter Score",
    "capital_efficiency": "capital efficiency",
    "composite_risk_index": "heuristic risk index",
    "cac_payback_months_latest": "CAC payback period",
    "mrr_latest": "monthly recurring revenue",
    "cash_on_hand_latest": "cash on hand",
    "burn_rate_latest": "monthly burn rate",
    "headcount_latest": "headcount",
    "founder_technical_pct": "founder technical background",
    "founder_prior_exits": "founders' prior exits",
}


def humanize(feat_name: str) -> str:
    if feat_name in FEATURE_LABELS:
        return FEATURE_LABELS[feat_name]
    return feat_name.replace("_latest", "").replace("_", " ")


st.markdown("### 🧮 Quick Risk Assessment")
st.title("Score a Company Right Now")
st.caption(
    "Fill in what you know about a company's current state and get an instant "
    "6-month failure risk score with a plain-English explanation."
)

with st.expander("ℹ️ About this form's model — please read"):
    st.markdown("""
    This form uses a **separate, simpler model** from the rest of the app.
    The main SFEWS model (used in Portfolio Monitor / Company Drilldown) relies on
    **56 features**, many of which are rolling 3-6 month trends and volatility
    measures (e.g. "6-month MRR growth slope") — those need historical data a
    person can't type into a form in one sitting.

    This form instead uses a model trained on **27 point-in-time-only features**
    — things you can reasonably know right now about a company. It's honestly a
    **slightly weaker model** as a result (test PR-AUC ~0.67 vs. ~0.70 for the
    full temporal model — see `reports/snapshot_model_test_metrics.json`), and
    that tradeoff is the price of being usable without a month-by-month history.
    """)

calibrated_model, raw_model, meta = load_snapshot_model()
feature_cols, num_cols, cat_cols = meta["feature_cols"], meta["num_cols"], meta["cat_cols"]
background_df = load_background()

st.divider()

# ---------------------------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------------------------
with st.form("risk_assessment_form"):
    st.markdown("#### Company Basics")
    b1, b2, b3 = st.columns(3)
    with b1:
        sector = st.selectbox("Sector", ["SaaS", "Fintech", "HealthTech", "E-commerce", "AI/ML",
                                          "EdTech", "Marketplace", "DeepTech", "ClimateTech", "ConsumerApps"])
        founded_year = st.number_input("Founded year", min_value=2010, max_value=2026, value=2022, step=1)
    with b2:
        region = st.selectbox("Region", ["US-WestCoast", "US-EastCoast", "US-Other", "Europe",
                                          "India", "SEA", "LatAm"])
        initial_funding_stage = st.selectbox("Funding stage", ["Pre-Seed", "Seed", "Series A", "Series B", "Series C+"])
    with b3:
        initial_capital = st.number_input("Total capital raised to date ($)", min_value=10_000,
                                            max_value=500_000_000, value=1_500_000, step=50_000)
        company_age_months = st.number_input("Company age (months since founding)", min_value=3,
                                               max_value=120, value=18, step=1)

    st.markdown("#### Founding Team")
    f1, f2, f3 = st.columns(3)
    with f1:
        founder_count = st.slider("Number of founders", 1, 5, 2)
    with f2:
        founder_prior_exits = st.slider("Founders' combined prior exits", 0, 5, 0)
    with f3:
        founder_technical_pct = st.slider("Fraction of founders who are technical", 0.0, 1.0, 0.5, step=0.05)

    st.markdown("#### Financial Health")
    fin1, fin2, fin3 = st.columns(3)
    with fin1:
        mrr = st.number_input("Monthly Recurring Revenue ($)", min_value=0, max_value=10_000_000, value=40_000, step=1000)
        cash_on_hand = st.number_input("Cash on hand ($)", min_value=0, max_value=200_000_000, value=600_000, step=10_000)
    with fin2:
        burn_rate = st.number_input("Monthly burn rate ($)", min_value=0, max_value=5_000_000, value=80_000, step=1000)
        runway_months = st.slider("Cash runway remaining (months)", 0.0, 48.0, 7.5, step=0.5)
    with fin3:
        growth_rate_mom = st.slider("Month-over-month revenue growth rate", -0.30, 0.50, 0.04, step=0.01,
                                      help="0.05 = 5% growth this month vs last month")
        cac_payback_months = st.slider("CAC payback period (months)", 0.0, 36.0, 12.0, step=0.5)

    st.markdown("#### Customer & Team Health")
    c1, c2, c3 = st.columns(3)
    with c1:
        churn_rate = st.slider("Monthly customer churn rate", 0.0, 0.6, 0.08, step=0.01)
        ltv_cac_ratio = st.slider("LTV:CAC ratio", 0.0, 8.0, 2.0, step=0.1)
    with c2:
        nps_score = st.slider("Net Promoter Score", -100, 100, 25)
        headcount = st.number_input("Current headcount", min_value=1, max_value=2000, value=15, step=1)
    with c3:
        attrition_rate = st.slider("Monthly employee attrition rate", 0.0, 0.5, 0.05, step=0.01)
        glassdoor_rating = st.slider("Glassdoor rating", 1.0, 5.0, 3.5, step=0.1)

    st.markdown("#### Product & Reputation")
    p1, p2, p3 = st.columns(3)
    with p1:
        product_release_count = st.number_input("Product releases in the last month", min_value=0, max_value=10, value=1)
    with p2:
        press_mentions = st.number_input("Press mentions in the last month", min_value=0, max_value=50, value=1)
    with p3:
        founder_conflict_ever = st.checkbox("Has there been visible founder conflict?", value=False)

    submitted = st.form_submit_button("🔍 Calculate Risk Score", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
if submitted:
    burn_multiple = burn_rate / mrr if mrr > 0 else 99.0
    capital_efficiency = (mrr * 12) / (burn_rate * max(company_age_months, 1)) if burn_rate > 0 else 0.0

    risk = 0.0
    risk += 25 * (runway_months < 6)
    risk += 15 * (growth_rate_mom < 0)
    risk += 15 * (churn_rate > 0.20)
    risk += 10 * (ltv_cac_ratio < 1.0)
    risk += 10 * (attrition_rate > 0.15)
    risk += 10 * (glassdoor_rating < 2.5)
    risk += 5 * int(founder_conflict_ever)
    composite_risk_index = min(risk, 100.0)

    input_row = pd.DataFrame([{
        "mrr_latest": mrr, "cash_on_hand_latest": cash_on_hand, "burn_rate_latest": burn_rate,
        "runway_months_latest": runway_months, "churn_rate_latest": churn_rate,
        "cac_payback_months_latest": cac_payback_months, "nps_score_latest": nps_score,
        "headcount_latest": headcount, "attrition_rate_latest": attrition_rate,
        "product_release_count_latest": product_release_count, "press_mentions_latest": press_mentions,
        "glassdoor_rating_latest": glassdoor_rating, "ltv_cac_ratio_latest": ltv_cac_ratio,
        "growth_rate_mom_latest": growth_rate_mom, "founder_conflict_ever": int(founder_conflict_ever),
        "burn_multiple": burn_multiple, "capital_efficiency": capital_efficiency,
        "composite_risk_index": composite_risk_index, "founded_year": founded_year,
        "founder_count": founder_count, "founder_prior_exits": founder_prior_exits,
        "founder_technical_pct": founder_technical_pct, "initial_capital": initial_capital,
        "company_age_months_at_t": company_age_months, "sector": sector, "region": region,
        "initial_funding_stage": initial_funding_stage,
    }])[feature_cols]

    risk_score = float(calibrated_model.predict_proba(input_row)[:, 1][0])

    st.divider()
    st.markdown("## Result")

    rcol1, rcol2 = st.columns([1, 2])
    with rcol1:
        if risk_score < 0.15:
            st.success(f"### ✅ LOW RISK\n## {risk_score:.0%}")
            tier_text = "This company's current profile looks healthy relative to the training population."
        elif risk_score < 0.45:
            st.warning(f"### ⚠️ WATCH ZONE\n## {risk_score:.0%}")
            tier_text = "Some warning signs are present. Worth monitoring closely over the next few months."
        else:
            st.error(f"### 🔴 HIGH RISK\n## {risk_score:.0%}")
            tier_text = "Multiple significant risk factors detected. This profile resembles companies that failed within 6 months in the training data."
        st.caption(tier_text)
        st.metric("Estimated 6-month failure probability", f"{risk_score:.1%}")

    with rcol2:
        with st.spinner("Computing explanation..."):
            explainer, feature_names_out = get_explainer(raw_model, background_df, feature_cols)
            pre = raw_model.named_steps["pre"]
            X_t = pre.transform(input_row)
            shap_vals = explainer.shap_values(X_t)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            shap_vals = np.asarray(shap_vals)
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
            shap_vals = shap_vals[0]

        contrib = pd.DataFrame({"feature": feature_names_out, "shap_value": shap_vals})
        contrib["abs_shap"] = contrib["shap_value"].abs()
        contrib = contrib.sort_values("abs_shap", ascending=False)
        top_risk = contrib[contrib["shap_value"] > 0].head(3)
        top_protective = contrib[contrib["shap_value"] < 0].head(2)

        risk_phrases = [humanize(f) for f in top_risk["feature"].tolist()]
        protective_phrases = [humanize(f) for f in top_protective["feature"].tolist()]

        narrative_parts = ["**Why this score:** "]
        if risk_phrases:
            narrative_parts.append("The biggest risk drivers are " + ", ".join(risk_phrases[:-1]) +
                                    (" and " if len(risk_phrases) > 1 else "") + risk_phrases[-1] + ".")
        if protective_phrases:
            narrative_parts.append(" This is partially offset by relatively healthy " +
                                    " and ".join(protective_phrases) + ".")
        st.info("".join(narrative_parts))

        if not top_risk.empty:
            chart_df = pd.concat([top_risk, top_protective])
            chart_df["label"] = chart_df["feature"].apply(humanize)
            st.markdown("**Feature contributions to this score**")
            st.bar_chart(chart_df.set_index("label")["shap_value"], height=220)

    st.divider()
    st.caption(
        "This is a predictive association, not a causal claim or investment recommendation. "
        "Trained on synthetic data calibrated to public industry statistics — see the Model "
        "Performance page and `reports/model_card.md` for full methodology and limitations."
    )
