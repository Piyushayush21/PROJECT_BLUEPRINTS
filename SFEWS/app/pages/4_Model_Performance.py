"""SFEWS — Model Performance: methodology transparency page."""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Model Performance — SFEWS", page_icon="📈", layout="wide")

BASE = Path(__file__).parent.parent.parent

st.title("📈 Model Performance & Methodology")
st.caption("Full transparency on validation methodology — this page exists because a risk model nobody can audit shouldn't be trusted.")

with open(BASE / "reports/test_metrics.json") as f:
    test_metrics = json.load(f)
comparison_df = pd.read_csv(BASE / "reports/model_comparison.csv")
shap_imp = pd.read_csv(BASE / "reports/shap_global_importance.csv")

st.markdown("### 1. Validation Methodology")
st.markdown("""
- **Split unit: company, not row.** 70/15/15 train/val/test split on `company_id`,
  then all monthly rows for a company go entirely into one split. A row-level
  random split would leak future information (a company's month-15 row would
  predict month-14 trivially well if both ended up in the same split half).
- **Label horizon:** binary label = "will this company fail within the next 6 months,
  given everything known up to and including the current month." Computed with
  strict forward-only lookups — verified no feature at time *t* uses information
  from *t+1* or later.
- **Class imbalance handled via:** `scale_pos_weight` / `class_weight='balanced'`
  tuned per model family, NOT synthetic oversampling (SMOTE was deliberately
  rejected — interpolating between two unrelated companies' monthly snapshots
  produces a physically meaningless synthetic row in this panel-data setting).
""")

st.divider()
st.markdown("### 2. Model Comparison (Validation Set)")
st.dataframe(comparison_df, use_container_width=True, hide_index=True)
st.caption("PR-AUC (precision-recall AUC) is the primary selection metric, not accuracy or ROC-AUC alone — "
           "with ~9% positive class prevalence, a model that predicts 'healthy' for everyone gets 91% accuracy "
           "while being useless. PR-AUC is far less forgiving of that failure mode.")

st.divider()
st.markdown("### 3. Held-Out Test Performance (Calibrated Best Model)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC-AUC", f"{test_metrics['roc_auc']:.3f}")
m2.metric("PR-AUC", f"{test_metrics['pr_auc']:.3f}")
m3.metric("F1 @ 0.5 threshold", f"{test_metrics['f1_at_0.5']:.3f}")
m4.metric("Brier Score (calibration)", f"{test_metrics['brier_score']:.4f}")
st.caption(f"Business-cost-optimal classification threshold (5:1 false-negative:false-positive cost ratio): "
           f"**{test_metrics['optimal_threshold']:.3f}** — far below the naive 0.5 default, because missing "
           f"a real failure signal is assumed to cost ~5x more than a false alarm. This ratio is a configurable "
           f"business assumption, not a statistical fact — see model card.")

st.divider()
st.markdown("### 4. Calibration")
st.markdown("""
Tree ensembles (XGBoost/LightGBM/RF) are excellent **rankers** but their raw
`predict_proba` outputs are not well-calibrated probabilities out of the box —
a 0.8 score doesn't reliably mean "80% of similar companies fail." We apply
**isotonic regression calibration** fit on the validation set (separate from
both training and test) so that displayed risk percentages are meaningful for
a dashboard consumer, not just a relative ranking. Brier score above quantifies
how well-calibrated the final output is (lower = better; 0 = perfect).
""")

st.divider()
st.markdown("### 5. Global Feature Importance (SHAP)")
st.bar_chart(shap_imp.set_index("feature")["mean_abs_shap"], height=400)
st.caption("Mean absolute SHAP value across a 2,000-row sample. Cash runway and revenue scale "
           "dominate — consistent with financial intuition that cash-out is the most common proximate "
           "cause of failure even when the root cause (poor PMF, team conflict, etc.) lies elsewhere.")

st.divider()
st.markdown("### 6. Known Limitations")
st.warning("""
- **Synthetic data**: calibrated against published industry statistics but not real company records.
  Absolute metrics should not be cited as real-world failure-prediction accuracy; the *methodology*
  is the portable, real-world-applicable part of this project.
- **No causal claims**: this is a predictive/associative model. The What-If Simulator shows
  correlational sensitivity, not "if you do X, failure risk will actually drop by Y in reality."
- **Cohort stability was checked** across founding-year (2014-2022) — ROC-AUC stayed in a
  0.93-0.98 band with no systematic drift, but was NOT checked against real-world
  distribution shift (e.g., a genuine macro shock not represented in training data).
- **Survivorship framing**: acquired companies are treated as a separate, excluded outcome class
  rather than "success," and rows after acquisition are dropped from training to avoid blending
  two different end states into one label.
""")
