# Model Card — SFEWS Failure Risk Classifier

## Overview
- **Task**: Binary classification — will a startup fail within the next 6 months, given its operating history up to the current month?
- **Type**: Tree ensemble (model selected by validation PR-AUC across Logistic Regression, Random Forest, XGBoost, LightGBM — see `reports/model_comparison.csv` for the exact run this repo ships with), isotonic-calibrated
- **Inputs**: 56 engineered features per company-month (point-in-time metrics, 3/6-month trends, volatility, stress counters, efficiency ratios, firmographics)
- **Output**: Calibrated probability [0, 1] + risk tier (Low / Watch / High)

## Training Data
**Synthetic, calibration-informed** — see `src/data/generate_dataset.py` for the full
generative model and `app/Home.py` "About this dataset" panel for the public-facing
disclosure. 1,500 simulated companies, 18-42 months each, ~44K company-month
observations, generated from a structural causal model (not i.i.d. random noise)
calibrated loosely against CB Insights post-mortem reports, the Startup Genome
Report, and First Round Capital operating benchmarks for sector failure-rate
multipliers, typical burn multiples, and churn distributions. (The generator
supports scaling to any company count — results were also validated at
N=4,200 with consistent metrics; the repo ships the smaller run to keep
clone size and Streamlit Cloud deploy time low.)

**This is disclosed prominently and is not intended to be cited as a real-world
predictive accuracy claim.** The transferable artifact is the methodology
(leakage-safe feature engineering, company-level splitting, calibration,
SHAP explainability, cost-based thresholding) — all of which apply unchanged
to a real Crunchbase/PitchBook-sourced panel if one were licensed.

## Evaluation

| Metric | Validation | Held-out Test |
|---|---|---|
| ROC-AUC | 0.960 | 0.970 |
| PR-AUC | 0.736 | 0.702 |
| F1 @ 0.5 threshold | 0.758 | 0.770 |
| Brier Score (calibration) | — | 0.036 |

Split: 70/15/15 by **company_id** (not row), so no company's months appear
in more than one split. Class balance: ~9.8% positive (fails within 6mo).

At the business-cost-optimal threshold (~0.115, assuming a 5:1 cost ratio of
missing a true failure vs. a false alarm): **91% recall, 67% precision** on
the held-out test set.

### Cohort stability (founding-year, 2014-2022)
ROC-AUC ranged 0.964–0.984 across cohorts with no monotonic drift — no
evidence of era-specific leakage from the macro-shock simulation mechanism.

## Explainability
SHAP `TreeExplainer` on the raw (uncalibrated) tree model. Top global
drivers: cash runway (by a wide margin), consecutive low-runway months, the
hand-crafted composite risk index, and runway trend — consistent with
financial intuition that cash-out is the dominant proximate trigger of
startup failure even when the root cause (poor PMF, team conflict, weak
unit economics) lies elsewhere upstream. Local explanations are converted to
plain-English narratives (`src/explainability/shap_explain.py::_build_narrative`)
for non-technical stakeholders (founders, board members).

**Calibration/SHAP tradeoff**: SHAP attribution is computed on the raw model
because `TreeExplainer` needs direct tree access; the displayed probability
uses the calibrated wrapper. The two can diverge slightly in absolute scale
(calibration is a monotonic-ish post-hoc remap) — SHAP should be read as
"what's driving the ranking," not as decomposing the exact calibrated number.

## Intended Use
- Portfolio-level monitoring for VC funds / accelerators tracking many companies
- Early internal signal for founders/operators to self-assess operating health
- Teaching/portfolio artifact demonstrating leakage-safe longitudinal ML design

## Out-of-Scope / Misuse
- **Not a real underwriting or investment-decision tool** — trained on synthetic data
- **Not causal** — the What-If Simulator shows learned associations, not
  guaranteed real-world effects of an intervention
- **Not validated against real distribution shift** (e.g., an actual macro
  shock outside the simulated training distribution)
- Should not be used to make employment, compensation, or individual-founder
  decisions about real people

## Limitations
1. Synthetic data, openly disclosed — absolute metrics are a methodology
   demonstration, not a real-world accuracy benchmark.
2. Acquired companies are excluded from training rows once acquired
   (different end-state than failure or continued operation) — the model
   has not learned to distinguish "will be acquired" from "will survive."
3. Composite heuristic risk index is a hand-crafted baseline for sanity-checking
   the ML model's outputs, not itself a validated scoring system.
4. No demographic/protected-attribute fairness audit was performed because
   the synthetic data contains no protected attributes — a real deployment
   on real company/founder data would require one (founder
   gender/ethnicity/age proxies are a known issue in real startup-funding ML
   and should be explicitly tested for before any production use).

## Maintenance
Retraining pipeline: `src/models/train.py`, automated via
`.github/workflows/retrain.yml` on a schedule / on new data push.

## Companion Model: Manual-Input Snapshot Model

`src/models/train_snapshot_model.py` trains a **separate, simpler model**
used by the Streamlit app's "Quick Risk Assessment" form. The main model
above uses 56 features, most requiring 3-6 months of historical data
(rolling slopes, volatility, consecutive-stress counters) that a person
cannot honestly type into a form in one sitting. The snapshot model uses only
the 27 point-in-time features a person can reasonably know right now.

| Metric | Full Temporal Model | Snapshot-Only Model |
|---|---|---|
| Test ROC-AUC | 0.970 | 0.961 |
| Test PR-AUC | 0.702 | 0.667 |
| Test Brier Score | 0.036 | 0.038 |

The gap is the explicit, honest cost of removing rolling-history features —
reported here rather than hidden. Both models are isotonic-calibrated and
share the same SHAP-based explainability approach.
