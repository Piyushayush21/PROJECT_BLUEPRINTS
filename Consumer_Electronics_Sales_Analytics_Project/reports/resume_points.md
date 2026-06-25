# Resume Points — SFEWS Project

Pick 3-4 of these depending on the role (DS/MLE-leaning vs. analytics/BI-leaning).
Numbers are pulled directly from `reports/model_card.md` — keep them in sync if you retrain.

## Strong, general-purpose set (DS / MLE roles)

- Designed and built an end-to-end **early-warning ML system** predicting
  6-month startup failure risk from a simulated 1,500-company, 44K-row
  monthly operating panel, achieving **0.97 ROC-AUC / 0.70 PR-AUC** on a
  held-out test set with **company-level (not row-level) train/test splitting**
  to eliminate temporal leakage.

- Engineered **56 leakage-safe features** (rolling trends, volatility,
  consecutive-stress counters, efficiency ratios) using vectorized pandas
  groupby/rolling operations across a 44K-row panel; validated zero
  look-ahead bias with automated pytest checks on synthetic edge cases.

- Benchmarked 4 model families (Logistic Regression, Random Forest, XGBoost,
  LightGBM) under **9.8% class imbalance**, selected by PR-AUC rather than
  accuracy, then **isotonic-calibrated** the winning model to produce
  decision-grade probabilities (Brier score 0.036) and derived a **cost-based
  classification threshold** (5:1 false-negative penalty) yielding 91% recall
  at 67% precision.

- Built a **SHAP-based explainability layer** translating local feature
  attributions into plain-English risk narratives for non-technical
  stakeholders, deployed via an interactive **Streamlit application** with
  portfolio monitoring, single-company drilldown, and a live what-if
  sensitivity simulator.

- Shipped with **automated CI/CD** (GitHub Actions: test-on-push + scheduled
  monthly retraining), a documented model card covering limitations and
  intended use, and 10 passing unit tests focused on data-leakage prevention.

## Analytics / BI-leaning set

- Built a **star-schema Power BI dashboard** (fact + 2 dimension tables, 5
  custom DAX measures) visualizing portfolio-wide failure risk across 1,500+
  simulated companies, with sector/region drill-down and a model-transparency
  page surfacing SHAP feature importance to non-technical stakeholders.

- Translated a calibrated ML risk score into a **3-page executive dashboard**
  (KPI summary, risk explorer matrix, model transparency) with consistent
  Low/Watch/High risk-tier thresholds shared between the BI tool and a
  companion Streamlit app.

- Designed a **scheduled data-refresh pipeline** (GitHub Actions → CSV export
  → BI ingestion) mirroring the producer/consumer pattern used in real BI
  team workflows where ML scoring runs on a schedule independent of the
  visualization layer.

## One-liner (LinkedIn headline / portfolio summary)

> Built an end-to-end startup failure early-warning system — leakage-safe
> feature engineering on a synthetic 44K-row operating panel, calibrated
> XGBoost/RF ensemble (0.97 ROC-AUC), SHAP explainability, Streamlit app, and
> Power BI dashboard, with full CI/CD and a published model card.

## Notes on talking about the synthetic data honestly

Don't hide that the dataset is synthetic — bring it up first, briefly, and
pivot immediately to why: real Crunchbase-style data is small, static, and
survivorship-biased, and doesn't support a genuine time-series early-warning
framing. Recruiters and interviewers respond far better to "I made a deliberate,
disclosed tradeoff here, and here's why" than to discovering it themselves and
wondering why you didn't mention it.
