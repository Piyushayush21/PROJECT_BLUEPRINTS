# 🛰️ SFEWS — Startup Failure Early Warning System

A 6-month-ahead startup failure prediction system trained on a simulated
longitudinal operating panel — leakage-safe feature engineering, calibrated
ML pipeline, SHAP explainability, Streamlit app, and Power BI dashboard.

**[Live Demo (Streamlit)](#) · [Power BI Build Guide](powerbi/POWERBI_GUIDE.md) · [Model Card](reports/model_card.md)**

---

## Why this project is different from the typical "startup success" Kaggle notebook

Most public startup-failure portfolio projects use the same handful of static
Crunchbase/Kaggle CSVs — one row per company, no time dimension, already
survivorship-biased toward companies that raised funding at all. That setup
can only answer *"did this company eventually fail?"* after the fact. It
cannot answer the actually useful question: **"is this specific company, right
now, showing the early warning signs of failure in the next 6 months?"**

SFEWS instead simulates an **18-42 month monthly operating panel** per company
(burn, churn, runway, hiring, NPS, founder dynamics) from a structural causal
model calibrated against public industry statistics, then frames the ML
problem as a genuine sliding-window early-warning classifier — the same shape
of problem a real fund's portfolio-monitoring tool or a real fintech credit-risk
system would solve. The dataset is openly disclosed as synthetic (see
[Data Disclosure](#data-disclosure) below) — that disclosure is itself part of
the engineering judgment on display here.

## What's in this repo

| Layer | What it demonstrates |
|---|---|
| **Data generation** (`src/data/`) | Structural simulation design, not naive random data |
| **Feature engineering** (`src/features/`) | Leakage-safe sliding-window labeling, vectorized pandas, 56 engineered features |
| **ML pipeline** (`src/models/`) | Company-level train/val/test split, 4-model comparison, class-imbalance handling, isotonic calibration, cost-based thresholding |
| **Explainability** (`src/explainability/`) | SHAP global + local explanations, plain-English narrative generation |
| **Streamlit app** (`app/`) | 5-page interactive product: **quick risk assessment form**, portfolio monitor, company drilldown, what-if simulator, model transparency page |
| **Power BI dashboard** (`powerbi/`) | Star-schema export pipeline + full DAX measure guide |
| **Tests** (`tests/`) | 10 automated tests, focused on leakage safety and model sanity |
| **CI/CD** (`.github/workflows/`) | Automated testing + scheduled retraining pipeline |

## Results

| Metric | Held-out Test Set |
|---|---|
| ROC-AUC | **0.970** |
| PR-AUC | **0.702** |
| Recall @ business-optimal threshold | **91%** |
| Precision @ business-optimal threshold | **67%** |

Full methodology, limitations, and cohort-stability checks in the
[Model Card](reports/model_card.md).

## Quickstart

```bash
git clone https://github.com/<your-username>/sfews.git
cd sfews
pip install -r requirements.txt

# Run the Streamlit app (uses the model + data already checked into the repo)
streamlit run app/Home.py
```

### Reproducing the full pipeline from scratch

```bash
python src/data/generate_dataset.py        # ~5 sec  -> data/raw/*.csv
python src/features/build_features.py      # ~10 sec -> data/processed/sfews_features.csv
python src/models/train.py                 # ~30 sec -> src/models/*.joblib, reports/*.json
python src/models/train_snapshot_model.py  # ~15 sec -> snapshot model for the manual-input form
python src/explainability/shap_explain.py  # ~20 sec -> reports/shap_*.csv
python powerbi/export_for_powerbi.py       # ~5 sec  -> powerbi/exports/*.csv
pytest tests/ -v                            # confirm 10/10 passing
```

To scale up the dataset: edit `N_COMPANIES` in `src/data/generate_dataset.py`
(the repo ships with 1,500 companies for a lean clone; results were also
validated at 4,200 companies with consistent metrics).

## Repository structure

```
sfews/
├── app/                          # Streamlit application
│   ├── Home.py                   #   Landing page + dataset disclosure
│   └── pages/
│       ├── 1_Portfolio_Monitor.py
│       ├── 2_Company_Drilldown.py
│       ├── 3_What_If_Simulator.py
│       ├── 4_Model_Performance.py
│       └── 5_Quick_Risk_Assessment.py   # Manual input -> live prediction form
├── data/
│   ├── raw/                      # Generated synthetic panel + master table
│   └── processed/                # Engineered feature matrix
├── notebooks/
│   └── 01_EDA.ipynb              # Exploratory analysis, fully executed
├── powerbi/
│   ├── export_for_powerbi.py     # Star-schema CSV export pipeline
│   ├── exports/                  # Fact + dimension tables
│   └── POWERBI_GUIDE.md          # Step-by-step dashboard build guide + DAX
├── reports/
│   ├── model_card.md             # Methodology, metrics, limitations
│   ├── model_comparison.csv
│   ├── test_metrics.json
│   └── shap_global_importance.csv
├── src/
│   ├── data/generate_dataset.py
│   ├── features/build_features.py
│   ├── models/train.py
│   └── explainability/shap_explain.py
├── tests/
│   ├── test_feature_engineering.py
│   └── test_model_artifacts.py
├── .github/workflows/
│   ├── ci.yml                    # Test on every push/PR
│   └── retrain.yml               # Scheduled monthly retrain
├── requirements.txt
└── README.md
```

## Data Disclosure

**This dataset is synthetic.** It is generated by a structural causal
simulation (`src/data/generate_dataset.py`) calibrated against publicly
reported industry statistics — not scraped from Crunchbase, PitchBook, or any
proprietary source. This is disclosed prominently in the app itself, the
model card, and here. Absolute performance numbers should be read as a
**methodology demonstration** (leakage-safe panel feature engineering,
calibration, explainability, cost-based decisioning) that transfers directly
to a real licensed dataset, not as a real-world predictive-accuracy claim.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · Streamlit ·
Power BI (DAX) · pytest · GitHub Actions

## License

MIT — see [LICENSE](LICENSE).
