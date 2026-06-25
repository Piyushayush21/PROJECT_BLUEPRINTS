"""
SFEWS — Simple Snapshot Model (for manual single-entry prediction)
=====================================================================

Why this exists as a SEPARATE model from src/models/train.py:

The main pipeline's best model uses 56 features, many of which are rolling
trends/volatility computed from 3-6 months of history (mrr_slope_6m,
churn_rate_volatility_6m, consecutive_low_runway_months, etc.). Those cannot
be honestly filled in by a person typing into a form in one sitting -- they
require a time series. Silently defaulting them to 0 would not be "a
simplified model," it would be a CORRUPTED input to the real model, producing
a prediction that looks legitimate but isn't.

So: this script trains a second, smaller model using ONLY the 27 point-in-time
features that a person can plausibly know off the top of their head right now
(current MRR, current churn, founder background, etc.) -- no rolling-window
features at all. This is the model behind the "Quick Risk Assessment" form in
the Streamlit app. It is explicitly labeled as a separate, lower-fidelity
model in the UI and in this docstring -- trading some predictive power for
honest single-snapshot usability.

Expect (and report) somewhat lower PR-AUC than the full temporal model, since
we've deliberately dropped the trend/volatility features that the SHAP
analysis showed carry real signal (e.g. consecutive_low_runway_months,
mrr_slope_6m). That gap IS the cost of making the model usable without
historical data, and is reported explicitly rather than glossed over.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import joblib

RANDOM_STATE = 42
TARGET = "label_fail_within_6m"

# The 27 point-in-time-only features -- no rolling/lag features included.
SNAPSHOT_NUM_COLS = [
    "mrr_latest", "cash_on_hand_latest", "burn_rate_latest", "runway_months_latest",
    "churn_rate_latest", "cac_payback_months_latest", "nps_score_latest",
    "headcount_latest", "attrition_rate_latest", "product_release_count_latest",
    "press_mentions_latest", "glassdoor_rating_latest", "ltv_cac_ratio_latest",
    "growth_rate_mom_latest", "founder_conflict_ever", "burn_multiple",
    "capital_efficiency", "composite_risk_index", "founded_year", "founder_count",
    "founder_prior_exits", "founder_technical_pct", "initial_capital",
    "company_age_months_at_t",
]
SNAPSHOT_CAT_COLS = ["sector", "region", "initial_funding_stage"]
SNAPSHOT_FEATURE_COLS = SNAPSHOT_NUM_COLS + SNAPSHOT_CAT_COLS


def company_level_split(df, test_size=0.15, val_size=0.15, seed=RANDOM_STATE):
    companies = df["company_id"].astype(str).unique().tolist()
    train_ids, test_ids = train_test_split(companies, test_size=test_size, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size / (1 - test_size), random_state=seed)
    return (df[df["company_id"].isin(train_ids)].reset_index(drop=True),
            df[df["company_id"].isin(val_ids)].reset_index(drop=True),
            df[df["company_id"].isin(test_ids)].reset_index(drop=True))


def main():
    df = pd.read_csv("data/processed/sfews_features.csv")
    train_df, val_df, test_df = company_level_split(df)

    pre = ColumnTransformer(transformers=[
        ("num", StandardScaler(), SNAPSHOT_NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), SNAPSHOT_CAT_COLS),
    ])

    X_train, y_train = train_df[SNAPSHOT_FEATURE_COLS], train_df[TARGET]
    X_val, y_val = val_df[SNAPSHOT_FEATURE_COLS], val_df[TARGET]
    X_test, y_test = test_df[SNAPSHOT_FEATURE_COLS], test_df[TARGET]

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Snapshot-only model | class imbalance ratio: {pos_weight:.2f}:1")

    pipe = Pipeline([("pre", pre), ("clf", xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        random_state=RANDOM_STATE, n_jobs=-1))])
    pipe.fit(X_train, y_train)

    val_proba = pipe.predict_proba(X_val)[:, 1]
    print(f"Validation ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")
    print(f"Validation PR-AUC:  {average_precision_score(y_val, val_proba):.4f}")

    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    except ImportError:
        calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)

    test_proba = calibrated.predict_proba(X_test)[:, 1]
    test_metrics = {
        "roc_auc": round(roc_auc_score(y_test, test_proba), 4),
        "pr_auc": round(average_precision_score(y_test, test_proba), 4),
        "brier_score": round(brier_score_loss(y_test, test_proba), 4),
    }
    print(f"\nHeld-out TEST metrics (snapshot-only model): {test_metrics}")
    print("(Compare to full temporal model in reports/test_metrics.json -- "
          "the gap is the honest cost of removing rolling-history features.)")

    joblib.dump(calibrated, "src/models/sfews_snapshot_model.joblib")
    joblib.dump(pipe, "src/models/sfews_snapshot_raw_model.joblib")
    with open("src/models/snapshot_feature_columns.json", "w") as f:
        json.dump({
            "feature_cols": SNAPSHOT_FEATURE_COLS,
            "num_cols": SNAPSHOT_NUM_COLS,
            "cat_cols": SNAPSHOT_CAT_COLS,
        }, f, indent=2)
    with open("reports/snapshot_model_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nSaved: src/models/sfews_snapshot_model.joblib, "
          "src/models/snapshot_feature_columns.json, "
          "reports/snapshot_model_test_metrics.json")


if __name__ == "__main__":
    main()
