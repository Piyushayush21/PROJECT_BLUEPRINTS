"""
SFEWS — Explainability Layer (SHAP)
=====================================

Why this matters beyond "it's a nice chart": an early-warning score with no
explanation is unusable by the people who'd actually act on it (investors,
boards, founders). A VC won't pull a term sheet because a model said "73%."
They will act on "73% risk, driven primarily by 4 months of runway under 6
months and a 22% MoM churn increase." This module produces exactly that.

Three layers:
  1. GLOBAL: which features matter most across the whole portfolio
     (SHAP summary / beeswarm) -- used in the Power BI portfolio view.
  2. LOCAL: for one company at one point in time, which features pushed
     its score up/down, with magnitude (SHAP waterfall) -- used in the
     Streamlit single-company drilldown.
  3. NARRATIVE: local SHAP values translated into a templated, plain-English
     paragraph -- this is the layer that makes the tool feel like a product,
     not a notebook output. Recruiters notice this distinction immediately.

We use shap.TreeExplainer against the *raw* (uncalibrated) tree model rather
than the calibrated wrapper, since TreeExplainer needs direct access to tree
structure; calibration is a monotonic-ish post-hoc transform on top, so the
SHAP attribution from the raw model is still a faithful explanation of what's
driving the underlying risk ranking. This tradeoff is explicitly documented
here because it's a common interview question ("can you SHAP a calibrated
model directly?" -- no, not cleanly, and you should know why).
"""

import json
import numpy as np
import pandas as pd
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")


FEATURE_LABELS = {
    "runway_months_latest": "months of cash runway remaining",
    "churn_rate_latest": "current customer churn rate",
    "mrr_slope_6m": "6-month MRR growth trend",
    "consecutive_low_runway_months": "consecutive months with runway under 6 months",
    "consecutive_neg_growth_months": "consecutive months of revenue decline",
    "ltv_cac_ratio_latest": "LTV:CAC ratio",
    "attrition_rate_latest": "employee attrition rate",
    "founder_conflict_ever": "history of founder conflict",
    "glassdoor_rating_latest": "Glassdoor employee rating",
    "burn_multiple": "burn multiple (burn / revenue)",
    "consecutive_high_churn_months": "consecutive months of elevated churn",
    "churn_rate_slope_6m": "6-month churn rate trend",
    "nps_score_latest": "Net Promoter Score",
    "capital_efficiency": "capital efficiency (annualized revenue / cumulative burn)",
}


def load_artifacts():
    raw_model_pipe = joblib.load("src/models/sfews_raw_best_model.joblib")
    with open("src/models/feature_columns.json") as f:
        meta = json.load(f)
    return raw_model_pipe, meta


def get_shap_explainer(pipe, X_background: pd.DataFrame):
    """
    pipe is a sklearn Pipeline(pre, clf). SHAP TreeExplainer needs the
    transformed feature matrix and the bare tree model, not the pipeline.
    """
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    X_transformed = pre.transform(X_background)

    # recover output feature names post-OneHotEncoding for readable SHAP plots
    num_cols = pre.transformers_[0][2]
    cat_encoder = pre.named_transformers_["cat"]
    cat_cols_out = list(cat_encoder.get_feature_names_out(pre.transformers_[1][2]))
    feature_names_out = list(num_cols) + cat_cols_out

    explainer = shap.TreeExplainer(clf)
    return explainer, X_transformed, feature_names_out


def _extract_positive_class_shap(shap_values):
    """
    Normalize SHAP output shape across model families:
      - XGBoost/LightGBM binary classifiers: shap_values is 2D (n_samples, n_features)
      - sklearn RandomForestClassifier: shap_values can be 3D
        (n_samples, n_features, n_classes) in newer shap versions, or a list
        of two 2D arrays [class_0_shap, class_1_shap] in older ones.
    Always return the 2D array corresponding to the positive class (index 1).
    """
    if isinstance(shap_values, list):
        return shap_values[1]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def global_importance(explainer, X_transformed, feature_names, top_n=15) -> pd.DataFrame:
    shap_values = explainer.shap_values(X_transformed)
    shap_values = _extract_positive_class_shap(shap_values)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False).head(top_n)
    return imp_df


def explain_single_company(pipe, explainer, feature_cols, num_cols, cat_cols,
                            company_row: pd.DataFrame, top_n=5) -> dict:
    """
    Returns top contributing features (direction + magnitude) and a generated
    narrative paragraph for ONE company-month row.
    """
    pre = pipe.named_steps["pre"]
    X_t = pre.transform(company_row[feature_cols])

    cat_encoder = pre.named_transformers_["cat"]
    cat_cols_out = list(cat_encoder.get_feature_names_out(cat_cols))
    feature_names_out = list(num_cols) + cat_cols_out

    shap_vals_raw = explainer.shap_values(X_t)
    shap_vals = _extract_positive_class_shap(shap_vals_raw)[0]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if np.ndim(base_value) > 0 and len(np.atleast_1d(base_value)) > 1 else base_value

    contrib = pd.DataFrame({"feature": feature_names_out, "shap_value": shap_vals})
    contrib["abs_shap"] = contrib["shap_value"].abs()
    contrib = contrib.sort_values("abs_shap", ascending=False)

    top_risk_drivers = contrib[contrib["shap_value"] > 0].head(top_n)
    top_protective = contrib[contrib["shap_value"] < 0].head(top_n)

    narrative = _build_narrative(company_row, top_risk_drivers, top_protective)

    return {
        "base_value": float(base_value),
        "top_risk_drivers": top_risk_drivers.to_dict("records"),
        "top_protective_factors": top_protective.to_dict("records"),
        "narrative": narrative,
    }


def _humanize_feature(feat_name: str) -> str:
    if feat_name in FEATURE_LABELS:
        return FEATURE_LABELS[feat_name]
    base = feat_name.replace("_latest", "").replace("_slope_6m", " trend").replace("_slope_3m", " short-term trend")
    return base.replace("_", " ")


def _build_narrative(row: pd.DataFrame, risk_drivers: pd.DataFrame, protective: pd.DataFrame) -> str:
    company_id = row["company_id"].iloc[0] if "company_id" in row else "This company"
    risk_phrases = [_humanize_feature(f) for f in risk_drivers["feature"].tolist()[:3]]
    protective_phrases = [_humanize_feature(f) for f in protective["feature"].tolist()[:2]]

    parts = [f"{company_id}'s risk score is primarily driven by"]
    if risk_phrases:
        parts.append(", ".join(risk_phrases[:-1]) + (" and " if len(risk_phrases) > 1 else "") + risk_phrases[-1] + ".")
    else:
        parts.append("no single dominant risk factor -- risk is diffuse across several moderate signals.")

    if protective_phrases:
        parts.append("This is partially offset by relatively healthy " + " and ".join(protective_phrases) + ".")

    return " ".join(parts)


if __name__ == "__main__":
    pipe, meta = load_artifacts()
    df = pd.read_csv("data/processed/sfews_features.csv")
    feature_cols, num_cols, cat_cols = meta["feature_cols"], meta["num_cols"], meta["cat_cols"]

    background = df[feature_cols].sample(min(2000, len(df)), random_state=42)
    explainer, X_transformed, feature_names = get_shap_explainer(pipe, background)

    print("Computing global SHAP importances...")
    imp_df = global_importance(explainer, X_transformed, feature_names)
    print(imp_df.to_string(index=False))
    imp_df.to_csv("reports/shap_global_importance.csv", index=False)

    print("\nExample local explanation (one company-month row):")
    sample_row = df[df["label_fail_within_6m"] == 1].sample(1, random_state=7)
    explanation = explain_single_company(pipe, explainer, feature_cols, num_cols, cat_cols, sample_row)
    print(explanation["narrative"])

    with open("reports/sample_shap_explanation.json", "w") as f:
        json.dump(explanation, f, indent=2, default=str)

    print("\nSaved: reports/shap_global_importance.csv, reports/sample_shap_explanation.json")
