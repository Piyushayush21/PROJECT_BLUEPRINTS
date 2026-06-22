"""
SFEWS — Feature Engineering (vectorized)
==========================================
See module docstring in build_features.py (v1) for full design rationale on
leakage discipline, horizon framing, and feature families. This version
replaces the per-row Python loop with vectorized pandas groupby/rolling/shift
operations for performance at panel scale (140K+ rows).

Key idea for the label (no leakage):
  For each company, fail_month is fixed. At row t:
    label = 1  if  t < fail_month <= t + HORIZON_MONTHS
    label = 0  if company never fails, OR fails after t+HORIZON
  Rows where company already failed at/before t are dropped (post-failure rows
  don't exist anyway since simulate_lifecycle stops at fail_month, but we
  guard for it).
"""

import numpy as np
import pandas as pd

HORIZON_MONTHS = 6
MIN_HISTORY_MONTHS = 3
ROLL_SHORT, ROLL_LONG = 3, 6

METRIC_COLS = ["mrr", "cash_on_hand", "burn_rate", "runway_months", "churn_rate",
               "cac_payback_months", "nps_score", "headcount", "attrition_rate",
               "product_release_count", "press_mentions", "glassdoor_rating",
               "ltv_cac_ratio", "growth_rate_mom"]

TREND_COLS = ["mrr", "churn_rate", "runway_months", "headcount", "nps_score", "burn_rate"]
VOL_COLS = ["churn_rate", "growth_rate_mom", "burn_rate"]


def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope; window length varies near series start (handled per-call)."""
    def slope_fn(y):
        n = len(y)
        if n < 2 or np.all(np.isnan(y)):
            return 0.0
        x = np.arange(n)
        x_mean = x.mean()
        y_mean = np.nanmean(y)
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(np.nansum((x - x_mean) * (y - y_mean)) / denom)

    return s.rolling(window=window, min_periods=2).apply(slope_fn, raw=True)


def _trailing_run_vectorized(flag: pd.Series) -> pd.Series:
    """Length of consecutive-True run ending at each row (within a single company)."""
    flag = flag.astype(int)
    # reset counter to 0 whenever flag is 0, else increment
    grp = (flag == 0).cumsum()
    run = flag.groupby(grp).cumsum()
    return run


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["company_id", "month_index"]).reset_index(drop=True).copy()
    g = df.groupby("company_id", sort=False)

    out = df[["company_id", "month_index", "fail_month", "failed", "acquired"]].copy()

    # ---- 1. point-in-time snapshot (just rename current values) ----
    for col in METRIC_COLS:
        out[f"{col}_latest"] = df[col]

    # ---- 2. trend / momentum ----
    for col in TREND_COLS:
        out[f"{col}_slope_3m"] = g[col].transform(lambda s: _rolling_slope(s, ROLL_SHORT)).fillna(0.0)
        out[f"{col}_slope_6m"] = g[col].transform(lambda s: _rolling_slope(s, ROLL_LONG)).fillna(0.0)
        base_6m = g[col].shift(ROLL_LONG - 1)
        out[f"{col}_pct_change_6m"] = np.where(
            (base_6m.notna()) & (base_6m != 0),
            (df[col] - base_6m) / base_6m.abs(),
            0.0,
        )

    # ---- 3. volatility ----
    for col in VOL_COLS:
        out[f"{col}_volatility_6m"] = g[col].transform(lambda s: s.rolling(ROLL_LONG, min_periods=2).std()).fillna(0.0)

    # ---- 4. cumulative stress counters ----
    out["consecutive_neg_growth_months"] = g["growth_rate_mom"].transform(lambda s: _trailing_run_vectorized(s < 0))
    out["consecutive_low_runway_months"] = g["runway_months"].transform(lambda s: _trailing_run_vectorized(s < 6))
    out["consecutive_high_churn_months"] = g["churn_rate"].transform(lambda s: _trailing_run_vectorized(s > 0.15))
    out["founder_conflict_ever"] = g["founder_conflict_flag"].transform(lambda s: s.cummax())

    def months_since_flag(s):
        idx = np.arange(len(s))
        flagged_idx = np.where(s.values == 1)[0]
        result = np.full(len(s), 999)
        last_flag = -999
        for i in range(len(s)):
            if s.values[i] == 1:
                last_flag = i
            result[i] = i - last_flag if last_flag >= 0 else 999
        return pd.Series(result, index=s.index)

    out["months_since_founder_conflict"] = g["founder_conflict_flag"].transform(months_since_flag)

    # ---- 5. efficiency ratios ----
    out["burn_multiple"] = np.where(df["mrr"] > 0, df["burn_rate"] / df["mrr"], 99.0)
    cum_burn = g["burn_rate"].transform(lambda s: s.cumsum())
    out["capital_efficiency"] = np.where(cum_burn > 0, df["mrr"] * 12 / cum_burn, 0.0)
    out["ltv_cac_slope_6m"] = g["ltv_cac_ratio"].transform(lambda s: _rolling_slope(s, ROLL_LONG)).fillna(0.0)

    # ---- 6. composite hand-crafted risk index ----
    risk = np.zeros(len(out))
    risk += 25 * (out["runway_months_latest"] < 6)
    risk += 15 * (out["consecutive_neg_growth_months"] >= 2)
    risk += 15 * (out["churn_rate_latest"] > 0.20)
    risk += 10 * (out["consecutive_high_churn_months"] >= 2)
    risk += 10 * (out["ltv_cac_ratio_latest"] < 1.0)
    risk += 10 * (out["attrition_rate_latest"] > 0.15)
    risk += 10 * (out["glassdoor_rating_latest"] < 2.5)
    risk += 5 * out["founder_conflict_ever"]
    out["composite_risk_index"] = np.clip(risk, 0, 100)

    # ---- LABEL (strictly forward-looking, no leakage) ----
    fm = out["fail_month"]
    t = out["month_index"]
    will_fail_in_window = (fm > t) & (fm <= t + HORIZON_MONTHS)
    out["label_fail_within_6m"] = will_fail_in_window.astype(int)

    # ---- filters: enforce min history, drop rows at/after failure ----
    out["already_failed_at_t"] = (fm != -1) & (fm <= t)
    already_acquired_at_t = out["acquired"] & (fm != -1) & (t >= fm)
    keep = (t >= MIN_HISTORY_MONTHS) & (~out["already_failed_at_t"]) & (~already_acquired_at_t)
    out = out[keep].drop(columns=["already_failed_at_t", "fail_month", "failed", "acquired"]).reset_index(drop=True)

    return out


def attach_static_features(feat_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    static_cols = ["company_id", "sector", "region", "founded_year", "founder_count",
                   "founder_prior_exits", "founder_technical_pct", "initial_funding_stage",
                   "initial_capital"]
    merged = feat_df.merge(master_df[static_cols], on="company_id", how="left")
    merged["company_age_months_at_t"] = merged["month_index"]
    return merged


if __name__ == "__main__":
    panel = pd.read_csv("data/raw/startup_monthly_panel.csv")
    master = pd.read_csv("data/raw/startup_master.csv")

    print("Engineering features (vectorized, leakage-checked)...")
    feat_df = engineer_features(panel)
    feat_df = attach_static_features(feat_df, master)
    feat_df.to_csv("data/processed/sfews_features.csv", index=False)

    print(f"Feature rows: {len(feat_df):,}")
    print(f"Feature columns: {feat_df.shape[1]}")
    print(f"Label balance:\n{feat_df['label_fail_within_6m'].value_counts(normalize=True)}")
    print(f"Companies represented: {feat_df['company_id'].nunique():,}")
