"""
Tests for SFEWS feature engineering — focused heavily on leakage safety,
since that's the single most important correctness property of this project.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.features.build_features import engineer_features, HORIZON_MONTHS, MIN_HISTORY_MONTHS


@pytest.fixture(scope="module")
def tiny_panel():
    """Hand-constructed mini panel: one company that fails at month 10."""
    rows = []
    for m in range(12):
        rows.append(dict(
            company_id="TEST-001", month_index=m,
            mrr=10000 + m * 100, cash_on_hand=max(50000 - m * 5000, 0),
            burn_rate=8000, runway_months=max(6 - m * 0.5, 0),
            churn_rate=0.05 + m * 0.01, cac_payback_months=10, nps_score=40,
            headcount=10, attrition_rate=0.05, founder_conflict_flag=0,
            product_release_count=1, press_mentions=0, glassdoor_rating=3.5,
            ltv_cac_ratio=2.0, growth_rate_mom=0.02 - m * 0.005,
            failed=(m == 9), acquired=False, fail_month=9,
        ))
    # second company: healthy survivor, never fails
    for m in range(15):
        rows.append(dict(
            company_id="TEST-002", month_index=m,
            mrr=50000 + m * 2000, cash_on_hand=1_000_000,
            burn_rate=20000, runway_months=40, churn_rate=0.03,
            cac_payback_months=6, nps_score=70, headcount=20,
            attrition_rate=0.02, founder_conflict_flag=0,
            product_release_count=2, press_mentions=3, glassdoor_rating=4.5,
            ltv_cac_ratio=4.0, growth_rate_mom=0.05,
            failed=False, acquired=False, fail_month=-1,
        ))
    return pd.DataFrame(rows)


def test_no_post_failure_rows_in_output(tiny_panel):
    """A company should never have a feature row at or after its fail_month."""
    feat = engineer_features(tiny_panel)
    failing_company = feat[feat["company_id"] == "TEST-001"]
    assert failing_company["month_index"].max() < 9, (
        "Found a feature row at or after the failure month — this is a leakage bug."
    )


def test_label_only_looks_forward(tiny_panel):
    """For the failing company, label should be 1 exactly for months in (fail_month - HORIZON, fail_month)."""
    feat = engineer_features(tiny_panel)
    failing_company = feat[feat["company_id"] == "TEST-001"].sort_values("month_index")

    for _, row in failing_company.iterrows():
        t = row["month_index"]
        expected = 1 if (9 > t and 9 <= t + HORIZON_MONTHS) else 0
        assert row["label_fail_within_6m"] == expected, (
            f"Mislabeled row at t={t}: got {row['label_fail_within_6m']}, expected {expected}"
        )


def test_min_history_enforced(tiny_panel):
    """No feature rows should exist before MIN_HISTORY_MONTHS."""
    feat = engineer_features(tiny_panel)
    assert feat["month_index"].min() >= MIN_HISTORY_MONTHS


def test_survivor_never_labeled_positive(tiny_panel):
    """A company that never fails should have label=0 for every row."""
    feat = engineer_features(tiny_panel)
    survivor = feat[feat["company_id"] == "TEST-002"]
    assert (survivor["label_fail_within_6m"] == 0).all()


def test_no_nans_in_output(tiny_panel):
    feat = engineer_features(tiny_panel)
    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    assert feat[numeric_cols].isna().sum().sum() == 0, "Found NaNs in engineered features."


def test_slope_features_use_only_past_data(tiny_panel):
    """
    Sanity check: the 6-month MRR slope at month t should be computable purely
    from months [t-5, t] -- changing a FUTURE row's mrr should not change the
    slope feature already computed at an earlier t.
    """
    feat_before = engineer_features(tiny_panel)
    mutated = tiny_panel.copy()
    mutated.loc[(mutated["company_id"] == "TEST-002") & (mutated["month_index"] == 14), "mrr"] = 999_999_999
    feat_after = engineer_features(mutated)

    early_row_before = feat_before[(feat_before["company_id"] == "TEST-002") & (feat_before["month_index"] == 5)]
    early_row_after = feat_after[(feat_after["company_id"] == "TEST-002") & (feat_after["month_index"] == 5)]

    pd.testing.assert_series_equal(
        early_row_before["mrr_slope_6m"].reset_index(drop=True),
        early_row_after["mrr_slope_6m"].reset_index(drop=True),
        check_names=False,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
