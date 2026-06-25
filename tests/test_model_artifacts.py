"""Tests for trained model artifacts -- sanity checks on inference behavior."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))


@pytest.fixture(scope="module")
def artifacts():
    model = joblib.load(BASE / "src/models/sfews_calibrated_model.joblib")
    with open(BASE / "src/models/feature_columns.json") as f:
        meta = json.load(f)
    features = pd.read_csv(BASE / "data/processed/sfews_features.csv")
    return model, meta, features


def test_model_outputs_valid_probabilities(artifacts):
    model, meta, features = artifacts
    sample = features[meta["feature_cols"]].sample(200, random_state=1)
    proba = model.predict_proba(sample)[:, 1]
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_higher_runway_lowers_risk_all_else_equal(artifacts):
    """Monotonicity sanity check: more runway should not increase predicted risk."""
    model, meta, features = artifacts
    row = features[meta["feature_cols"]].sample(1, random_state=3).copy()
    low_runway = row.copy()
    low_runway["runway_months_latest"] = 1.0
    high_runway = row.copy()
    high_runway["runway_months_latest"] = 36.0

    p_low = model.predict_proba(low_runway)[:, 1][0]
    p_high = model.predict_proba(high_runway)[:, 1][0]
    assert p_high <= p_low, "Model assigns higher risk to MORE runway -- check feature direction."


def test_feature_columns_match_training_schema(artifacts):
    model, meta, features = artifacts
    for col in meta["feature_cols"]:
        assert col in features.columns, f"Missing expected feature column: {col}"


def test_no_missing_values_in_feature_matrix(artifacts):
    _, meta, features = artifacts
    assert features[meta["feature_cols"]].isna().sum().sum() == 0


@pytest.fixture(scope="module")
def snapshot_artifacts():
    model = joblib.load(BASE / "src/models/sfews_snapshot_model.joblib")
    with open(BASE / "src/models/snapshot_feature_columns.json") as f:
        meta = json.load(f)
    features = pd.read_csv(BASE / "data/processed/sfews_features.csv")
    return model, meta, features


def test_snapshot_model_outputs_valid_probabilities(snapshot_artifacts):
    model, meta, features = snapshot_artifacts
    sample = features[meta["feature_cols"]].sample(200, random_state=1)
    proba = model.predict_proba(sample)[:, 1]
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_snapshot_model_runway_direction(snapshot_artifacts):
    """A manually-entered low-runway company should score riskier than a high-runway twin."""
    model, meta, features = snapshot_artifacts
    row = features[meta["feature_cols"]].sample(1, random_state=5).copy()
    low_runway, high_runway = row.copy(), row.copy()
    low_runway["runway_months_latest"] = 1.0
    high_runway["runway_months_latest"] = 36.0
    p_low = model.predict_proba(low_runway)[:, 1][0]
    p_high = model.predict_proba(high_runway)[:, 1][0]
    assert p_high <= p_low


def test_snapshot_feature_cols_have_no_rolling_features(snapshot_artifacts):
    """The whole point of the snapshot model is no rolling/lag features -- verify that holds."""
    _, meta, _ = snapshot_artifacts
    banned_substrings = ["slope_", "volatility_", "pct_change_", "consecutive_", "months_since_"]
    for col in meta["feature_cols"]:
        for banned in banned_substrings:
            assert banned not in col, f"Found rolling-history feature '{col}' in snapshot model -- defeats the purpose."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
