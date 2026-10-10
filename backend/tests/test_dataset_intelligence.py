"""Tests for the Dataset Intelligence and Transparent Model Recommendation systems.

Validates:
- Feature profiling & type detection
- Missing percentages and statistics (mean, median, std, min, max, range)
- Feature composition (numeric, categorical, text)
- Target metadata & problem type detection
- Data quality scoring & factual observations
- Dynamic recommendation reason & dataset evidence
- Unit 5 workflow notes
- Dashboard summary enhancements
- Intelligence API endpoints
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.eda import column_profiles
from app.services.dashboard_service import dashboard_summary
from app.services.intelligence_service import (
    dataset_intelligence,
    enhanced_recommendation,
)
from app.services.store import DatasetRecord, store


@pytest.fixture
def sample_df():
    """Create a realistic synthetic student dataset for deterministic testing."""
    np.random.seed(42)
    n = 250
    return pd.DataFrame({
        "study_hours": np.random.uniform(1, 10, n),
        "attendance": np.random.uniform(50, 100, n),
        "previous_score": np.random.uniform(40, 100, n),
        "exam_score": np.random.uniform(40, 100, n),  # regression target
        "gender": np.random.choice(["Male", "Female"], n),
        "internet_access": np.random.choice(["Yes", "No"], n),
        "student_notes": [f"Student note id_{i} with unique comments" for i in range(n)],  # text feature
        "placement_status": np.random.choice(["Placed", "Not Placed"], n, p=[0.7, 0.3]),  # classification target
    })


@pytest.fixture
def registered_dataset(sample_df):
    """Register sample_df in the store as a test dataset."""
    did = "ds_test_intel_001"
    rec = DatasetRecord(
        id=did,
        filename="test_students.csv",
        extension="csv",
        storage_path="",
        file_size=10240,
        num_rows=len(sample_df),
        num_cols=len(sample_df.columns),
        column_names=list(sample_df.columns),
        dtypes={c: str(sample_df[c].dtype) for c in sample_df.columns},
        missing_summary={c: int(sample_df[c].isnull().sum()) for c in sample_df.columns},
        duplicate_count=0,
        detected_targets=["placement_status", "exam_score"],
    )
    store.add_dataset(rec, sample_df)
    return did


def test_column_profiles_enrichment(sample_df):
    """Verify column profiles compute enriched statistics."""
    meta = {
        "dtypes": {c: str(sample_df[c].dtype) for c in sample_df.columns},
        "missing_summary": {c: int(sample_df[c].isnull().sum()) for c in sample_df.columns},
    }
    profiles = column_profiles(sample_df, meta)
    assert len(profiles) == len(sample_df.columns)

    # Check numeric profile
    study_prof = next(p for p in profiles if p["name"] == "study_hours")
    assert study_prof["kind"] == "numeric"
    assert study_prof["min"] is not None
    assert study_prof["max"] is not None
    assert study_prof["mean"] is not None
    assert study_prof["median"] is not None
    assert study_prof["std"] is not None
    assert study_prof["missing_pct"] == 0.0

    # Check categorical profile
    gender_prof = next(p for p in profiles if p["name"] == "gender")
    assert gender_prof["kind"] == "categorical"
    assert "top_categories" in gender_prof
    assert len(gender_prof["top_categories"]) == 2


def test_compute_dataset_intelligence(registered_dataset):
    """Verify dataset_intelligence generates accurate factual structure."""
    intel = dataset_intelligence(registered_dataset)

    assert intel["dataset_id"] == registered_dataset
    assert intel["rows"] == 250
    assert intel["columns"] == 8
    assert intel["numeric_features"] == 4  # study_hours, attendance, previous_score, exam_score
    assert intel["categorical_features"] == 3  # gender, internet_access, placement_status
    assert intel["text_features"] == 1  # student_notes
    assert intel["duplicate_rows"] == 0
    assert intel["missing_values"] == 0

    # Composition
    comp = intel["composition"]
    assert comp["numeric"]["count"] == 4
    assert comp["categorical"]["count"] == 3
    assert comp["text"]["count"] == 1

    # Story
    assert len(intel["story"]) >= 3
    assert any("250" in s for s in intel["story"])

    # Target overview
    target_ov = intel["target_overview"]
    assert target_ov is not None
    assert target_ov["column"] == "placement_status"
    assert target_ov["problem_type"] == "classification"
    assert target_ov["class_count"] == 2

    # Data Quality
    quality = intel["quality"]
    # 100 minus 3 point deduction for 1 high-cardinality column (student_notes)
    assert quality["score"] == 97
    assert len(quality["high_cardinality"]) == 1
    assert quality["high_cardinality"][0]["column"] == "student_notes"
    assert len(quality["observations"]) > 0

    # Insights
    assert len(intel["insights"]) >= 3


def test_enhanced_model_recommendation_classification(registered_dataset):
    """Verify transparent recommendation for classification."""
    rec = enhanced_recommendation(
        dataset_id=registered_dataset,
        target_column="placement_status",
        problem_type="classification",
    )
    assert rec["recommended_model"] == "Random Forest"
    assert "heuristic" in rec["recommendation_reason"]
    assert "250" in rec["recommendation_reason"]
    assert rec["recommendation_evidence"]["rows"] == 250
    assert rec["recommendation_evidence"]["problem_type"] == "classification"
    assert len(rec["workflow_notes"]) > 0


def test_enhanced_model_recommendation_regression(registered_dataset):
    """Verify transparent recommendation for regression."""
    rec = enhanced_recommendation(
        dataset_id=registered_dataset,
        target_column="exam_score",
        problem_type="regression",
    )
    assert rec["recommended_model"] in ("Random Forest Regressor", "Linear Regression")
    assert "continuous" in rec["recommendation_reason"].lower()
    assert rec["recommendation_evidence"]["problem_type"] == "regression"


def test_dashboard_summary_with_intelligence(registered_dataset):
    """Verify dashboard summary includes enhanced composition and target info."""
    summary = dashboard_summary(registered_dataset, "session_test_dash")
    assert summary["total_records"] == 250
    assert summary["numeric_features"] == 4
    assert summary["categorical_features"] in (3, 4)
    assert summary["detected_target"] == "placement_status"
    assert summary["detected_problem_type"] == "classification"
    assert "detected_target_breakdown" in summary
    assert summary["detected_target_breakdown"]["kind"] == "classification"


def test_intelligence_api_endpoints(registered_dataset):
    """Test HTTP API endpoints for dataset intelligence and recommendation."""
    client = TestClient(app)

    # GET /api/intelligence/{id}
    res = client.get(f"/api/intelligence/{registered_dataset}")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert "intelligence" in body
    assert body["intelligence"]["rows"] == 250

    # GET /api/intelligence/{id}/recommendation
    res_rec = client.get(
        f"/api/intelligence/{registered_dataset}/recommendation?target=placement_status&problem_type=classification"
    )
    assert res_rec.status_code == 200
    rec_body = res_rec.json()
    assert rec_body["success"] is True
    assert "recommendation" in rec_body
    assert rec_body["recommendation"]["recommended_model"] == "Random Forest"
    assert len(rec_body["recommendation"]["workflow_notes"]) > 0


def test_training_config_endpoint_and_model_options(registered_dataset):
    """Regression test: verify /api/model/{id}/training-config succeeds without NameError,
    auto-detects target candidates, and populates model_options."""
    client = TestClient(app)

    res = client.get(f"/api/model/config/{registered_dataset}")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["success"] is True
    assert body["auto_detected"] is True
    assert "placement_status" in body["detected_targets"]
    assert "exam_score" in body["detected_targets"]
    # exam_score has higher priority in TARGET_COLUMN_KEYWORDS
    assert body["default_target"] == "exam_score"

    analysis = body["analysis"]
    assert analysis["target_column"] == "exam_score"
    assert analysis["problem_type"] == "regression"
    assert len(analysis["model_options"]) == 3
    assert "Random Forest Regressor" in analysis["model_options"]
    assert analysis["recommended_model"] in ("Random Forest Regressor", "Linear Regression")
    assert analysis["distinct_values"] > 0
    assert analysis["missing_values"] == 0
    assert analysis["recommendation_reason"] is not None
    assert analysis["recommendation_evidence"]["rows"] == 250
    assert len(analysis["workflow_notes"]) > 0
