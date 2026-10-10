"""Unit and integration tests for missing-value handling, imputation,
target protection, and data cleaning explanation.

Covers the 10 scenarios requested in Section 23:
- TEST 1: Complete dataset -> training works
- TEST 2: Numeric missing values -> training works
- TEST 3: Categorical missing values -> training works
- TEST 4: Mixed numeric + categorical missing values -> training works
- TEST 5: Missing target values -> affected rows excluded -> clear message shown
- TEST 6: Column with 100% missing values -> safely detected -> excluded/warned
- TEST 7: Very high missing percentage -> warning shown -> no crash
- TEST 8: Prediction with missing feature input -> prediction works
- TEST 9: What-If with missing feature -> prediction works
- TEST 10: Dataset with no missing values -> existing workflow behaves exactly as before
- Additional tests: Missing analysis endpoint, cleaning report CSV, and student_performance_missing_10000.csv
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.eda import missing_analysis, generate_cleaning_report_csv
from app.ml.helpers import get_numeric_columns, get_categorical_like_columns
from app.ml.preprocessing import create_preprocessor
from app.ml.training import train_model, validate_before_training, ModelTrainingError
from app.ml.prediction import predict_single, build_feature_schema
from app.utils.file_utils import normalize_missing_markers, parse_tabular_bytes

client = TestClient(app)


def _synthetic_dataset(n=100, seed=42):
    np.random.seed(seed)
    return pd.DataFrame({
        "study_hours": np.random.uniform(1.0, 10.0, n),
        "attendance": np.random.uniform(50.0, 100.0, n),
        "department": np.random.choice(["CS", "Math", "Physics"], n),
        "gender": np.random.choice(["Male", "Female"], n),
        "final_exam_score": np.random.uniform(40.0, 100.0, n),
    })


def test_1_complete_dataset_training():
    """TEST 1: Complete dataset -> training works without error."""
    df = _synthetic_dataset(100)
    result = train_model(df, "final_exam_score", "Random Forest Regressor")
    assert result["pipeline"] is not None
    assert "metrics" in result
    assert result["target_missing_excluded"] == 0
    assert len(result["unusable_columns_excluded"]) == 0


def test_2_numeric_missing_values_training():
    """TEST 2: Numeric missing values -> training works with median imputation."""
    df = _synthetic_dataset(100)
    # Inject missing values into numeric columns
    df.loc[0:20, "study_hours"] = np.nan
    df.loc[10:30, "attendance"] = np.nan

    result = train_model(df, "final_exam_score", "Random Forest Regressor")
    assert result["pipeline"] is not None
    assert result["metrics"]["rmse"] > 0
    assert result["target_missing_excluded"] == 0


def test_3_categorical_missing_values_training():
    """TEST 3: Categorical missing values -> training works with most-frequent imputation."""
    df = _synthetic_dataset(100)
    # Inject missing values into categorical columns
    df.loc[5:25, "department"] = np.nan
    df.loc[15:35, "gender"] = np.nan

    result = train_model(df, "final_exam_score", "Random Forest Regressor")
    assert result["pipeline"] is not None
    assert result["metrics"]["rmse"] > 0


def test_4_mixed_numeric_and_categorical_missing_training():
    """TEST 4: Mixed numeric + categorical missing values -> training works."""
    df = _synthetic_dataset(100)
    df.loc[0:15, "study_hours"] = np.nan
    df.loc[10:25, "department"] = np.nan
    df.loc[20:35, "attendance"] = np.nan
    df.loc[30:45, "gender"] = np.nan

    result = train_model(df, "final_exam_score", "Random Forest Regressor")
    assert result["pipeline"] is not None
    assert "metrics" in result


def test_5_missing_target_values_excluded():
    """TEST 5: Missing target values -> affected rows excluded -> clear message shown."""
    df = _synthetic_dataset(100)
    # 12 rows missing target
    df.loc[0:11, "final_exam_score"] = np.nan

    # Preflight should report excluded rows
    validation = validate_before_training(df, "final_exam_score", "regression", "Random Forest Regressor")
    assert validation["ok"] is True
    assert any("12 rows have missing target values and will be excluded" in w for w in validation["warnings"])

    # Training should proceed on 88 valid rows
    result = train_model(df, "final_exam_score", "Random Forest Regressor")
    assert result["target_missing_excluded"] == 12
    assert result["n_train"] + result["n_test"] == 88


def test_6_column_with_100_percent_missing():
    """TEST 6: Column with 100% missing values -> safely detected -> excluded/warned."""
    df = _synthetic_dataset(100)
    df["empty_col"] = np.nan

    validation = validate_before_training(df, "final_exam_score", "regression", "Random Forest Regressor")
    assert any("empty_col' contains only missing values (100% missing)" in w for w in validation["warnings"])

    result = train_model(df, "final_exam_score", "Random Forest Regressor")
    assert "empty_col" in result["unusable_columns_excluded"]
    assert "empty_col" not in result["feature_columns"]


def test_7_very_high_missing_percentage():
    """TEST 7: Very high missing percentage (>40%) -> warning shown -> no crash."""
    df = _synthetic_dataset(100)
    # 55% missing
    df.loc[0:54, "study_hours"] = np.nan

    validation = validate_before_training(df, "final_exam_score", "regression", "Random Forest Regressor")
    assert any("very high missingness" in w for w in validation["warnings"])

    result = train_model(df, "final_exam_score", "Random Forest Regressor")
    assert result["pipeline"] is not None


def test_8_prediction_with_missing_feature_input():
    """TEST 8: Prediction with missing feature input -> prediction works cleanly."""
    df = _synthetic_dataset(100)
    result = train_model(df, "final_exam_score", "Random Forest Regressor")
    pipeline = result["pipeline"]
    feature_cols = result["feature_columns"]

    schema = build_feature_schema(df, feature_cols)
    # Leave study_hours completely blank and attendance as None
    raw_inputs = {
        "study_hours": "",
        "attendance": None,
        "department": "CS",
        "gender": "Female",
    }

    pred_res = predict_single(pipeline, feature_cols, raw_inputs, schema)
    assert pred_res["prediction"] is not None
    assert pred_res["had_missing"] is True
    assert "Some input values were missing and were automatically handled" in pred_res["notice"]


def test_9_what_if_clearing_feature_imputes_cleanly():
    """TEST 9: What-If with missing feature -> clearing inputs does not crash."""
    df = _synthetic_dataset(100)
    result = train_model(df, "final_exam_score", "Random Forest Regressor")
    pipeline = result["pipeline"]
    feature_cols = result["feature_columns"]
    schema = build_feature_schema(df, feature_cols)

    # All fields cleared / sent as empty strings
    cleared_inputs = {col: "" for col in feature_cols}
    pred_res = predict_single(pipeline, feature_cols, cleared_inputs, schema)
    assert pred_res["prediction"] is not None
    assert np.isfinite(float(pred_res["prediction"]))
    assert pred_res["had_missing"] is True


def test_10_dataset_with_no_missing_values_unchanged():
    """TEST 10: Dataset with no missing values -> behaves exactly as expected."""
    df = _synthetic_dataset(100)
    analysis = missing_analysis(df)
    assert analysis["total_missing_cells"] == 0
    assert analysis["overall_missing_pct"] == 0.0
    assert analysis["quality_status"] == "Valid"
    assert analysis["affected_columns_count"] == 0
    assert len(analysis["columns_100_pct_missing"]) == 0

    csv_report = generate_cleaning_report_csv(df)
    assert "Column,Data Type,Original Missing Count,Missing %,Cleaning Method,Final Status" in csv_report
    assert "Clean" in csv_report


def test_sentinel_markers_normalization():
    """Test normalization of common missing markers ('NA', 'null', '?', '-', 'missing')."""
    raw_csv = (
        "study_hours,attendance,department,final_exam_score\n"
        "5.0,85.0,CS,80.0\n"
        "?,90.0,Math,75.0\n"
        "6.5,-,null,92.0\n"
        "missing,95.0,N/A,88.0\n"
    ).encode("utf-8")

    parsed = parse_tabular_bytes(raw_csv, "csv")
    assert parsed["study_hours"].isna().sum() == 2
    assert parsed["attendance"].isna().sum() == 1
    assert parsed["department"].isna().sum() == 2
    # Ensure numeric columns are properly coerced
    assert pd.api.types.is_numeric_dtype(parsed["study_hours"])
    assert pd.api.types.is_numeric_dtype(parsed["attendance"])


def test_missing_analysis_and_cleaning_report():
    """Test missing_analysis outputs real statistics and explanations."""
    df = _synthetic_dataset(100)
    df.loc[0:9, "study_hours"] = np.nan
    df.loc[0:19, "department"] = np.nan

    analysis = missing_analysis(df)
    assert analysis["total_missing_cells"] == 30
    assert analysis["affected_columns_count"] == 2
    assert analysis["quality_status"] == "Needs Cleaning"
    assert "30 missing values detected" in analysis["dynamic_explanation"]

    csv_text = generate_cleaning_report_csv(df)
    assert "study_hours,Numeric,10,10.00%,Median imputation,Imputed" in csv_text
    assert "department,Categorical,20,20.00%,Most frequent,Imputed" in csv_text


def test_student_performance_missing_10000_dataset_end_to_end():
    """End-to-end test on the user's actual student_performance_missing_10000.csv dataset."""
    filepath = r"C:\Users\Mohamed shahjan s\Downloads\student_performance_missing_10000.csv"
    if not os.path.exists(filepath):
        pytest.skip(f"Test dataset not found at {filepath}")

    with open(filepath, "rb") as fh:
        file_bytes = fh.read()

    # 1. Upload
    res = client.post(
        "/api/dataset/upload",
        files={"file": ("student_performance_missing_10000.csv", file_bytes, "text/csv")},
        headers={"X-Session-Id": "test_missing_sess"},
    )
    assert res.status_code == 200, res.text
    dataset_id = res.json()["dataset"]["dataset_id"]
    assert res.json()["dataset"]["num_rows"] == 10000
    assert res.json()["dataset"]["num_cols"] == 16

    # 2. Missing Analysis endpoint
    analysis_res = client.get(f"/api/analysis/{dataset_id}/missing", headers={"X-Session-Id": "test_missing_sess"})
    assert analysis_res.status_code == 200, analysis_res.text
    data = analysis_res.json()
    assert data["available"] is True
    analysis = data["analysis"]
    assert analysis["total_missing_cells"] == 12028
    assert analysis["affected_columns_count"] == 12
    assert analysis["highest_missing_column"]["column"] == "internet_usage"
    assert analysis["highest_missing_column"]["count"] == 2001
    assert analysis["highest_missing_column"]["percentage"] == 20.01
    assert analysis["overall_missing_pct"] == 7.52
    assert "12,028 missing values detected across 12 columns" in analysis["dynamic_explanation"]

    # 3. Cleaning report CSV download
    report_res = client.get(f"/api/analysis/{dataset_id}/cleaning-report", headers={"X-Session-Id": "test_missing_sess"})
    assert report_res.status_code == 200
    assert "Column,Data Type,Original Missing Count,Missing %,Cleaning Method,Final Status" in report_res.text
    assert "study_hours,Numeric,1003,10.03%,Median imputation,Imputed" in report_res.text
    assert "parental_support,Categorical,1004,10.04%,Most frequent,Imputed" in report_res.text

    # 4. Mandatory Gate: Attempt preflight and training before refinement (TEST 11 & 12)
    preflight_blocked = client.post(
        "/api/model/preflight",
        json={
            "dataset_id": dataset_id,
            "target_column": "final_exam_score",
            "problem_type": "regression",
            "model_choice": "Linear Regression",
        },
        headers={"X-Session-Id": "test_missing_sess"},
    )
    assert preflight_blocked.status_code == 200
    assert preflight_blocked.json()["validation"]["ok"] is False
    assert "Missing values must be handled before training" in preflight_blocked.json()["validation"]["reason"]

    # Direct train API bypass attempt (TEST 12)
    train_blocked = client.post(
        "/api/model/train",
        json={
            "dataset_id": dataset_id,
            "target_column": "final_exam_score",
            "problem_type": "regression",
            "model_choice": "Linear Regression",
        },
        headers={"X-Session-Id": "test_missing_sess"},
    )
    assert train_blocked.status_code == 400
    assert "Missing values must be handled before training" in train_blocked.json()["error"]["message"]

    # 5. User Refines Dataset (/api/dataset/{dataset_id}/refine)
    refine_res = client.post(f"/api/dataset/{dataset_id}/refine", headers={"X-Session-Id": "test_missing_sess"})
    assert refine_res.status_code == 200
    ref_data = refine_res.json()["refinement"]
    assert ref_data["refined"] is True
    assert ref_data["total_missing"] == 12028
    assert ref_data["columns_affected"] == 12
    assert ref_data["rows"] == 10000
    assert ref_data["cols"] == 16
    assert ref_data["ready"] is True

    # 6. Preflight succeeds after refinement
    preflight_res = client.post(
        "/api/model/preflight",
        json={
            "dataset_id": dataset_id,
            "target_column": "final_exam_score",
            "problem_type": "regression",
            "model_choice": "Linear Regression",
        },
        headers={"X-Session-Id": "test_missing_sess"},
    )
    assert preflight_res.status_code == 200
    assert preflight_res.json()["validation"]["ok"] is True

    # 7. Train succeeds after refinement
    import time
    train_res = client.post(
        "/api/model/train",
        json={
            "dataset_id": dataset_id,
            "target_column": "final_exam_score",
            "problem_type": "regression",
            "model_choice": "Linear Regression",
        },
        headers={"X-Session-Id": "test_missing_sess"},
    )
    assert train_res.status_code == 200
    job_id = train_res.json()["job_id"]

    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)

    assert job["status"] == "completed", job
    model_id = job["model_id"]

    # 6. Single Prediction with missing / blank inputs (Section 14 & 15)
    pred_res = client.post(
        "/api/predict/single",
        json={
            "model_id": model_id,
            "inputs": {
                "study_hours": "",  # user left blank
                "attendance": "",   # user left blank
                "sleep_hours": 7.0,
                "internet_usage": 3.0,
                "assignments_completed": 8,
                "previous_score": 75.0,
                "participation": 80.0,
                "extracurricular": "Yes",
                "parental_support": "Medium",
                "learning_style": "Visual",
                "gender": "Female",
                "department": "Engineering",
                "performance_level": "Good",
                "placement_status": "Eligible",
            }
        },
        headers={"X-Session-Id": "test_missing_sess"},
    )
    assert pred_res.status_code == 200, pred_res.text
    pred_data = pred_res.json()["result"]
    assert pred_data["prediction"] is not None
    assert pred_data["had_missing"] is True
    assert "Some input values were missing and were automatically handled" in pred_data["notice"]

