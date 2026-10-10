"""End-to-end API checks against the real ML pipeline (no mocks).

Run from the backend directory:  python -m pytest tests -q
DATA_DIR points at a folder of sample CSV/XLSX files.
"""

import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

DATA_DIR = os.environ.get("DATA_DIR", "/home/claude/testdata")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(DATA_DIR),
    reason="test data directory not found",
)
client = TestClient(app)


def upload(name, session="t1"):
    path = os.path.join(DATA_DIR, name)
    with open(path, "rb") as fh:
        return client.post(
            "/api/dataset/upload",
            files={"file": (name, fh.read(), "text/csv")},
            headers={"X-Session-Id": session},
        )


def train_and_wait(dataset_id, target, problem_type, model, session="t1", timeout=180):
    started = client.post("/api/model/train", json={
        "dataset_id": dataset_id, "target_column": target,
        "problem_type": problem_type, "model_choice": model,
    }, headers={"X-Session-Id": session})
    assert started.status_code == 200, started.text
    job_id = started.json()["job_id"]
    seen = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in seen:
            seen.append(job["status"])
        if job["status"] in ("completed", "failed"):
            return job, seen
        time.sleep(0.05)
    raise AssertionError("training job did not finish")


# --- dataset -----------------------------------------------------------

def test_csv_upload_and_metadata():
    res = upload("students_classification.csv")
    assert res.status_code == 200, res.text
    ds = res.json()["dataset"]
    assert ds["num_rows"] == 608 and ds["num_cols"] == 8
    assert ds["duplicate_count"] == 8
    assert "pass_fail" in ds["detected_targets"]
    assert ds["missing_summary"]["attendance"] == 25
    assert len(ds["column_profiles"]) == 8


def test_xlsx_upload():
    res = upload("students_regression.xlsx")
    assert res.status_code == 200, res.text
    assert res.json()["dataset"]["num_rows"] == 800


def test_rejects_unsupported_format():
    res = upload("notes.txt")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_rejects_empty_file():
    res = upload("empty.csv")
    assert res.status_code in (400, 422)
    assert res.json()["error"]["code"] in ("EMPTY_FILE", "EMPTY_DATASET")


def test_rejects_duplicate_columns():
    # pandas silently renames a repeated CSV header to "grade.1", so the guard
    # is exercised directly here — it exists for frames that really do carry
    # duplicate labels (the original project kept the same check).
    import pandas as pd

    from app.core.errors import AppError
    from app.utils.file_utils import validate_frame

    frame = pd.DataFrame([[1, 2]], columns=["grade", "grade"])
    try:
        validate_frame(frame)
    except AppError as exc:
        assert exc.code == "DUPLICATE_COLUMNS"
    else:
        raise AssertionError("duplicate columns were not rejected")


def test_no_target_detected():
    # height_cm / weight_kg match no performance keyword, so the page must
    # fall back to manual target selection.
    res = upload("no_target.csv")
    assert res.json()["dataset"]["detected_targets"] == []


def test_preview_pagination_and_search():
    ds = upload("students_classification.csv").json()["dataset"]
    page = client.get(f"/api/dataset/{ds['dataset_id']}/preview?page=2&page_size=10").json()
    assert page["page"] == 2 and len(page["rows"]) == 10
    assert page["total_pages"] == 61
    hit = client.get(f"/api/dataset/{ds['dataset_id']}/preview?search=Female&page_size=5").json()
    assert hit["total_rows"] < ds["num_rows"] and hit["total_rows"] > 0


def test_missing_dataset_is_clean_404():
    res = client.get("/api/dataset/ds_doesnotexist/summary")
    assert res.status_code == 404
    body = res.json()
    assert body["success"] is False and body["error"]["code"] == "DATASET_NOT_FOUND"
    assert "Traceback" not in res.text


# --- analysis ----------------------------------------------------------

def test_analysis_suite():
    ds = upload("students_classification.csv").json()["dataset"]
    did = ds["dataset_id"]

    overview = client.get(f"/api/analysis/{did}/overview").json()["overview"]
    assert overview["total_rows"] == 608
    assert overview["duplicate_rows"] == 8
    assert overview["total_missing"] == 41

    numeric_hist = client.get(f"/api/analysis/{did}/histogram?column=previous_score").json()
    assert numeric_hist["available"] and numeric_hist["chart"]["kind"] == "numeric"

    cat_hist = client.get(f"/api/analysis/{did}/histogram?column=gender").json()
    assert cat_hist["available"] and cat_hist["chart"]["kind"] == "categorical"

    box_ok = client.get(f"/api/analysis/{did}/boxplot?column=previous_score").json()
    assert box_ok["available"] and box_ok["chart"]["q1"] < box_ok["chart"]["q3"]

    box_bad = client.get(f"/api/analysis/{did}/boxplot?column=gender").json()
    assert box_bad["available"] is False and "numeric" in box_bad["message"]

    corr = client.get(f"/api/analysis/{did}/correlation").json()
    assert corr["available"] and len(corr["chart"]["columns"]) >= 2

    missing = client.get(f"/api/analysis/{did}/missing").json()
    assert missing["available"] and missing["chart"]["total_missing"] == 41

    scatter = client.get(f"/api/analysis/{did}/scatter?x=attendance&y=previous_score").json()
    assert scatter["available"] and len(scatter["chart"]["x"]) > 0

    same = client.get(f"/api/analysis/{did}/scatter?x=attendance&y=attendance").json()
    assert same["available"] is False


def test_correlation_needs_two_numeric_columns():
    ds = upload("single_class.csv").json()["dataset"]
    body = client.get(f"/api/analysis/{ds['dataset_id']}/correlation").json()
    assert body["available"] is False and "two numeric" in body["message"]


def test_missing_chart_absent_when_complete():
    ds = upload("students_regression.csv").json()["dataset"]
    body = client.get(f"/api/analysis/{ds['dataset_id']}/missing").json()
    assert body["available"] is False


# --- filtering is exploratory only -------------------------------------

def test_filters_do_not_mutate_the_stored_dataset():
    ds = upload("students_classification.csv").json()["dataset"]
    did = ds["dataset_id"]

    options = client.get(f"/api/dataset/{did}/filter-options").json()
    assert any(c["name"] == "attendance" for c in options["numeric"])

    filtered = client.post(f"/api/dataset/{did}/filter", json={
        "numeric": {"column": "previous_score", "min": 80, "max": 100},
        "categorical": {"column": "gender", "values": ["Female"]},
        "drop_missing": True,
        "limit": 10,
    }).json()
    assert filtered["matched_rows"] < filtered["total_rows"]
    assert len(filtered["rows"]) <= 10

    # the master dataset is unchanged
    again = client.get(f"/api/dataset/{did}/summary").json()["dataset"]
    assert again["num_rows"] == 608
    overview = client.get(f"/api/analysis/{did}/overview").json()["overview"]
    assert overview["total_rows"] == 608

    csv_res = client.post(f"/api/dataset/{did}/filter/download", json={
        "numeric": {"column": "previous_score", "min": 80, "max": 100},
    })
    assert csv_res.status_code == 200
    assert csv_res.headers["content-type"].startswith("text/csv")
    assert len(csv_res.content.decode().strip().splitlines()) > 1


# --- training ----------------------------------------------------------

def test_training_config_and_detection():
    ds = upload("students_classification.csv").json()["dataset"]
    cfg = client.get(f"/api/model/config/{ds['dataset_id']}").json()
    assert cfg["auto_detected"] and cfg["default_target"] == "pass_fail"
    assert cfg["analysis"]["auto_problem_type"] == "classification"
    assert cfg["analysis"]["recommended_model"] == "Random Forest"
    assert cfg["analysis"]["model_options"] == ["Logistic Regression", "Decision Tree", "Random Forest"]


def test_regression_detection_and_recommendation():
    ds = upload("students_regression.csv").json()["dataset"]
    body = client.post("/api/model/target-analysis", json={
        "dataset_id": ds["dataset_id"], "target_column": "exam_score", "problem_override": "auto",
    }).json()["analysis"]
    assert body["auto_problem_type"] == "regression"
    assert body["recommended_model"] == "Random Forest Regressor"


def test_manual_problem_type_override():
    ds = upload("students_regression.csv").json()["dataset"]
    body = client.post("/api/model/target-analysis", json={
        "dataset_id": ds["dataset_id"], "target_column": "exam_score",
        "problem_override": "classification",
    }).json()["analysis"]
    assert body["problem_type"] == "classification" and body["overridden"] is True


def test_classification_training_produces_metrics_and_charts():
    ds = upload("students_classification.csv").json()["dataset"]
    job, stages = train_and_wait(ds["dataset_id"], "pass_fail", "classification", "Random Forest")
    assert job["status"] == "completed", job
    assert "training" in stages and "evaluating" in stages

    model = client.get(f"/api/model/{job['model_id']}").json()["model"]
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= model["metrics"][key] <= 1.0
    assert model["estimator_name"] == "RandomForestClassifier"
    assert "confusion_matrix" in model["charts"]
    assert "roc_curve" in model["charts"]  # binary target
    assert model["charts"]["roc_curve"]["auc"] > 0.5
    assert "feature_importance" in model["charts"]
    assert "residuals" not in model["charts"]
    assert model["feature_columns"] == [
        "student_id", "hours_studied", "attendance", "gender",
        "internet_access", "parental_support", "previous_score",
    ]


def test_regression_training_produces_residuals_not_confusion_matrix():
    ds = upload("students_regression.csv").json()["dataset"]
    job, _ = train_and_wait(ds["dataset_id"], "exam_score", "regression", "Linear Regression")
    assert job["status"] == "completed"
    model = client.get(f"/api/model/{job['model_id']}").json()["model"]
    assert set(model["metrics"]) == {"mae", "mse", "rmse", "r2"}
    assert model["metrics"]["r2"] > 0.8
    assert "residuals" in model["charts"]
    assert "confusion_matrix" not in model["charts"]
    assert "feature_importance" not in model["charts"]  # LinearRegression has none


def test_logistic_regression_and_decision_tree_are_available():
    ds = upload("students_classification.csv").json()["dataset"]
    for model_name in ("Logistic Regression", "Decision Tree"):
        job, _ = train_and_wait(ds["dataset_id"], "pass_fail", "classification", model_name)
        assert job["status"] == "completed", (model_name, job)


def test_preflight_blocks_regression_on_text_target():
    ds = upload("students_classification.csv").json()["dataset"]
    check = client.post("/api/model/preflight", json={
        "dataset_id": ds["dataset_id"], "target_column": "pass_fail",
        "problem_type": "regression", "model_choice": "Linear Regression",
    }).json()["validation"]
    assert check["ok"] is False
    assert "cannot be used for a classification dataset" in check["reason"]
    assert check["recommended_models"] == ["Logistic Regression", "Decision Tree", "Random Forest"]


def test_training_job_fails_cleanly_on_single_class_target():
    ds = upload("single_class.csv").json()["dataset"]
    job, _ = train_and_wait(ds["dataset_id"], "grade", "classification", "Random Forest")
    assert job["status"] == "failed"
    assert job["error"]["code"] == "MODEL_NOT_SUITABLE"
    assert "at least 2" in job["error"]["message"]
    assert "Traceback" not in str(job)


def test_training_blocked_on_all_null_target():
    ds = upload("null_target.csv").json()["dataset"]
    job, _ = train_and_wait(ds["dataset_id"], "grade", "classification", "Random Forest")
    assert job["status"] == "failed"
    assert "only null values" in job["error"]["message"]


def test_training_blocked_on_one_row():
    ds = upload("one_row.csv").json()["dataset"]
    job, _ = train_and_wait(ds["dataset_id"], "grade", "classification", "Random Forest")
    assert job["status"] == "failed"
    assert "At least 2 rows" in job["error"]["message"]


def test_invalid_model_for_problem_type_is_rejected():
    ds = upload("students_regression.csv").json()["dataset"]
    res = client.post("/api/model/train", json={
        "dataset_id": ds["dataset_id"], "target_column": "exam_score",
        "problem_type": "regression", "model_choice": "Random Forest",
    })
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_MODEL"


def test_warnings_are_surfaced_not_hidden():
    ds = upload("students_classification.csv").json()["dataset"]
    job, _ = train_and_wait(ds["dataset_id"], "pass_fail", "classification", "Random Forest")
    assert any("imputed" in w for w in job.get("warnings", []))


# --- prediction --------------------------------------------------------

def _trained_regression(session="t2"):
    ds = upload("students_regression.csv", session=session).json()["dataset"]
    job, _ = train_and_wait(ds["dataset_id"], "exam_score", "regression",
                            "Random Forest Regressor", session=session)
    assert job["status"] == "completed"
    return ds, job["model_id"]


def test_feature_schema_is_built_from_trained_columns():
    _, model_id = _trained_regression()
    schema = client.get(f"/api/model/{model_id}/feature-schema").json()
    names = [f["name"] for f in schema["fields"]]
    assert names == ["hours_studied", "attendance", "sleep_hours", "tutoring", "school_type"]
    assert "exam_score" not in names  # target never appears as an input
    attendance = next(f for f in schema["fields"] if f["name"] == "attendance")
    assert attendance["kind"] == "integer" and attendance["step"] == 1
    assert attendance["min"] < attendance["observed_min"]
    tutoring = next(f for f in schema["fields"] if f["name"] == "tutoring")
    assert tutoring["kind"] == "categorical" and sorted(tutoring["options"]) == ["No", "Yes"]


def test_single_prediction_regression():
    _, model_id = _trained_regression()
    res = client.post("/api/predict/single", json={
        "model_id": model_id,
        "inputs": {"hours_studied": 8.0, "attendance": 95, "sleep_hours": 7.5,
                   "tutoring": "Yes", "school_type": "Private"},
    }, headers={"X-Session-Id": "t2"})
    assert res.status_code == 200, res.text
    result = res.json()["result"]
    assert isinstance(result["prediction"], float)
    assert result["confidence"] is None


def test_single_prediction_classification_has_confidence():
    ds = upload("students_classification.csv", session="t3").json()["dataset"]
    job, _ = train_and_wait(ds["dataset_id"], "pass_fail", "classification",
                            "Random Forest", session="t3")
    schema = client.get(f"/api/model/{job['model_id']}/feature-schema").json()
    inputs = {f["name"]: f["default"] for f in schema["fields"]}
    res = client.post("/api/predict/single", json={"model_id": job["model_id"], "inputs": inputs},
                      headers={"X-Session-Id": "t3"})
    result = res.json()["result"]
    assert result["predicted_class"] in ("Pass", "Fail")
    assert 0.0 < result["confidence"] <= 1.0


def test_single_prediction_rejects_missing_inputs():
    _, model_id = _trained_regression()
    res = client.post("/api/predict/single", json={
        "model_id": model_id, "inputs": {"hours_studied": 8.0},
    }, headers={"X-Session-Id": "t2"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "MISSING_INPUTS"


def _batch(model_id, filename, session="t2"):
    with open(os.path.join(DATA_DIR, filename), "rb") as fh:
        return client.post(
            "/api/predict/batch",
            data={"model_id": model_id},
            files={"file": (filename, fh.read(), "text/csv")},
            headers={"X-Session-Id": session},
        )


def test_batch_prediction_happy_path():
    _, model_id = _trained_regression()
    res = _batch(model_id, "batch_valid.csv")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["batch"]["row_count"] == 120
    assert "Prediction" in body["batch"]["columns"]
    assert body["results"]["total_pages"] == 5
    assert len(body["results"]["rows"]) == 25
    assert body["batch"]["distribution"]["kind"] == "regression"

    pid = body["batch"]["prediction_id"]
    page3 = client.get(f"/api/predictions/{pid}?page=3&page_size=25").json()
    assert page3["page"] == 3 and len(page3["rows"]) == 25

    dl = client.get(f"/api/predictions/{pid}/download")
    assert dl.status_code == 200
    lines = dl.content.decode().strip().splitlines()
    assert len(lines) == 121 and "Prediction" in lines[0]


def test_batch_missing_columns_blocks_prediction():
    _, model_id = _trained_regression()
    res = _batch(model_id, "batch_missing_cols.csv")
    assert res.status_code == 400
    err = res.json()["error"]
    assert err["code"] == "BATCH_VALIDATION_FAILED"
    assert "attendance" in err["message"]


def test_batch_extra_columns_warn_but_succeed():
    _, model_id = _trained_regression()
    res = _batch(model_id, "batch_extra_cols.csv")
    assert res.status_code == 200
    warnings = res.json()["batch"]["warnings"]
    assert any("Extra columns" in w for w in warnings)
    assert "extra_note" in res.json()["batch"]["columns"]  # original columns preserved


def test_batch_text_numeric_is_coerced_with_warning():
    _, model_id = _trained_regression()
    res = _batch(model_id, "batch_text_numeric.csv")
    assert res.status_code == 200
    assert any("converted to numeric" in w for w in res.json()["batch"]["warnings"])


def test_batch_classification_adds_confidence_and_class():
    ds = upload("students_classification.csv", session="t4").json()["dataset"]
    job, _ = train_and_wait(ds["dataset_id"], "pass_fail", "classification",
                            "Random Forest", session="t4")
    # score the training file itself (it has the feature columns plus the target)
    res = _batch(job["model_id"], "students_classification.csv", session="t4")
    assert res.status_code == 200, res.text
    batch = res.json()["batch"]
    assert batch["has_confidence"] is True
    assert "Confidence" in batch["columns"] and "Predicted Class" in batch["columns"]
    assert batch["distribution"]["kind"] == "classification"


# --- model / dataset invalidation --------------------------------------

def test_new_dataset_invalidates_the_previous_model():
    ds = upload("students_regression.csv", session="t5").json()["dataset"]
    job, _ = train_and_wait(ds["dataset_id"], "exam_score", "regression",
                            "Linear Regression", session="t5")
    model_id = job["model_id"]

    status = client.get("/api/session/status", headers={"X-Session-Id": "t5"}).json()
    assert status["active_model_id"] == model_id

    # uploading a different dataset must clear the active model
    upload("students_classification.csv", session="t5")
    status = client.get("/api/session/status", headers={"X-Session-Id": "t5"}).json()
    assert status["active_model_id"] == ""
    assert status["model"] is None

    # and the old model must refuse to predict
    res = client.post("/api/predict/single", json={
        "model_id": model_id,
        "inputs": {"hours_studied": 8.0, "attendance": 95, "sleep_hours": 7.5,
                   "tutoring": "Yes", "school_type": "Private"},
    }, headers={"X-Session-Id": "t5"})
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "STALE_MODEL"


def test_session_status_reports_stage_and_last_operation():
    upload("students_regression.csv", session="t6")
    status = client.get("/api/session/status", headers={"X-Session-Id": "t6"}).json()
    assert status["stage"] == "analyze"
    assert "Uploaded" in status["last_operation"]
    assert status["dataset"]["num_rows"] == 800


def test_health():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
