"""Tests for the student-level endpoints added in Phase A: at-risk
detection (18), individual student profile (19), prediction explorer (20),
comparison (34) and confidence summary (37/52).

Self-contained: generates its own small CSVs rather than depending on the
external DATA_DIR fixtures test_api.py uses, so it runs anywhere.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def _classification_csv(tmp_path):
    rng = np.random.default_rng(7)
    n = 200
    attendance = rng.uniform(40, 100, n)
    study_hours = rng.uniform(0, 10, n)
    previous_score = rng.uniform(30, 100, n)
    score = 0.4 * attendance + 3 * study_hours + 0.3 * previous_score + rng.normal(0, 5, n)
    pass_fail = np.where(score >= score.mean(), "Pass", "Fail")
    df = pd.DataFrame({
        "student_name": [f"Student {i}" for i in range(n)],
        "attendance": attendance.round(1),
        "study_hours": study_hours.round(1),
        "previous_score": previous_score.round(1),
        "pass_fail": pass_fail,
    })
    path = tmp_path / "students_clf.csv"
    df.to_csv(path, index=False)
    return path


def _regression_csv(tmp_path):
    rng = np.random.default_rng(11)
    n = 150
    attendance = rng.uniform(40, 100, n)
    study_hours = rng.uniform(0, 10, n)
    exam_score = (0.5 * attendance + 4 * study_hours + rng.normal(0, 6, n)).clip(0, 100)
    df = pd.DataFrame({
        "attendance": attendance.round(1),
        "study_hours": study_hours.round(1),
        "exam_score": exam_score.round(1),
    })
    path = tmp_path / "students_reg.csv"
    df.to_csv(path, index=False)
    return path


def _upload(path, session):
    with open(path, "rb") as fh:
        return client.post(
            "/api/dataset/upload",
            files={"file": (os.path.basename(path), fh.read(), "text/csv")},
            headers={"X-Session-Id": session},
        )


def _train_and_wait(dataset_id, target, problem_type, model_choice, session, timeout=60):
    started = client.post("/api/model/train", json={
        "dataset_id": dataset_id, "target_column": target,
        "problem_type": problem_type, "model_choice": model_choice,
    }, headers={"X-Session-Id": session})
    assert started.status_code == 200, started.text
    job_id = started.json()["job_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            assert job["status"] == "completed", job
            return job
        time.sleep(0.05)
    raise AssertionError("training job did not finish")


def _batch(model_id, path, session):
    with open(path, "rb") as fh:
        return client.post(
            "/api/predict/batch",
            data={"model_id": model_id},
            files={"file": (os.path.basename(path), fh.read(), "text/csv")},
            headers={"X-Session-Id": session},
        )


@pytest.fixture(scope="module")
def classification_prediction(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("clf")
    path = _classification_csv(tmp_path)
    session = "students-clf"
    ds = _upload(path, session).json()["dataset"]
    job = _train_and_wait(ds["dataset_id"], "pass_fail", "classification", "Random Forest", session)
    res = _batch(job["model_id"], path, session)
    assert res.status_code == 200, res.text
    return res.json()["batch"]["prediction_id"]


@pytest.fixture(scope="module")
def regression_prediction(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("reg")
    path = _regression_csv(tmp_path)
    session = "students-reg"
    ds = _upload(path, session).json()["dataset"]
    job = _train_and_wait(ds["dataset_id"], "exam_score", "regression", "Random Forest Regressor", session)
    res = _batch(job["model_id"], path, session)
    assert res.status_code == 200, res.text
    return res.json()["batch"]["prediction_id"]


# --- capability ----------------------------------------------------------

def test_classification_capability_supports_risk_and_finds_identifier(classification_prediction):
    res = client.get(f"/api/students/{classification_prediction}/capability")
    assert res.status_code == 200, res.text
    cap = res.json()["capability"]
    assert cap["risk"]["supported"] is True
    assert cap["identifier_column"] == "student_name"
    assert cap["has_confidence"] is True
    assert set(cap["prediction_labels"]) <= {"Fail", "Pass"}


def test_regression_capability_supports_percentile_risk(regression_prediction):
    res = client.get(f"/api/students/{regression_prediction}/capability")
    assert res.status_code == 200, res.text
    cap = res.json()["capability"]
    assert cap["risk"]["supported"] is True
    assert "percentile" in cap["risk"]["basis"]


# --- explorer / at-risk ---------------------------------------------------

def test_explorer_pagination_and_total_rows(classification_prediction):
    res = client.get(f"/api/students/{classification_prediction}?page=1&page_size=25")
    body = res.json()
    assert res.status_code == 200, res.text
    assert body["total_rows"] == 200
    assert len(body["rows"]) == 25
    assert body["rows"][0]["_row_index"] == 0


def test_explorer_risk_filter_only_returns_matching_rows(classification_prediction):
    res = client.get(f"/api/students/{classification_prediction}?risk=high&page_size=200")
    body = res.json()
    assert res.status_code == 200
    assert body["total_rows"] > 0
    assert all(row["_risk"] == "high" for row in body["rows"])


def test_explorer_search_matches_identifier_column(classification_prediction):
    res = client.get(f"/api/students/{classification_prediction}?search=Student 3&page_size=200")
    body = res.json()
    assert res.status_code == 200
    assert body["total_rows"] > 0
    assert all("Student 3" in str(row["student_name"]) for row in body["rows"])


def test_explorer_sort_by_confidence(classification_prediction):
    res = client.get(f"/api/students/{classification_prediction}?sort_by=Confidence&sort_dir=desc&page_size=200")
    body = res.json()
    confidences = [row["Confidence"] for row in body["rows"]]
    assert confidences == sorted(confidences, reverse=True)


def test_explorer_confidence_range_filter(classification_prediction):
    res = client.get(f"/api/students/{classification_prediction}?confidence_min=0.9&page_size=200")
    body = res.json()
    assert all(row["Confidence"] >= 0.9 for row in body["rows"])


def test_explorer_download_respects_filters(classification_prediction):
    res = client.get(f"/api/students/{classification_prediction}/download?risk=high")
    assert res.status_code == 200
    lines = res.content.decode().strip().splitlines()
    assert "Risk" in lines[0]
    assert len(lines) - 1 > 0


# --- profile / compare -----------------------------------------------------

def test_student_profile_organizes_sections(classification_prediction):
    res = client.get(f"/api/students/{classification_prediction}/profile/0")
    assert res.status_code == 200, res.text
    profile = res.json()["profile"]
    assert profile["display_label"] == "Student 0"
    academic_names = {f["name"] for f in profile["sections"]["academic"]}
    assert {"attendance", "study_hours", "previous_score"} <= academic_names
    assert profile["prediction"]["value"] in ("Pass", "Fail")
    assert profile["risk"]["value"] in ("high", "medium", "low")


def test_student_profile_missing_row_is_404(classification_prediction):
    res = client.get(f"/api/students/{classification_prediction}/profile/999999")
    assert res.status_code == 404


def test_student_compare_requires_two_to_four(classification_prediction):
    res = client.post(f"/api/students/{classification_prediction}/compare", json={"row_indices": [0]})
    assert res.status_code == 400

    res = client.post(f"/api/students/{classification_prediction}/compare", json={"row_indices": [0, 1, 2]})
    assert res.status_code == 200
    assert len(res.json()["comparison"]["students"]) == 3


# --- confidence summary ----------------------------------------------------

def test_confidence_summary_totals_match_row_count(classification_prediction):
    res = client.get(f"/api/students/{classification_prediction}/confidence-summary")
    summary = res.json()["summary"]
    assert summary["available"] is True
    assert summary["total"] == 200
    assert summary["high"] + summary["medium"] + summary["low"] == 200
