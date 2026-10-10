"""Tests for the Phase 1 additions: dashboard, performance monitor and
prediction history. Self-contained — generates its own tiny in-memory
dataset rather than depending on the external DATA_DIR fixtures the original
suite uses, so it always runs.
"""

import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def _csv_bytes():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame(
        {
            "study_hours": rng.uniform(0, 10, n),
            "attendance": rng.uniform(50, 100, n),
            "gpa": rng.uniform(0, 4, n),
        }
    )
    df["pass_fail"] = np.where(df["gpa"] > 2.0, "Pass", "Fail")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode(), df


def _upload(session):
    csv_bytes, df = _csv_bytes()
    resp = client.post(
        "/api/dataset/upload",
        files={"file": ("students.csv", csv_bytes, "text/csv")},
        headers={"X-Session-Id": session},
    )
    assert resp.status_code == 200
    return resp.json()["dataset"]["dataset_id"], df


def _train(dataset_id, session):
    resp = client.post(
        "/api/model/train",
        json={
            "dataset_id": dataset_id,
            "target_column": "pass_fail",
            "problem_type": "classification",
            "model_choice": "Logistic Regression",
        },
        headers={"X-Session-Id": session},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    job = None
    for _ in range(80):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert job is not None and job["status"] == "completed", job
    return job["model_id"]


def test_dashboard_has_no_model_before_training():
    session = "dash-1"
    dataset_id, _ = _upload(session)
    resp = client.get(f"/api/dashboard/{dataset_id}", headers={"X-Session-Id": session})
    assert resp.status_code == 200
    body = resp.json()["dashboard"]
    assert body["total_records"] == 120
    assert body["model"] is None
    assert body["target_breakdown"] == {}


def test_dashboard_reflects_real_model_and_prediction_data():
    session = "dash-2"
    dataset_id, df = _upload(session)
    model_id = _train(dataset_id, session)

    resp = client.get(f"/api/dashboard/{dataset_id}", headers={"X-Session-Id": session})
    body = resp.json()["dashboard"]
    assert body["models_trained"] == 1
    assert body["model"]["model_id"] == model_id
    assert body["model"]["is_active"] is True
    # Target breakdown must come from the real column values, not an estimate.
    assert body["target_breakdown"]["kind"] == "classification"
    assert sum(body["target_breakdown"]["counts"]) == 120
    assert body["target_breakdown"]["pass_percentage"] + body["target_breakdown"]["fail_percentage"] == 100.0

    # No predictions run yet.
    assert body["prediction_runs"] == 0
    assert body["records_scored"] == 0

    # Run a batch prediction, then confirm the dashboard and history pick it up.
    buf = io.StringIO()
    df[["study_hours", "attendance", "gpa"]].to_csv(buf, index=False)
    batch = client.post(
        "/api/predict/batch",
        data={"model_id": model_id},
        files={"file": ("batch.csv", buf.getvalue().encode(), "text/csv")},
        headers={"X-Session-Id": session},
    )
    assert batch.status_code == 200

    resp = client.get(f"/api/dashboard/{dataset_id}", headers={"X-Session-Id": session})
    body = resp.json()["dashboard"]
    assert body["prediction_runs"] == 1
    assert body["records_scored"] == 120

    history = client.get(f"/api/predictions/history/{dataset_id}").json()["history"]
    assert len(history) == 1
    assert history[0]["row_count"] == 120
    assert history[0]["model_id"] == model_id


def test_performance_log_only_contains_real_measured_operations():
    session = "perf-1"
    before = client.get("/api/performance?limit=300").json()["entries"]
    dataset_id, _ = _upload(session)
    after = client.get("/api/performance?limit=300").json()["entries"]

    assert len(after) > len(before)
    # The newest entry should be the upload we just triggered, with a
    # non-negative, real duration — not a placeholder.
    newest = after[0]
    assert newest["operation"] == "Dataset Processing"
    assert newest["duration"] >= 0
    assert newest["status"] == "success"


def test_prediction_history_empty_for_new_dataset():
    session = "hist-1"
    dataset_id, _ = _upload(session)
    history = client.get(f"/api/predictions/history/{dataset_id}").json()["history"]
    assert history == []
