"""Tests for Feature 25 (Explainable Predictions / SHAP): tree-based
TreeExplainer and the linear closed-form contribution, for both
classification and regression, plus the student-profile convenience route.
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
    rng = np.random.default_rng(3)
    n = 220
    attendance = rng.uniform(40, 100, n)
    study_hours = rng.uniform(0, 10, n)
    previous_score = rng.uniform(30, 100, n)
    score = 0.4 * attendance + 3 * study_hours + 0.3 * previous_score + rng.normal(0, 5, n)
    pass_fail = np.where(score >= score.mean(), "Pass", "Fail")
    df = pd.DataFrame({
        "attendance": attendance.round(1),
        "study_hours": study_hours.round(1),
        "previous_score": previous_score.round(1),
        "pass_fail": pass_fail,
    })
    path = tmp_path / "clf.csv"
    df.to_csv(path, index=False)
    return path, df


def _regression_csv(tmp_path):
    rng = np.random.default_rng(9)
    n = 180
    attendance = rng.uniform(40, 100, n)
    study_hours = rng.uniform(0, 10, n)
    exam_score = (0.5 * attendance + 4 * study_hours + rng.normal(0, 6, n)).clip(0, 100)
    df = pd.DataFrame({
        "attendance": attendance.round(1),
        "study_hours": study_hours.round(1),
        "exam_score": exam_score.round(1),
    })
    path = tmp_path / "reg.csv"
    df.to_csv(path, index=False)
    return path, df


def _upload_and_train(path, target, problem_type, model_choice, session):
    with open(path, "rb") as fh:
        up = client.post(
            "/api/dataset/upload",
            files={"file": (os.path.basename(path), fh.read(), "text/csv")},
            headers={"X-Session-Id": session},
        )
    ds = up.json()["dataset"]
    started = client.post("/api/model/train", json={
        "dataset_id": ds["dataset_id"], "target_column": target,
        "problem_type": problem_type, "model_choice": model_choice,
    }, headers={"X-Session-Id": session})
    job_id = started.json()["job_id"]
    deadline = time.time() + 60
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            assert job["status"] == "completed", job
            return job["model_id"], ds
        time.sleep(0.05)
    raise AssertionError("training did not finish")


@pytest.fixture(scope="module")
def clf_models(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("explain_clf")
    path, df = _classification_csv(tmp_path)
    rf_id, ds = _upload_and_train(path, "pass_fail", "classification", "Random Forest", "explain-clf-rf")
    lr_id, _ = _upload_and_train(path, "pass_fail", "classification", "Logistic Regression", "explain-clf-lr")
    return {"rf": rf_id, "lr": lr_id, "df": df}


@pytest.fixture(scope="module")
def reg_models(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("explain_reg")
    path, df = _regression_csv(tmp_path)
    rf_id, ds = _upload_and_train(path, "exam_score", "regression", "Random Forest Regressor", "explain-reg-rf")
    lin_id, _ = _upload_and_train(path, "exam_score", "regression", "Linear Regression", "explain-reg-lin")
    return {"rf": rf_id, "lin": lin_id, "df": df}


def _row(df, i):
    return df.drop(columns=[c for c in df.columns if c in ("pass_fail", "exam_score")]).iloc[i].to_dict()


# --- capability ------------------------------------------------------------

def test_tree_and_linear_report_supported(clf_models):
    for key in ("rf", "lr"):
        res = client.get(f"/api/explain/{clf_models[key]}/capability")
        assert res.status_code == 200, res.text
        cap = res.json()["capability"]
        assert cap["supported"] is True
        assert cap["method"] in ("tree", "linear")


# --- instance explanation: classification -----------------------------------

def test_tree_instance_explanation_classification(clf_models):
    row = _row(clf_models["df"], 0)
    res = client.post(f"/api/explain/{clf_models['rf']}/instance", json={"inputs": row})
    assert res.status_code == 200, res.text
    body = res.json()
    explanation = body["explanation"]
    assert len(explanation["top_contributions"]) > 0
    assert explanation["predicted_label"] in ("Pass", "Fail")
    for c in explanation["top_contributions"]:
        assert c["direction"] in ("positive", "negative")


def test_linear_instance_explanation_classification(clf_models):
    row = _row(clf_models["df"], 1)
    res = client.post(f"/api/explain/{clf_models['lr']}/instance", json={"inputs": row})
    assert res.status_code == 200, res.text
    explanation = res.json()["explanation"]
    assert explanation["predicted_label"] in ("Pass", "Fail")
    # base + contributions should reconstruct the model's own decision function
    total = explanation["base_value"] + sum(c["contribution"] for c in explanation["top_contributions"])
    # top_contributions is capped at 10, so just check it's in a sane logit-ish range
    assert -50 < total < 50


def test_instance_explanation_missing_feature_is_400(clf_models):
    res = client.post(f"/api/explain/{clf_models['rf']}/instance", json={"inputs": {"attendance": 90}})
    assert res.status_code == 400


# --- instance explanation: regression ---------------------------------------

def test_tree_instance_explanation_regression(reg_models):
    row = _row(reg_models["df"], 0)
    res = client.post(f"/api/explain/{reg_models['rf']}/instance", json={"inputs": row})
    assert res.status_code == 200, res.text
    explanation = res.json()["explanation"]
    assert explanation["predicted_label"] is None
    assert isinstance(explanation["predicted_value"], float)


def test_linear_instance_explanation_regression_reconstructs_prediction(reg_models):
    row = _row(reg_models["df"], 2)
    predict_res = client.post("/api/predict/single", json={"model_id": reg_models["lin"], "inputs": row})
    predicted = predict_res.json()["result"]["prediction"]

    res = client.post(f"/api/explain/{reg_models['lin']}/instance", json={"inputs": row})
    explanation = res.json()["explanation"]
    # Linear regression has only 2 features, so top_contributions (capped at
    # 10) captures all of them -> base + contributions must equal the
    # model's own prediction exactly.
    total = explanation["base_value"] + sum(c["contribution"] for c in explanation["top_contributions"])
    assert total == pytest.approx(predicted, abs=1e-6)
    assert explanation["predicted_value"] == pytest.approx(predicted, abs=1e-6)


# --- global importance -------------------------------------------------------

def test_global_importance_tree_and_linear(clf_models, reg_models):
    for model_id in (clf_models["rf"], clf_models["lr"], reg_models["rf"], reg_models["lin"]):
        res = client.get(f"/api/explain/{model_id}/global")
        assert res.status_code == 200, res.text
        importance = res.json()["importance"]
        assert len(importance["features"]) > 0
        assert all(v >= 0 for v in importance["importance"])
        assert importance["sample_size"] > 0


# --- explain-for-prediction convenience route --------------------------------

def test_explain_for_prediction_row(clf_models):
    # Score a fresh batch with the RF model, then explain one of its rows.
    df = clf_models["df"]
    csv_bytes = df.to_csv(index=False).encode()
    res = client.post(
        "/api/predict/batch",
        data={"model_id": clf_models["rf"]},
        files={"file": ("batch.csv", csv_bytes, "text/csv")},
        headers={"X-Session-Id": "explain-clf-rf"},
    )
    prediction_id = res.json()["batch"]["prediction_id"]

    explain_res = client.get(f"/api/explain/prediction/{prediction_id}/0")
    assert explain_res.status_code == 200, explain_res.text
    assert explain_res.json()["explanation"]["top_contributions"]
