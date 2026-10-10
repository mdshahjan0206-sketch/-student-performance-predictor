"""Training configuration, background training jobs and result preparation.

Training runs on a worker thread and reports truthful stage transitions
(queued -> processing -> training -> evaluating -> completed/failed), so the
React interface stays responsive and never shows invented progress.

The ML itself is untouched: this module calls detect_problem_type,
recommend_model, validate_before_training and train_model from app.ml.training
and only turns their output into JSON.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from app.core.config import MAX_ROC_POINTS, MODEL_OPTIONS, SCATTER_SAMPLE_LIMIT
from app.core.errors import AppError, model_not_found
from app.ml.helpers import detect_target_candidates
from app.ml.prediction import build_feature_schema
from app.ml.training import (
    ModelTrainingError,
    detect_problem_type,
    recommend_model,
    train_model,
    validate_before_training,
)
from app.services.dataset_service import require_dataset
from app.services.store import (
    JobRecord,
    ModelRecord,
    model_path,
    new_id,
    store,
)
from app.utils.serialization import to_jsonable

logger = logging.getLogger("edupredict.model_service")

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="training")


# ----------------------------------------------------------------------
# Configuration the Train Model page needs before anything is submitted
# ----------------------------------------------------------------------

def training_config(dataset_id: str) -> dict:
    """Detected target candidates plus every column, so the page can offer
    automatic detection with a manual override."""
    record, df = require_dataset(dataset_id)
    candidates = detect_target_candidates(df)
    default_target = candidates[0] if candidates else record.column_names[0]
    return {
        "dataset_id": record.id,
        "filename": record.filename,
        "num_rows": record.num_rows,
        "num_cols": record.num_cols,
        "columns": record.column_names,
        "detected_targets": candidates,
        "auto_detected": bool(candidates),
        "default_target": default_target,
        "model_options": MODEL_OPTIONS,
        "analysis": target_analysis(dataset_id, default_target, "auto"),
    }


def target_analysis(dataset_id: str, target_column: str, problem_override: str = "auto") -> dict:
    """Auto problem type, the effective problem type after any override, the
    model list for that type, and the recommended model."""
    _, df = require_dataset(dataset_id)
    if target_column not in df.columns:
        raise AppError(
            "INVALID_TARGET",
            f"Target column '{target_column}' is not part of this dataset.",
        )

    auto_type = detect_problem_type(df, target_column)
    override = (problem_override or "auto").lower()
    if override in ("classification", "regression"):
        problem_type = override
    else:
        problem_type = auto_type

    target = df[target_column]

    from app.services.intelligence_service import enhanced_recommendation
    rec = enhanced_recommendation(dataset_id, target_column, problem_type)

    return {
        "target_column": target_column,
        "auto_problem_type": auto_type,
        "problem_type": problem_type,
        "overridden": problem_type != auto_type,
        "model_options": MODEL_OPTIONS[problem_type],
        "recommended_model": rec["recommended_model"],
        "distinct_values": int(target.nunique(dropna=True)),
        "missing_values": int(target.isnull().sum()),
        "target_dtype": str(target.dtype),
        # Enhanced recommendation fields (backward-compatible additions)
        "recommendation_reason": rec["recommendation_reason"],
        "recommendation_evidence": rec["recommendation_evidence"],
        "workflow_notes": rec["workflow_notes"],
    }


def preflight(dataset_id: str, target_column: str, problem_type: str, model_choice: str) -> dict:
    """Run pre-flight validation without training, so the page can warn
    before the user commits."""
    record, df = require_dataset(dataset_id)
    total_missing = sum(record.missing_summary.values()) if isinstance(record.missing_summary, dict) else 0
    if not record.refined and total_missing > 0:
        return {
            "ok": False,
            "reason": "Missing values must be handled before training.",
            "recommended_models": [],
            "warnings": ["Unresolved missing data: refinement required."],
        }
    result = validate_before_training(df, target_column, problem_type, model_choice, is_refined=record.refined)
    return {
        "ok": result["ok"],
        "reason": result["reason"],
        "recommended_models": result["recommended_models"],
        "warnings": result["warnings"],
    }


# ----------------------------------------------------------------------
# Background training
# ----------------------------------------------------------------------

def submit_training(dataset_id: str, target_column: str, problem_type: str,
                    model_choice: str, session_id: str) -> JobRecord:
    """Validate the request shape, queue the job, and return immediately."""
    record, df = require_dataset(dataset_id)
    total_missing = sum(record.missing_summary.values()) if isinstance(record.missing_summary, dict) else 0
    if not record.refined and total_missing > 0:
        raise AppError("UNRESOLVED_MISSING_DATA", "Missing values must be handled before training.", status_code=400)

    if target_column not in df.columns:
        raise AppError("INVALID_TARGET", f"Target column '{target_column}' is not part of this dataset.")

    if problem_type not in MODEL_OPTIONS:
        raise AppError("INVALID_PROBLEM_TYPE", "Problem type must be classification or regression.")

    if model_choice not in MODEL_OPTIONS[problem_type]:
        raise AppError(
            "INVALID_MODEL",
            f"'{model_choice}' is not a supported {problem_type} model. "
            f"Choose one of: {', '.join(MODEL_OPTIONS[problem_type])}",
        )

    job = store.add_job(JobRecord(id=new_id("job"), kind="training", dataset_id=dataset_id))
    store.touch(session_id, f"Queued training for {target_column}", stage="train")
    _executor.submit(_run_training, job.id, dataset_id, target_column, problem_type, model_choice, session_id)
    return job


def _run_training(job_id, dataset_id, target_column, problem_type, model_choice, session_id):
    try:
        store.update_job(job_id, status="processing")
        record, df = require_dataset(dataset_id)

        total_missing = sum(record.missing_summary.values()) if isinstance(record.missing_summary, dict) else 0
        if not record.refined and total_missing > 0:
            store.update_job(
                job_id,
                status="failed",
                error_code="UNRESOLVED_MISSING_DATA",
                error_message="Missing values must be handled before training.",
                recommended_models=[],
            )
            return

        validation = validate_before_training(df, target_column, problem_type, model_choice)
        if not validation["ok"]:
            store.update_job(
                job_id,
                status="failed",
                error_code="MODEL_NOT_SUITABLE",
                error_message=validation["reason"],
                recommended_models=validation["recommended_models"],
            )
            return

        store.update_job(job_id, warnings=validation["warnings"])

        def progress(stage):
            store.update_job(job_id, status=stage)

        result = train_model(df, target_column, model_choice, problem_type, progress=progress)

        model_id = new_id("mdl")
        pipeline = result["pipeline"]
        estimator = pipeline.named_steps["model"]

        model_record = ModelRecord(
            id=model_id,
            dataset_id=dataset_id,
            dataset_filename=record.filename,
            target_column=result["target_column"],
            problem_type=result["problem_type"],
            model_choice=model_choice,
            estimator_name=estimator.__class__.__name__,
            feature_columns=[str(c) for c in result["feature_columns"]],
            training_time=float(result["training_time"]),
            n_train=int(result["n_train"]),
            n_test=int(result["n_test"]),
            metrics={k: to_jsonable(v) for k, v in result["metrics"].items()},
            charts=_build_charts(result),
            warnings=validation["warnings"],
            storage_path=model_path(model_id),
            target_missing_excluded=int(result.get("target_missing_excluded", 0)),
        )

        store.add_model(model_record, pipeline, {
            "feature_columns": model_record.feature_columns,
            "target_column": model_record.target_column,
            "problem_type": model_record.problem_type,
            "X_test": result["X_test"],
            "y_test": result["y_test"],
        })
        store.set_active_model(session_id, model_id)
        store.touch(
            session_id,
            f"Trained {model_choice} on {target_column}",
            stage="predict",
        )
        store.update_job(job_id, status="completed", model_id=model_id)
        # Real, measured training duration — already computed by app.ml.training.
        store.log_performance(
            "Training",
            model_record.training_time,
            meta={"model_id": model_id, "model_choice": model_choice, "dataset_id": dataset_id},
        )

    except ModelTrainingError:
        # Already logged with full traceback inside train_model().
        store.update_job(
            job_id,
            status="failed",
            error_code="TRAINING_FAILED",
            error_message=(
                f"The selected model ('{model_choice}') could not be trained on this dataset. "
                "Check the target column and feature values, or try another algorithm."
            ),
        )
    except AppError as exc:
        store.update_job(job_id, status="failed", error_code=exc.code, error_message=exc.message)
    except Exception:
        logger.exception("Unexpected training failure job=%s model=%r target=%r",
                         job_id, model_choice, target_column)
        store.update_job(
            job_id,
            status="failed",
            error_code="TRAINING_FAILED",
            error_message=(
                "Training stopped unexpectedly. Review the dataset or choose another algorithm."
            ),
        )


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

def _build_charts(result: dict) -> dict:
    """Turn the training result into chart-ready arrays.

    The conditions match the original exactly: a confusion matrix only for
    classification, a ROC curve only for binary classification with
    probability scores, a residual plot only for regression, and feature
    importance only when the estimator exposes feature_importances_.
    """
    charts = {}
    problem_type = result["problem_type"]

    if problem_type == "classification":
        matrix = result["confusion_matrix"]
        charts["confusion_matrix"] = {
            "labels": [to_jsonable(v) for v in result["confusion_matrix_labels"]],
            "matrix": [[int(v) for v in row] for row in matrix],
        }

        roc = result.get("roc_curve")
        if roc is not None:
            fpr = np.asarray(roc["fpr"], dtype="float64")
            tpr = np.asarray(roc["tpr"], dtype="float64")
            if fpr.size > MAX_ROC_POINTS:
                # Keep endpoints; thin the middle evenly so the curve shape
                # is preserved without shipping one point per test row.
                idx = np.unique(
                    np.linspace(0, fpr.size - 1, MAX_ROC_POINTS).astype(int)
                )
                fpr, tpr = fpr[idx], tpr[idx]
            charts["roc_curve"] = {
                "fpr": [float(v) for v in fpr],
                "tpr": [float(v) for v in tpr],
                "auc": float(roc["auc"]),
                "positive_class": to_jsonable(roc["positive_class"]),
            }
    else:
        predicted = pd.Series(np.asarray(result["y_pred"], dtype="float64"))
        residuals = pd.Series(np.asarray(result["residuals"], dtype="float64"))
        frame = pd.DataFrame({"predicted": predicted.to_numpy(), "residual": residuals.to_numpy()})
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        sampled = frame
        was_sampled = False
        if len(frame) > SCATTER_SAMPLE_LIMIT:
            sampled = frame.sample(n=SCATTER_SAMPLE_LIMIT, random_state=42).sort_index()
            was_sampled = True
        charts["residuals"] = {
            "predicted": [float(v) for v in sampled["predicted"].to_numpy()],
            "residual": [float(v) for v in sampled["residual"].to_numpy()],
            "sampled": was_sampled,
            "plotted_points": int(len(sampled)),
            "total_points": int(len(frame)),
        }

    importance = result.get("feature_importance")
    names = result.get("feature_names")
    if importance is not None and names is not None:
        frame = pd.DataFrame({"feature": names, "importance": np.asarray(importance, dtype="float64")})
        frame = frame.sort_values("importance", ascending=False)
        charts["feature_importance"] = {
            "features": [str(f) for f in frame["feature"]],
            "importance": [float(v) for v in frame["importance"]],
        }

    return charts


def require_model(model_id: str) -> ModelRecord:
    record = store.get_model(model_id)
    if record is None:
        raise model_not_found(model_id)
    return record


def model_details(model_id: str) -> dict:
    record = require_model(model_id)
    body = record.to_info()
    body["charts"] = record.charts
    return body


def model_metrics(model_id: str) -> dict:
    record = require_model(model_id)
    return {
        "model_id": record.id,
        "problem_type": record.problem_type,
        "metrics": record.metrics,
        "n_train": record.n_train,
        "n_test": record.n_test,
        "training_time": record.training_time,
    }


def reference_frame(record: ModelRecord):
    """Data used to build input fields and validate batch uploads.

    Prefers the live dataset the model was trained on, so column ranges stay
    in step with the active data; falls back to the test split stored with the
    model, then to an empty frame with the right columns. This mirrors the
    original Prediction page's reference_df selection, with the dataset
    identity check done by id rather than filename.
    """
    df = store.get_frame(record.dataset_id)
    if df is not None:
        return df
    extras = store.get_model_extras(record.id)
    x_test = extras.get("X_test")
    if x_test is not None:
        return x_test
    return pd.DataFrame(columns=record.feature_columns)


def feature_schema(model_id: str) -> dict:
    record = require_model(model_id)
    reference = reference_frame(record)
    return {
        "model_id": record.id,
        "target_column": record.target_column,
        "problem_type": record.problem_type,
        "feature_columns": record.feature_columns,
        "fields": [
            {k: to_jsonable(v) for k, v in field.items()}
            for field in build_feature_schema(reference, record.feature_columns)
        ],
    }


def training_history(dataset_id: str) -> list:
    return [record.to_info() for record in store.models_for_dataset(dataset_id)]
