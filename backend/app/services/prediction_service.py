"""Single and batch prediction, result storage and export.

Batch results are held on the backend and paged to the client, so a
100,000-row prediction run never arrives in the browser as one payload. The
CSV download is generated from the same stored frame, which keeps the export
identical to what the table shows.
"""

import numpy as np
import pandas as pd

from app.core.config import MAX_HISTOGRAM_BINS
from app.core.errors import AppError, stale_model
from app.core.timing import track
from app.ml.helpers import validate_batch_prediction_data
from app.ml.prediction import PredictionError, build_feature_schema, predict_batch, predict_single
from app.services.model_service import reference_frame, require_model
from app.services.store import PredictionRecord, new_id, output_path, store
from app.utils.file_utils import parse_tabular_bytes, safe_display_name, validate_upload
from app.utils.serialization import frame_to_records, to_jsonable

_FAILURE_MESSAGES = {
    "Prediction": (
        "We couldn't complete this prediction. Check that the input values match the "
        "format the model was trained on."
    ),
    "Batch Prediction": (
        "We couldn't complete this batch prediction. Check the uploaded file for missing "
        "or invalid values in the required columns."
    ),
}


def require_usable_model(model_id: str, session_id: str):
    """Fetch a model and refuse it if it belongs to a dataset that is no
    longer active. This is the backend counterpart of the original rule that
    uploading a new dataset invalidates the trained model."""
    record = require_model(model_id)
    session = store.session(session_id)
    if session.active_dataset_id and session.active_dataset_id != record.dataset_id:
        raise stale_model()
    pipeline = store.get_pipeline(model_id)
    if pipeline is None:
        raise AppError(
            "MODEL_UNAVAILABLE",
            "The trained model could not be loaded. Train the model again to continue.",
            status_code=410,
        )
    return record, pipeline


def run_single(model_id: str, inputs: dict, session_id: str) -> dict:
    with track("Single Prediction", meta={"model_id": model_id}):
        return _run_single(model_id, inputs, session_id)


def _run_single(model_id: str, inputs: dict, session_id: str) -> dict:
    record, pipeline = require_usable_model(model_id, session_id)
    reference = reference_frame(record)
    schema = build_feature_schema(reference, record.feature_columns)

    try:
        result = predict_single(pipeline, record.feature_columns, inputs, schema)
    except PredictionError as exc:
        raise AppError("PREDICTION_FAILED", _FAILURE_MESSAGES.get(str(exc), _FAILURE_MESSAGES["Prediction"])) from exc

    store.touch(session_id, f"Predicted {record.target_column} for one student", stage="predict")
    return {
        "prediction_id": new_id("pred"),
        "model_id": record.id,
        "problem_type": record.problem_type,
        "target_column": record.target_column,
        "prediction": result["prediction"],
        "confidence": result.get("confidence"),
        "predicted_class": result.get("predicted_class"),
        "had_missing": result.get("had_missing", False),
        "notice": result.get("notice"),
        "inputs": {k: to_jsonable(v) for k, v in inputs.items()},
    }


def run_batch(model_id: str, file_bytes: bytes, filename: str, session_id: str) -> dict:
    with track("Batch Prediction", meta={"model_id": model_id}):
        return _run_batch(model_id, file_bytes, filename, session_id)


def _run_batch(model_id: str, file_bytes: bytes, filename: str, session_id: str) -> dict:
    record, pipeline = require_usable_model(model_id, session_id)

    display_name = safe_display_name(filename)
    extension = validate_upload(display_name, len(file_bytes))
    pred_df = parse_tabular_bytes(file_bytes, extension)

    reference = reference_frame(record)
    validation = validate_batch_prediction_data(pred_df, record.feature_columns, reference)

    if not validation["valid"]:
        # Missing feature columns block prediction, exactly as before.
        raise AppError(
            "BATCH_VALIDATION_FAILED",
            validation["errors"][0],
            details={"errors": validation["errors"], "warnings": validation["warnings"]},
        )

    try:
        result_df = predict_batch(pipeline, record.feature_columns, pred_df, reference)
    except PredictionError as exc:
        raise AppError("PREDICTION_FAILED", _FAILURE_MESSAGES["Batch Prediction"]) from exc

    prediction_id = new_id("pred")
    prediction = PredictionRecord(
        id=prediction_id,
        model_id=record.id,
        dataset_id=record.dataset_id,
        kind="batch",
        row_count=int(len(result_df)),
        columns=[str(c) for c in result_df.columns],
        storage_path=output_path(prediction_id),
        has_confidence="Confidence" in result_df.columns,
        problem_type=record.problem_type,
        warnings=validation["warnings"],
        distribution=_distribution(result_df, record.problem_type),
    )
    store.add_prediction(prediction, result_df)
    store.touch(session_id, f"Scored {len(result_df):,} rows from {display_name}", stage="export")

    return {
        "prediction_id": prediction_id,
        "model_id": record.id,
        "source_filename": display_name,
        "row_count": prediction.row_count,
        "columns": prediction.columns,
        "has_confidence": prediction.has_confidence,
        "problem_type": prediction.problem_type,
        "warnings": prediction.warnings,
        "distribution": prediction.distribution,
    }


def _distribution(result_df: pd.DataFrame, problem_type: str) -> dict:
    """Prediction distribution: counts per class for classification, binned
    predicted values for regression (the original used 20 bins)."""
    series = result_df["Prediction"].dropna()
    if series.empty:
        return {"kind": "empty"}

    if problem_type == "classification":
        counts = series.astype(str).value_counts()
        return {
            "kind": "classification",
            "labels": [str(i) for i in counts.index],
            "counts": [int(c) for c in counts.to_numpy()],
        }

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"kind": "empty"}
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return {"kind": "regression", "bins": [f"{lo:g}"], "counts": [int(values.size)]}
    counts, edges = np.histogram(values, bins=min(20, MAX_HISTOGRAM_BINS))
    return {
        "kind": "regression",
        "bins": [f"{edges[i]:.3g} – {edges[i + 1]:.3g}" for i in range(len(counts))],
        "counts": [int(c) for c in counts],
    }


def prediction_page(prediction_id: str, page: int, page_size: int) -> dict:
    record = store.get_prediction(prediction_id)
    if record is None:
        raise AppError(
            "PREDICTION_NOT_FOUND",
            "Those prediction results are no longer available. Run the batch prediction again.",
            status_code=404,
        )
    df = store.get_prediction_frame(prediction_id)
    if df is None:
        raise AppError(
            "PREDICTION_NOT_FOUND",
            "Those prediction results are no longer available. Run the batch prediction again.",
            status_code=404,
        )

    page_size = max(1, min(page_size, 200))
    total_pages = max(1, -(-record.row_count // page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    window = df.iloc[start:start + page_size]

    return {
        "prediction_id": record.id,
        "model_id": record.model_id,
        "columns": record.columns,
        "rows": frame_to_records(window),
        "page": page,
        "page_size": page_size,
        "total_rows": record.row_count,
        "total_pages": total_pages,
        "has_confidence": record.has_confidence,
        "problem_type": record.problem_type,
        "warnings": record.warnings,
        "distribution": record.distribution,
    }


def prediction_history(dataset_id: str) -> list:
    """Lightweight metadata for every batch prediction run against this
    dataset — feature 7. Reuses PredictionRecord fields already stored, so
    nothing is duplicated: the full result frame stays where it already
    lives, on disk under outputs/, and is only loaded on demand."""
    records = store.predictions_for_dataset(dataset_id)
    history = []
    for record in records:
        entry = {
            "prediction_id": record.id,
            "model_id": record.model_id,
            "dataset_id": record.dataset_id,
            "kind": record.kind,
            "row_count": record.row_count,
            "problem_type": record.problem_type,
            "has_confidence": record.has_confidence,
            "created_at": record.created_at,
            "warnings": record.warnings,
        }
        model = store.get_model(record.model_id)
        if model is not None:
            entry["model_choice"] = model.model_choice
            entry["target_column"] = model.target_column
        if record.has_confidence:
            entry["distribution"] = record.distribution
        history.append(entry)
    return history


def prediction_csv(prediction_id: str) -> bytes:
    df = store.get_prediction_frame(prediction_id)
    if df is None:
        raise AppError(
            "PREDICTION_NOT_FOUND",
            "Those prediction results are no longer available. Run the batch prediction again.",
            status_code=404,
        )
    return df.to_csv(index=False).encode("utf-8")
