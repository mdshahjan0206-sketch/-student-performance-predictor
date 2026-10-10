"""Aggregated data for the Student Performance Dashboard (feature 1).

Every figure here is either read straight off an existing record (dataset
summary, model metrics) or computed directly from the active dataset's own
target column. Nothing is invented: if a figure cannot be honestly derived
(no model trained yet, target column not numeric or not binary, etc.) the
corresponding field is omitted rather than guessed at.
"""

import numpy as np
import pandas as pd

from app.core.config import NEGATIVE_LABELS, POSITIVE_LABELS, TARGET_COLUMN_KEYWORDS
from app.services.dataset_service import require_dataset
from app.services.store import store

# Kept as local aliases so the rest of this module is unchanged; the values
# themselves now live in app.core.config so app.ml.risk (at-risk detection,
# feature 18) can share the identical pass/fail semantics.
_POSITIVE_LABELS = POSITIVE_LABELS
_NEGATIVE_LABELS = NEGATIVE_LABELS


def _target_breakdown(df: pd.DataFrame, target_column: str, problem_type: str) -> dict:
    """Real statistics about the actual target column values. Classification
    with exactly two classes gets a pass/fail-style split ONLY when the class
    labels themselves are recognisably pass/fail-shaped; otherwise a plain
    class distribution is returned instead of guessing which class is
    "passing"."""
    if target_column not in df.columns:
        return {}
    series = df[target_column].dropna()
    if series.empty:
        return {}

    if problem_type == "regression":
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return {}
        return {
            "kind": "regression",
            "average": float(numeric.mean()),
            "highest": float(numeric.max()),
            "lowest": float(numeric.min()),
        }

    counts = series.astype(str).str.strip().value_counts()
    total = int(counts.sum())
    distribution = {
        "kind": "classification",
        "labels": [str(i) for i in counts.index],
        "counts": [int(c) for c in counts.to_numpy()],
    }

    if len(counts) == 2:
        lowered = {str(k).strip().lower(): int(v) for k, v in counts.items()}
        pass_count = next((v for k, v in lowered.items() if k in _POSITIVE_LABELS), None)
        fail_count = next((v for k, v in lowered.items() if k in _NEGATIVE_LABELS), None)
        if pass_count is not None and fail_count is not None and total:
            distribution["pass_percentage"] = round(100 * pass_count / total, 1)
            distribution["fail_percentage"] = round(100 * fail_count / total, 1)

    return distribution


def dashboard_summary(dataset_id: str, session_id: str) -> dict:
    record, df = require_dataset(dataset_id)

    session = store.session(session_id)
    models = store.models_for_dataset(dataset_id)
    active_model = None
    if session.active_model_id:
        active_model = next((m for m in models if m.id == session.active_model_id), None)
    latest_model = active_model or (models[0] if models else None)

    predictions = store.predictions_for_dataset(dataset_id)
    latest_prediction_count = sum(p.row_count for p in predictions)

    total_missing = int(sum(record.missing_summary.values())) if record.missing_summary else 0

    from app.ml.helpers import get_numeric_columns, get_text_category_columns
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_text_category_columns(df)

    at_risk_count = None
    if predictions:
        for p in predictions:
            if p.kind == "batch":
                try:
                    from app.services import student_service
                    ctx = student_service._context(p.id)
                    if ctx.get("risk_series") is not None:
                        rs = ctx["risk_series"]
                        at_risk_count = int((rs == "high").sum())
                        break
                except Exception:
                    pass

    body = {
        "dataset_id": dataset_id,
        "filename": record.filename,
        "total_records": record.num_rows,
        "total_columns": record.num_cols,
        "missing_values": total_missing,
        "duplicate_records": record.duplicate_count,
        "models_trained": len(models),
        "prediction_runs": len(predictions),
        "records_scored": latest_prediction_count,
        "at_risk_count": at_risk_count,
        "model": None,
        "target_breakdown": {},
        # Enhanced dashboard fields
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "numeric_columns": numeric_cols[:20],
        "categorical_columns": categorical_cols[:20],
    }

    if latest_model is not None:
        body["model"] = {
            "model_id": latest_model.id,
            "model_choice": latest_model.model_choice,
            "target_column": latest_model.target_column,
            "problem_type": latest_model.problem_type,
            "metrics": latest_model.metrics,
            "is_active": active_model is not None,
            "feature_importance": latest_model.charts.get("feature_importance"),
        }
        body["target_breakdown"] = _target_breakdown(df, latest_model.target_column, latest_model.problem_type)
    elif record.detected_targets:
        # Even without a model, provide detected target info under detected_target_breakdown
        from app.ml.training import detect_problem_type
        target_col = record.detected_targets[0]
        if target_col in df.columns:
            ptype = detect_problem_type(df, target_col)
            body["detected_target_breakdown"] = _target_breakdown(df, target_col, ptype)
            body["detected_target"] = target_col
            body["detected_problem_type"] = ptype

    return body
