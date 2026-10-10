"""Single-record and batch prediction.

Ported from the original Prediction page's predict_single() / predict_batch()
/ create_single_inputs(). The prediction calls, the column ordering, the
confidence calculation and the output column names are unchanged.
"""

import logging

import numpy as np
import pandas as pd

from app.ml.helpers import coerce_batch_dtypes, get_feature_types

logger = logging.getLogger("edupredict.prediction")


class PredictionError(Exception):
    """Raised when a prediction cannot be produced. The underlying exception
    is logged server-side; callers surface a clean message only."""
    pass


def build_feature_schema(reference_df, feature_columns):
    """Describe one input field per trained feature column, so the frontend
    can generate the single-prediction form dynamically.

    Ranges, steps and default values reproduce the original number_input /
    selectbox / text_input configuration exactly:

      * numeric + integer -> min/max are the observed range widened by a
        margin of max((max - min) * 0.2, 1.0); the default value is the
        column mean (rounded for integers); integer step 1, float step 0.1
      * categorical -> the sorted distinct values of the column
      * anything else, or a column missing from the reference data -> free text

    Fields are returned in the model's own feature order.
    """
    feature_types = get_feature_types(reference_df, feature_columns)
    fields = []

    for col in feature_columns:
        dtype = feature_types[col]

        if dtype in ("numeric", "integer") and col in reference_df.columns and not reference_df[col].dropna().empty:
            data_min = float(reference_df[col].min())
            data_max = float(reference_df[col].max())
            data_mean = float(reference_df[col].mean())
            # Allow a modest margin beyond the observed range so users can
            # explore reasonable "what-if" values, while still guarding
            # against wildly out-of-distribution inputs.
            margin = max((data_max - data_min) * 0.2, 1.0)
            lo, hi = data_min - margin, data_max + margin

            if dtype == "integer":
                fields.append({
                    "name": col,
                    "kind": "integer",
                    "min": int(lo),
                    "max": int(hi),
                    "step": 1,
                    "default": int(round(data_mean)),
                    "observed_min": data_min,
                    "observed_max": data_max,
                })
            else:
                fields.append({
                    "name": col,
                    "kind": "numeric",
                    "min": lo,
                    "max": hi,
                    "step": 0.1,
                    "default": data_mean,
                    "observed_min": data_min,
                    "observed_max": data_max,
                })

        elif dtype == "categorical" and col in reference_df.columns:
            unique_vals = reference_df[col].dropna().unique()
            if len(unique_vals) > 0:
                options = [_py(v) for v in sorted(unique_vals, key=str)]
                fields.append({
                    "name": col,
                    "kind": "categorical",
                    "options": options,
                    "default": options[0],
                })
            else:
                fields.append({"name": col, "kind": "text", "default": ""})

        else:
            fields.append({"name": col, "kind": "text", "default": ""})

    return fields


def _py(value):
    """Convert a numpy/pandas scalar to a plain JSON-safe Python value."""
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _is_missing(val):
    if val is None:
        return True
    if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
        return True
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in ("", "na", "n/a", "null", "none", "-", "?", "missing", "nan"):
            return True
    return False


def _cast_inputs(raw_inputs, schema):
    """Cast the JSON values posted by the form back to the types the model
    was trained on. Blanks and missing sentinels are converted to np.nan so the
    trained SimpleImputer automatically fills them using learned rules."""
    cast = {}
    for field in schema:
        name = field["name"]
        value = raw_inputs.get(name) if raw_inputs is not None else None
        kind = field["kind"]

        if _is_missing(value):
            cast[name] = np.nan
        elif kind in ("integer", "numeric"):
            try:
                num = float(value)
                if np.isnan(num) or np.isinf(num):
                    cast[name] = np.nan
                elif kind == "integer":
                    cast[name] = int(round(num))
                else:
                    cast[name] = num
            except (ValueError, TypeError):
                cast[name] = np.nan
        else:
            cast[name] = str(value)
    return cast


def predict_single(pipeline, feature_columns, raw_inputs, schema):
    """Score one record. Mirrors the original single-prediction path:
    build a one-row DataFrame in the model's feature order, predict, and add
    confidence + predicted class when the estimator exposes predict_proba."""
    try:
        inputs = _cast_inputs(raw_inputs, schema)
        input_df = pd.DataFrame([inputs])[feature_columns]
        prediction = pipeline.predict(input_df)[0]

        result = {"prediction": _py(prediction)}

        if hasattr(pipeline, "predict_proba"):
            probas = pipeline.predict_proba(input_df)[0]
            max_proba = float(np.max(probas))
            predicted_class = pipeline.classes_[np.argmax(probas)]
            result["confidence"] = max_proba
            result["predicted_class"] = _py(predicted_class)

        # Detect whether any input values were missing and automatically handled
        had_missing = any(
            _is_missing(raw_inputs.get(f["name"])) if raw_inputs is not None else True
            for f in schema
        )
        result["had_missing"] = had_missing
        if had_missing:
            result["notice"] = (
                "Some input values were missing and were automatically handled using the "
                "preprocessing rules learned during training."
            )

        return result
    except Exception as exc:
        logger.exception("Single-record prediction failed for features=%r", feature_columns)
        raise PredictionError("Prediction") from exc


def predict_batch(pipeline, feature_columns, pred_df, reference_df):
    """Score every row of an uploaded file. Mirrors the original batch path:
    the uploaded frame is copied untouched, dtype-coerced features are passed
    to the model in training order, and Prediction / Confidence /
    Predicted Class columns are appended."""
    try:
        result_df = pred_df.copy()
        input_df = coerce_batch_dtypes(pred_df, feature_columns, reference_df)[feature_columns]

        predictions = pipeline.predict(input_df)
        result_df['Prediction'] = predictions

        if hasattr(pipeline, 'predict_proba'):
            probas = pipeline.predict_proba(input_df)
            max_probas = np.max(probas, axis=1)
            predicted_classes = pipeline.classes_[np.argmax(probas, axis=1)]
            result_df['Confidence'] = max_probas
            result_df['Predicted Class'] = predicted_classes

        return result_df
    except Exception as exc:
        logger.exception("Batch prediction failed for %d input rows", len(pred_df))
        raise PredictionError("Batch Prediction") from exc
