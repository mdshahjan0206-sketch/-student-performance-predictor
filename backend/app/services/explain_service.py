"""Feature 25 (Explainable Predictions / SHAP).

Two explanation methods, chosen by model type — never a new model, never a
change to preprocessing, training or prediction:

  * Tree-based models (Decision Tree, Random Forest, and their regressor
    variants) use shap.TreeExplainer against the model's own fitted
    estimator: the exact SHAP algorithm for trees, not an approximation.
  * Linear models (Logistic Regression, Linear Regression) use the closed
    form SHAP reduces to for a linear model — coefficient x (value - a
    typical value) — computed directly rather than through shap's linear
    explainer, since the two are mathematically the same for this case and
    the closed form has no background-data plumbing to get wrong.

Everything here is computed lazily, on request, for one student (or the
small background sample SHAP itself needs) — never for an entire batch
automatically. Global importance uses a capped sample of the training
dataset (or the model's held-out test split as a fallback), not the whole
table.
"""

import numpy as np
import pandas as pd

from app.core.errors import AppError
from app.ml.helpers import get_categorical_like_columns, get_numeric_columns
from app.services import model_service
from app.services.store import store

_TREE_MODELS = {"Decision Tree", "Random Forest", "Decision Tree Regressor", "Random Forest Regressor"}
_LINEAR_MODELS = {"Logistic Regression", "Linear Regression"}
_BACKGROUND_SIZE = 100
_TOP_N = 10


def _method_for(model_choice: str) -> str | None:
    if model_choice in _TREE_MODELS:
        return "tree"
    if model_choice in _LINEAR_MODELS:
        return "linear"
    return None


def capability(model_id: str) -> dict:
    record = model_service.require_model(model_id)
    method = _method_for(record.model_choice)
    if method == "tree":
        return {
            "supported": True,
            "method": "tree",
            "model_choice": record.model_choice,
            "description": "Exact SHAP values from the trained tree ensemble (TreeExplainer). No background data or approximation involved.",
        }
    if method == "linear":
        return {
            "supported": True,
            "method": "linear",
            "model_choice": record.model_choice,
            "description": "Exact per-feature contribution for a linear model: its coefficient multiplied by how far this student's value is from the typical value.",
        }
    return {
        "supported": False,
        "method": None,
        "model_choice": record.model_choice,
        "description": f"Explanations are not available for {record.model_choice}. Only the linear and tree-based models this app trains support them.",
    }


def _feature_frame(record):
    reference = model_service.reference_frame(record)
    if reference is None or len(reference) == 0:
        raise AppError(
            "NO_REFERENCE_DATA",
            "No reference data is available to build an explanation for this model.",
            status_code=400,
        )
    missing = [c for c in record.feature_columns if c not in reference.columns]
    if missing:
        raise AppError("MISSING_FEATURES", f"The reference data is missing columns: {', '.join(missing)}", status_code=400)
    return reference[record.feature_columns]


def _column_split(X: pd.DataFrame):
    return get_numeric_columns(X), get_categorical_like_columns(X)


def _expanded_names(pipeline, numeric_cols, categorical_cols) -> list[str]:
    cat_names: list[str] = []
    if categorical_cols and "cat" in pipeline.named_steps["preprocessor"].named_transformers_:
        cat_names = list(
            pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_cols)
        )
    return list(numeric_cols or []) + cat_names


def _background(record, pipeline):
    X = _feature_frame(record)
    numeric_cols, categorical_cols = _column_split(X)
    sample = X.sample(n=min(len(X), _BACKGROUND_SIZE), random_state=42) if len(X) > _BACKGROUND_SIZE else X
    transformed = np.asarray(pipeline.named_steps["preprocessor"].transform(sample), dtype="float64")
    names = _expanded_names(pipeline, numeric_cols, categorical_cols)
    return transformed, names


def _row_matrix(record, pipeline, inputs: dict):
    missing = [c for c in record.feature_columns if c not in inputs]
    if missing:
        raise AppError("MISSING_FEATURES", f"Missing values for: {', '.join(missing)}", status_code=400)
    row_dict = {}
    for c in record.feature_columns:
        val = inputs.get(c)
        if val in (None, "", "null", "NULL", "NA", "N/A", "-", "?", "missing"):
            row_dict[c] = np.nan
        else:
            row_dict[c] = val
    row_df = pd.DataFrame([row_dict])[record.feature_columns]
    return np.asarray(pipeline.named_steps["preprocessor"].transform(row_df), dtype="float64")


def _predicted_class_index(pipeline, matrix: np.ndarray) -> int:
    probs = pipeline.named_steps["model"].predict_proba(matrix)[0]
    return int(np.argmax(probs))


def _tree_shap(pipeline, matrix: np.ndarray, problem_type: str, class_index: int | None):
    import shap

    estimator = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(estimator)
    raw = np.asarray(explainer.shap_values(matrix))
    expected = np.asarray(explainer.expected_value, dtype="float64").reshape(-1)

    if problem_type == "classification" and raw.ndim == 3:
        # (n_samples, n_features, n_classes)
        idx = class_index if class_index is not None else 0
        values = raw[:, :, idx]
        base_value = float(expected[idx]) if expected.size > idx else float(expected[0])
    else:
        values = raw if raw.ndim == 2 else raw.reshape(1, -1)
        base_value = float(expected[0])
    return values, base_value


def _linear_contribution(pipeline, matrix: np.ndarray, background: np.ndarray, problem_type: str, class_index: int | None):
    estimator = pipeline.named_steps["model"]
    coef = np.asarray(estimator.coef_, dtype="float64") if hasattr(estimator, "coef_") else None
    if coef is None:
        raise AppError("EXPLANATION_UNAVAILABLE", "This model has no coefficients to explain.", status_code=400)
    intercept = np.asarray(getattr(estimator, "intercept_", 0.0), dtype="float64").reshape(-1)

    if problem_type == "classification" and coef.ndim == 2:
        if coef.shape[0] == 1:
            # Binary logistic regression: coef_ is oriented toward class 1.
            sign = 1.0 if (class_index is None or class_index == 1) else -1.0
            effective_coef = coef[0] * sign
            effective_intercept = float(intercept[0]) * sign
        else:
            idx = class_index if class_index is not None else 0
            effective_coef = coef[idx]
            effective_intercept = float(intercept[idx])
    else:
        effective_coef = coef.reshape(-1)
        effective_intercept = float(intercept[0]) if intercept.size else 0.0

    mean = background.mean(axis=0)
    contributions = (matrix - mean) * effective_coef
    base_value = effective_intercept + float(np.dot(mean, effective_coef))
    return contributions, base_value


def _top_contributions(values_row: np.ndarray, names: list[str], top_n: int = _TOP_N) -> list[dict]:
    order = np.argsort(-np.abs(values_row))[:top_n]
    return [
        {"feature": names[i], "contribution": float(values_row[i]), "direction": "positive" if values_row[i] >= 0 else "negative"}
        for i in order
    ]


def explain_instance(model_id: str, inputs: dict) -> dict:
    record = model_service.require_model(model_id)
    cap = capability(model_id)
    if not cap["supported"]:
        return {"model_id": model_id, "capability": cap, "explanation": None}

    pipeline = store.get_pipeline(model_id)
    if pipeline is None:
        raise AppError("MODEL_UNAVAILABLE", "This model's fitted pipeline is no longer available.", status_code=404)

    matrix = _row_matrix(record, pipeline, inputs)
    background, names = _background(record, pipeline)

    class_index = None
    predicted_label = None
    if record.problem_type == "classification" and hasattr(pipeline, "predict_proba"):
        class_index = _predicted_class_index(pipeline, matrix)
        classes = list(getattr(pipeline, "classes_", []))
        if class_index < len(classes):
            predicted_label = classes[class_index]

    if cap["method"] == "tree":
        values, base_value = _tree_shap(pipeline, matrix, record.problem_type, class_index)
    else:
        values, base_value = _linear_contribution(pipeline, matrix, background, record.problem_type, class_index)

    row_values = values[0]
    predicted_value = float(base_value + row_values.sum())

    return {
        "model_id": model_id,
        "capability": cap,
        "explanation": {
            "base_value": float(base_value),
            "predicted_value": predicted_value,
            "predicted_label": predicted_label,
            "top_contributions": _top_contributions(row_values, names),
        },
    }


def explain_for_prediction(prediction_id: str, row_index: int) -> dict:
    """Convenience wrapper for the Student Profile page: pulls the row's own
    feature values out of an already-computed batch prediction, then reuses
    explain_instance for the model that produced it."""
    record = store.get_prediction(prediction_id)
    df = store.get_prediction_frame(prediction_id)
    if record is None or df is None:
        raise AppError("PREDICTION_NOT_FOUND", "Those prediction results are no longer available.", status_code=404)
    if row_index not in df.index:
        raise AppError("STUDENT_NOT_FOUND", "That student row is not part of this prediction run.", status_code=404)

    model_record = model_service.require_model(record.model_id)
    row = df.loc[row_index]
    missing = [c for c in model_record.feature_columns if c not in row.index]
    if missing:
        raise AppError("MISSING_FEATURES", f"This prediction run is missing columns: {', '.join(missing)}", status_code=400)
    inputs = {c: row[c] for c in model_record.feature_columns}
    return explain_instance(record.model_id, inputs)


def global_importance(model_id: str) -> dict:
    record = model_service.require_model(model_id)
    cap = capability(model_id)
    if not cap["supported"]:
        return {"model_id": model_id, "capability": cap, "importance": None}

    pipeline = store.get_pipeline(model_id)
    if pipeline is None:
        raise AppError("MODEL_UNAVAILABLE", "This model's fitted pipeline is no longer available.", status_code=404)

    background, names = _background(record, pipeline)

    if cap["method"] == "tree":
        if record.problem_type == "classification" and hasattr(pipeline, "predict_proba"):
            import shap

            estimator = pipeline.named_steps["model"]
            explainer = shap.TreeExplainer(estimator)
            raw = np.asarray(explainer.shap_values(background))
            magnitude = np.mean(np.abs(raw), axis=(0, 2)) if raw.ndim == 3 else np.mean(np.abs(raw), axis=0)
        else:
            values, _ = _tree_shap(pipeline, background, record.problem_type, None)
            magnitude = np.mean(np.abs(values), axis=0)
    else:
        # Average over each background row's own predicted class so a
        # multinomial model's global importance isn't tied to one class.
        if record.problem_type == "classification" and hasattr(pipeline, "predict_proba"):
            predicted = pipeline.named_steps["model"].predict_proba(background)
            class_indices = np.argmax(predicted, axis=1)
            rows = []
            for i in range(background.shape[0]):
                contrib, _ = _linear_contribution(pipeline, background[i : i + 1], background, record.problem_type, int(class_indices[i]))
                rows.append(contrib[0])
            magnitude = np.mean(np.abs(np.asarray(rows)), axis=0)
        else:
            contributions, _ = _linear_contribution(pipeline, background, background, record.problem_type, None)
            magnitude = np.mean(np.abs(contributions), axis=0)

    order = np.argsort(-magnitude)[:_TOP_N]
    return {
        "model_id": model_id,
        "capability": cap,
        "importance": {
            "features": [names[i] for i in order],
            "importance": [float(magnitude[i]) for i in order],
            "sample_size": int(background.shape[0]),
        },
    }
