"""Column-kind introspection, schema validation, target detection and batch
validation helpers.

Ported from the original project's utils/helpers.py. The logic is unchanged
— only the Streamlit import (which was never used here) is gone.
"""

import re

import pandas as pd

from app.core.config import TARGET_COLUMN_KEYWORDS


def _column_kind(series: pd.Series) -> str:
    """Classify a single column as 'bool', 'numeric', 'category', 'text' or
    'other', using pandas' public dtype-introspection API rather than
    comparing `.dtype` against string literals.

    WHY THIS EXISTS: `df.select_dtypes(include=['object'])` used to be the
    standard way to grab text columns, but pandas 3.x introduced a dedicated
    default string dtype (reported as dtype name "str") for plain text
    columns instead of "object". Using is_numeric_dtype/is_bool_dtype/
    is_string_dtype/is_object_dtype is the version-stable way to ask "is this
    column numbers, booleans, or text/category?" and behaves identically
    whether the column reports as "object", "str" or "category".
    """
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return "text"
    return "other"


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Purely numeric columns (excludes bool)."""
    return [c for c in df.columns if _column_kind(df[c]) == "numeric"]


def get_text_category_columns(df: pd.DataFrame) -> list:
    """Text/category columns, excluding bool."""
    return [c for c in df.columns if _column_kind(df[c]) in ("text", "category")]


def get_categorical_like_columns(df: pd.DataFrame) -> list:
    """Text/category/bool columns, used wherever bool columns should be
    treated as categorical features (e.g. building the preprocessing
    pipeline)."""
    return [c for c in df.columns if _column_kind(df[c]) in ("text", "category", "bool")]


def validate_schema(df, target_column):
    """Validate dataset schema: ensure target column exists and is not all null."""
    if target_column not in df.columns:
        return False, f"Target column '{target_column}' not found in dataset."
    if df[target_column].isnull().all():
        return False, f"Target column '{target_column}' contains only null values."
    return True, "Schema is valid."


def get_feature_columns(df, target_column):
    """Return list of feature columns excluding target."""
    return [col for col in df.columns if col != target_column]


def _normalize(name: str) -> str:
    """Lowercase and strip everything except letters/digits, so 'Exam_Score',
    'exam score' and 'ExamScore' all normalize the same."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_target_candidates(df):
    """Return dataset columns that look like a student-performance target
    (exam_score, grade, pass_fail, etc.), ordered by keyword priority.

    This is what keeps the app behaving like a Student Performance Prediction
    System instead of a generic "pick any column" AutoML tool: columns are
    only suggested if their name matches a known performance indicator. If
    nothing matches, the caller should fall back to manual column selection.
    """
    normalized_cols = {col: _normalize(col) for col in df.columns}
    candidates = []
    for keyword in TARGET_COLUMN_KEYWORDS:
        nk = _normalize(keyword)
        for col, ncol in normalized_cols.items():
            if col in candidates:
                continue
            if nk and (nk in ncol or ncol in nk):
                candidates.append(col)
    return candidates


def validate_batch_prediction_data(pred_df, feature_columns, reference_df):
    """Validate an uploaded batch-prediction dataset against the trained
    model's expected feature columns.

    Checks: missing required columns, extra/unexpected columns, and datatype
    mismatches (numeric vs categorical) against the training data. Returns
    errors (blocking) and warnings (non-blocking).
    """
    errors = []
    warnings = []

    if pred_df is None or pred_df.empty:
        return {"valid": False, "errors": ["The uploaded file contains no rows."], "warnings": []}

    missing_cols = [c for c in feature_columns if c not in pred_df.columns]
    extra_cols = [c for c in pred_df.columns if c not in feature_columns]

    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    if extra_cols:
        warnings.append(f"Extra columns detected and will be ignored: {', '.join(extra_cols)}")

    for col in feature_columns:
        if col not in pred_df.columns:
            continue
        expected_numeric = pd.api.types.is_numeric_dtype(reference_df[col])
        actual_numeric = pd.api.types.is_numeric_dtype(pred_df[col])
        if expected_numeric and not actual_numeric:
            # Try to coerce common cases like "85" read as text
            coerced = pd.to_numeric(pred_df[col], errors="coerce")
            if coerced.notna().sum() == 0:
                errors.append(f"Column '{col}' is expected to be numeric but contains non-numeric values.")
            else:
                warnings.append(f"Column '{col}' was text but has been converted to numeric.")
        elif not expected_numeric and actual_numeric:
            warnings.append(f"Column '{col}' is expected to be categorical but numeric values were found; treating as text.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def coerce_batch_dtypes(pred_df, feature_columns, reference_df):
    """Best-effort coercion of a batch-prediction dataframe's dtypes to match
    the training data, so numeric columns uploaded as text (a common CSV/Excel
    quirk) don't silently break the preprocessing pipeline's ColumnTransformer
    column matching."""
    df = pred_df.copy()
    for col in feature_columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(reference_df[col]) and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif not pd.api.types.is_numeric_dtype(reference_df[col]) and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df


def get_feature_types(df, feature_cols):
    """Map each trained feature column to the input kind the prediction form
    should render: 'integer', 'numeric', 'categorical' or 'unknown'.

    Ported from the original Prediction page's get_feature_types().
    """
    feature_types = {}
    for col in feature_cols:
        if col not in df.columns:
            feature_types[col] = "unknown"
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            feature_types[col] = "categorical"
        elif pd.api.types.is_integer_dtype(df[col]):
            feature_types[col] = "integer"
        elif pd.api.types.is_float_dtype(df[col]):
            feature_types[col] = "numeric"
        elif (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        ):
            feature_types[col] = "categorical"
        else:
            feature_types[col] = "unknown"
    return feature_types
