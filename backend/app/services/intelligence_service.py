"""Post-upload dataset intelligence, transparent model recommendation,
data quality analysis, and factual insight generation.

Everything here is derived from the actual uploaded data — no values are
invented. The module reuses existing EDA functions, column profiles, and
model recommendation logic from the current codebase and extends them with
richer profiling, dynamic recommendation reasoning, and Unit-5-style
workflow context notes.
"""

import math

import numpy as np
import pandas as pd

from app.core.config import MODEL_OPTIONS, TARGET_COLUMN_KEYWORDS
from app.ml.helpers import (
    get_categorical_like_columns,
    get_numeric_columns,
    get_text_category_columns,
)
from app.ml.training import detect_problem_type, recommend_model
from app.services.dataset_service import require_dataset, _cached
from app.utils.serialization import to_jsonable

# ── Feature meaning dictionary ────────────────────────────────────────────
# Confident semantic labels for common student-performance column names.
# Only included when the column name matches; unknown columns get a
# factual type-based description instead.
_FEATURE_MEANINGS: dict[str, str] = {
    "attendance": "Attendance measure for the student.",
    "attendance_rate": "Proportion of classes attended.",
    "study_hours": "Recorded study hours.",
    "hours_studied": "Recorded study hours.",
    "study_hours_per_week": "Weekly study time in hours.",
    "previous_score": "Previous academic score.",
    "previous_scores": "Previous academic scores.",
    "assignments_completed": "Number of completed assignments.",
    "sleep_hours": "Recorded sleep duration.",
    "tutoring_sessions": "Number of tutoring sessions attended.",
    "physical_activity": "Recorded physical activity level.",
    "exam_score": "Exam score — typically the prediction target.",
    "final_score": "Final exam or course score.",
    "test_score": "Test score result.",
    "grade": "Academic grade classification.",
    "gpa": "Grade point average.",
    "cgpa": "Cumulative grade point average.",
    "pass_fail": "Binary pass/fail outcome.",
    "placement_status": "Whether the student was placed.",
    "performance_index": "Composite performance index.",
    "performance": "Overall performance measure.",
    "gender": "Student gender.",
    "age": "Student age.",
    "parental_education": "Parent education level.",
    "parental_involvement": "Level of parental involvement.",
    "parental_support": "Level of parental support.",
    "family_income": "Family income bracket.",
    "motivation_level": "Recorded motivation level.",
    "extracurricular_activities": "Participation in extracurricular activities.",
    "internet_access": "Whether the student has internet access.",
    "school_type": "Type of school attended.",
    "peer_influence": "Level of peer influence.",
    "learning_disabilities": "Presence of learning disabilities.",
    "teacher_quality": "Assessed teacher quality.",
    "distance_from_home": "Distance between home and school.",
    "access_to_resources": "Access to learning resources.",
    "result": "Outcome result.",
    "outcome": "Outcome measure.",
    "target": "Target column.",
    "label": "Label column.",
}


def _normalize_name(name: str) -> str:
    """Lowercase, strip whitespace, replace spaces/hyphens with underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def feature_meaning(column_name: str, kind: str, distinct: int) -> str:
    """Return a semantic label if the column name is confidently recognized,
    otherwise return a factual type-based description."""
    normed = _normalize_name(column_name)
    if normed in _FEATURE_MEANINGS:
        return _FEATURE_MEANINGS[normed]
    if kind == "numeric":
        return "Numeric feature in the uploaded dataset."
    if kind == "text":
        return f"Text feature with {distinct} distinct values."
    return f"Categorical feature with {distinct} distinct values."


# ── Dataset Intelligence ──────────────────────────────────────────────────

def dataset_intelligence(dataset_id: str) -> dict:
    """Full post-upload dataset intelligence payload."""
    record, df = require_dataset(dataset_id)
    return _cached(dataset_id, "intelligence", lambda: _build_intelligence(record, df))


def _detect_text_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that contain free-form text (average word count > 3)."""
    text_cols = []
    for col in get_text_category_columns(df):
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        sample = non_null.head(200)
        avg_words = sample.astype(str).str.split().str.len().mean()
        if avg_words > 3:
            text_cols.append(col)
    return text_cols


def _build_intelligence(record, df: pd.DataFrame) -> dict:
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_text_category_columns(df)
    text_cols = _detect_text_columns(df)

    # Remove text columns from categorical count
    pure_categorical = [c for c in categorical_cols if c not in text_cols]

    total_missing = int(df.isnull().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    missing_pct = round(total_missing / max(total_cells, 1) * 100, 2)

    # Target detection
    target_column = record.detected_targets[0] if record.detected_targets else None
    problem_type = None
    if target_column and target_column in df.columns:
        problem_type = detect_problem_type(df, target_column)

    # Feature composition
    composition = {
        "numeric": {"count": len(numeric_cols), "columns": numeric_cols},
        "categorical": {"count": len(pure_categorical), "columns": pure_categorical},
        "text": {"count": len(text_cols), "columns": text_cols},
        "total": len(df.columns),
    }

    # Dataset story
    story = _build_story(record, df, numeric_cols, pure_categorical, text_cols,
                         target_column, problem_type)

    # Enhanced column profiles
    profiles = _build_enhanced_profiles(df, numeric_cols, categorical_cols, text_cols)

    # Target overview
    target_overview = None
    if target_column and target_column in df.columns:
        target_overview = _build_target_overview(df, target_column, problem_type)

    # Data quality
    quality = _build_data_quality(df, record)

    # Dataset insights
    insights = _build_insights(df, record, numeric_cols, pure_categorical,
                               text_cols, target_column, problem_type)

    return {
        "dataset_id": record.id,
        "filename": record.filename,
        "rows": record.num_rows,
        "columns": record.num_cols,
        "numeric_features": len(numeric_cols),
        "categorical_features": len(pure_categorical),
        "text_features": len(text_cols),
        "missing_values": total_missing,
        "missing_percentage": missing_pct,
        "duplicate_rows": record.duplicate_count,
        "target_column": target_column,
        "problem_type": problem_type,
        "file_size": record.file_size,
        "composition": composition,
        "story": story,
        "profiles": profiles,
        "target_overview": target_overview,
        "quality": quality,
        "insights": insights,
    }


def _build_story(record, df, numeric_cols, categorical_cols, text_cols,
                 target_column, problem_type) -> list[str]:
    """Generate factual summary sentences from real data."""
    lines = []
    lines.append(
        f"This dataset contains {record.num_rows:,} records and "
        f"{record.num_cols} features."
    )

    parts = []
    if numeric_cols:
        parts.append(f"{len(numeric_cols)} numeric feature{'s' if len(numeric_cols) != 1 else ''}")
    if categorical_cols:
        parts.append(f"{len(categorical_cols)} categorical feature{'s' if len(categorical_cols) != 1 else ''}")
    if text_cols:
        parts.append(f"{len(text_cols)} text feature{'s' if len(text_cols) != 1 else ''}")
    if parts:
        lines.append("It contains " + ", ".join(parts) + ".")

    if target_column:
        lines.append(f"The target column is {target_column}.")
    else:
        lines.append("No standard target column was automatically detected.")

    if problem_type:
        lines.append(f"The detected task is {problem_type}.")

    missing_total = int(df.isnull().sum().sum())
    if missing_total > 0:
        missing_cols = int((df.isnull().sum() > 0).sum())
        lines.append(
            f"There are {missing_total:,} missing values across "
            f"{missing_cols} column{'s' if missing_cols != 1 else ''}."
        )
    else:
        lines.append("The dataset has no missing values.")

    if record.duplicate_count > 0:
        lines.append(f"{record.duplicate_count:,} duplicate rows were found.")

    return lines


def _build_enhanced_profiles(df, numeric_cols, categorical_cols, text_cols) -> list[dict]:
    """Rich per-column profiles with statistics, meaning, and role."""
    profiles = []
    text_set = set(text_cols)
    num_set = set(numeric_cols)
    total_rows = len(df)

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        missing_count = int(series.isnull().sum())
        missing_pct = round(missing_count / max(total_rows, 1) * 100, 2)
        distinct = int(series.nunique(dropna=True))

        if col in text_set:
            kind = "text"
        elif col in num_set:
            kind = "numeric"
        else:
            kind = "categorical"

        profile: dict = {
            "name": str(col),
            "dtype": str(series.dtype),
            "kind": kind,
            "missing": missing_count,
            "missing_pct": missing_pct,
            "distinct": distinct,
            "meaning": feature_meaning(str(col), kind, distinct),
        }

        # Example values (up to 5)
        if len(non_null) > 0:
            examples = non_null.head(5).tolist()
            profile["examples"] = [to_jsonable(v) for v in examples]
        else:
            profile["examples"] = []

        if kind == "numeric" and len(non_null) > 0:
            arr = non_null.to_numpy(dtype="float64", na_value=np.nan)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                profile["min"] = to_jsonable(float(arr.min()))
                profile["max"] = to_jsonable(float(arr.max()))
                profile["mean"] = to_jsonable(float(arr.mean()))
                profile["median"] = to_jsonable(float(np.median(arr)))
                profile["std"] = to_jsonable(float(arr.std()))
                profile["range"] = to_jsonable(float(arr.max() - arr.min()))

        if kind == "categorical" and len(non_null) > 0:
            top = non_null.astype(str).value_counts().head(10)
            profile["top_categories"] = [
                {"value": str(v), "count": int(c), "pct": round(int(c) / max(total_rows, 1) * 100, 1)}
                for v, c in top.items()
            ]

        profiles.append(profile)

    return profiles


def _build_target_overview(df, target_column, problem_type) -> dict:
    """Target column overview with distribution information."""
    series = df[target_column].dropna()
    result: dict = {
        "column": target_column,
        "dtype": str(df[target_column].dtype),
        "problem_type": problem_type,
        "total": int(len(series)),
        "missing": int(df[target_column].isnull().sum()),
    }

    if problem_type == "classification":
        counts = series.astype(str).value_counts()
        total = int(counts.sum())
        labels = [str(v) for v in counts.index]
        values = [int(c) for c in counts.to_numpy()]
        percentages = [round(int(c) / max(total, 1) * 100, 1) for c in counts.to_numpy()]

        # Imbalance detection
        if len(values) >= 2:
            ratio = min(values) / max(max(values), 1)
            imbalanced = ratio < 0.3
        else:
            ratio = 1.0
            imbalanced = False

        result["class_count"] = len(labels)
        result["distribution"] = {
            "labels": labels,
            "counts": values,
            "percentages": percentages,
        }
        result["imbalance"] = {
            "detected": imbalanced,
            "ratio": round(ratio, 3),
            "description": (
                f"Class imbalance detected: minority/majority ratio is {ratio:.2f}."
                if imbalanced else
                "Class distribution appears reasonably balanced."
            ),
        }
    else:
        # Regression
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if not numeric.empty:
            result["min"] = to_jsonable(float(numeric.min()))
            result["max"] = to_jsonable(float(numeric.max()))
            result["mean"] = to_jsonable(float(numeric.mean()))
            result["median"] = to_jsonable(float(numeric.median()))
            result["std"] = to_jsonable(float(numeric.std()))
            result["spread"] = to_jsonable(float(numeric.max() - numeric.min()))

            # Build histogram for regression target
            arr = numeric.to_numpy(dtype="float64")
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                n_bins = min(30, max(5, int(np.unique(arr).size)))
                hist_counts, edges = np.histogram(arr, bins=n_bins)
                result["histogram"] = {
                    "bins": [f"{edges[i]:.3g} – {edges[i+1]:.3g}" for i in range(len(hist_counts))],
                    "counts": [int(c) for c in hist_counts],
                }

    return result


def _build_data_quality(df, record) -> dict:
    """Dataset quality assessment with concrete observations."""
    total_rows = len(df)
    total_cells = total_rows * len(df.columns)

    # Missing values
    missing_per_col = df.isnull().sum()
    missing_cols = missing_per_col[missing_per_col > 0]
    total_missing = int(missing_per_col.sum())

    missing_info = []
    for col in missing_cols.index:
        count = int(missing_cols[col])
        pct = round(count / max(total_rows, 1) * 100, 1)
        missing_info.append({"column": str(col), "count": count, "pct": pct})

    # Constant features (only 1 unique value)
    constant_features = [str(col) for col in df.columns if df[col].nunique(dropna=True) <= 1]

    # High-cardinality categorical
    cat_cols = get_text_category_columns(df)
    high_cardinality = []
    for col in cat_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique > 50:
            high_cardinality.append({
                "column": str(col),
                "unique_values": int(nunique),
                "ratio": round(nunique / max(total_rows, 1), 3),
            })

    # Duplicate rows
    duplicate_count = record.duplicate_count

    # Quality observations
    observations = []
    if total_missing == 0:
        observations.append({"type": "good", "text": "No missing values found."})
    else:
        pct = round(total_missing / max(total_cells, 1) * 100, 2)
        observations.append({
            "type": "warn" if pct > 5 else "info",
            "text": f"{total_missing:,} missing values ({pct}% of all cells) across {len(missing_info)} columns.",
        })

    if duplicate_count == 0:
        observations.append({"type": "good", "text": "No duplicate rows detected."})
    else:
        pct = round(duplicate_count / max(total_rows, 1) * 100, 1)
        observations.append({
            "type": "warn",
            "text": f"{duplicate_count:,} duplicate rows ({pct}% of dataset).",
        })

    if constant_features:
        observations.append({
            "type": "warn",
            "text": f"{len(constant_features)} constant feature(s): {', '.join(constant_features[:5])}.",
        })

    if high_cardinality:
        cols = ", ".join(h["column"] for h in high_cardinality[:3])
        observations.append({
            "type": "info",
            "text": f"High-cardinality categorical columns: {cols}.",
        })

    # Quality score: computed transparently
    # Start at 100, deduct for issues
    score = 100
    if total_missing > 0:
        score -= min(30, round(total_missing / max(total_cells, 1) * 300))
    if duplicate_count > 0:
        score -= min(15, round(duplicate_count / max(total_rows, 1) * 150))
    if constant_features:
        score -= min(10, len(constant_features) * 5)
    if high_cardinality:
        score -= min(10, len(high_cardinality) * 3)
    score = max(0, score)

    return {
        "score": score,
        "score_explanation": (
            "Starts at 100. Deductions: up to 30 for missing values "
            "(proportional to % of cells), up to 15 for duplicates "
            "(proportional to % of rows), up to 10 for constant features "
            "(5 per feature), up to 10 for high-cardinality categoricals "
            "(3 per column)."
        ),
        "total_missing": total_missing,
        "missing_columns": missing_info,
        "duplicate_rows": duplicate_count,
        "constant_features": constant_features,
        "high_cardinality": high_cardinality,
        "observations": observations,
    }


def _build_insights(df, record, numeric_cols, categorical_cols, text_cols,
                    target_column, problem_type) -> list[dict]:
    """Generate factual dataset insights from real data."""
    insights = []

    insights.append({
        "category": "dimensions",
        "text": f"The dataset has {record.num_rows:,} rows and {record.num_cols} columns.",
    })

    if numeric_cols:
        insights.append({
            "category": "composition",
            "text": f"{len(numeric_cols)} numeric features: {', '.join(numeric_cols[:5])}"
                    + (f" and {len(numeric_cols) - 5} more." if len(numeric_cols) > 5 else "."),
        })

    if categorical_cols:
        insights.append({
            "category": "composition",
            "text": f"{len(categorical_cols)} categorical features: {', '.join(categorical_cols[:5])}"
                    + (f" and {len(categorical_cols) - 5} more." if len(categorical_cols) > 5 else "."),
        })

    if text_cols:
        insights.append({
            "category": "text",
            "text": f"{len(text_cols)} text feature(s) detected: {', '.join(text_cols[:3])}.",
        })

    # Target-specific insights
    if target_column and target_column in df.columns:
        if problem_type == "classification":
            counts = df[target_column].dropna().astype(str).value_counts()
            if len(counts) >= 2:
                dominant = str(counts.index[0])
                pct = round(int(counts.iloc[0]) / max(int(counts.sum()), 1) * 100, 1)
                insights.append({
                    "category": "target",
                    "text": f"The dominant class is '{dominant}' at {pct}% of records.",
                })
                if len(counts) >= 2:
                    ratio = int(counts.iloc[-1]) / max(int(counts.iloc[0]), 1)
                    if ratio < 0.3:
                        insights.append({
                            "category": "imbalance",
                            "text": (
                                f"Class imbalance is present: the minority/majority ratio is "
                                f"{ratio:.2f}. Consider evaluation metrics beyond accuracy."
                            ),
                        })
        elif problem_type == "regression":
            numeric = pd.to_numeric(df[target_column], errors="coerce").dropna()
            if not numeric.empty:
                insights.append({
                    "category": "target",
                    "text": (
                        f"The target '{target_column}' ranges from "
                        f"{numeric.min():.3g} to {numeric.max():.3g} "
                        f"(mean {numeric.mean():.3g})."
                    ),
                })

    # Correlation insight (top pair)
    if len(numeric_cols) >= 2:
        try:
            # Only correlate up to 20 columns to stay fast
            cols_to_use = numeric_cols[:20]
            corr_matrix = df[cols_to_use].corr()
            # Mask diagonal and lower triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper = corr_matrix.where(mask)
            abs_corr = upper.abs()
            max_val = abs_corr.max().max()
            if not math.isnan(max_val) and max_val > 0.5:
                # Find the pair
                for col1 in abs_corr.columns:
                    for col2 in abs_corr.index:
                        val = abs_corr.loc[col2, col1]
                        if not math.isnan(val) and abs(val - max_val) < 0.001:
                            actual = corr_matrix.loc[col2, col1]
                            insights.append({
                                "category": "correlation",
                                "text": (
                                    f"Strongest observed correlation: {col2} and "
                                    f"{col1} (r = {actual:.3f})."
                                ),
                            })
                            break
                    else:
                        continue
                    break
        except Exception:
            pass

    # Missing data insight
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        worst_col = df.isnull().sum().idxmax()
        worst_count = int(df[worst_col].isnull().sum())
        worst_pct = round(worst_count / max(len(df), 1) * 100, 1)
        insights.append({
            "category": "missing",
            "text": (
                f"Column '{worst_col}' has the most missing values: "
                f"{worst_count:,} ({worst_pct}%)."
            ),
        })

    # Feature range insights
    for col in numeric_cols[:3]:
        non_null = df[col].dropna()
        if not non_null.empty:
            insights.append({
                "category": "range",
                "text": (
                    f"'{col}' ranges from {non_null.min():.3g} to "
                    f"{non_null.max():.3g}."
                ),
            })

    return insights


# ── Enhanced Model Recommendation ─────────────────────────────────────────

def enhanced_recommendation(dataset_id: str, target_column: str,
                            problem_type: str) -> dict:
    """Build a transparent model recommendation with evidence and
    Unit-5-style workflow notes."""
    _, df = require_dataset(dataset_id)

    recommended = recommend_model(df, target_column, problem_type)

    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_text_category_columns(df)
    text_cols = _detect_text_columns(df)
    n_rows = len(df)

    # Build dynamic recommendation reason
    reason = _build_recommendation_reason(
        recommended, problem_type, n_rows, numeric_cols, categorical_cols
    )

    # Build evidence dict
    evidence = {
        "rows": n_rows,
        "problem_type": problem_type,
        "numeric_features": len(numeric_cols),
        "categorical_features": len([c for c in categorical_cols if c not in text_cols]),
        "text_features": len(text_cols),
        "target_column": target_column,
    }

    # Build Unit 5 workflow notes
    workflow_notes = _build_workflow_notes(
        df, target_column, problem_type, numeric_cols, categorical_cols, text_cols
    )

    return {
        "recommended_model": recommended,
        "recommendation_reason": reason,
        "recommendation_evidence": evidence,
        "workflow_notes": workflow_notes,
        "model_options": MODEL_OPTIONS.get(problem_type, []),
    }


def _build_recommendation_reason(recommended, problem_type, n_rows,
                                 numeric_cols, categorical_cols) -> str:
    """Generate a dynamic explanation tied to the actual dataset."""
    feature_desc = []
    if numeric_cols:
        feature_desc.append(f"{len(numeric_cols)} numeric")
    if categorical_cols:
        feature_desc.append(f"{len(categorical_cols)} categorical")
    feat_str = " and ".join(feature_desc) + " features" if feature_desc else "features"

    if problem_type == "classification":
        if n_rows < 200:
            return (
                f"Selected by the current dataset heuristic because this is a tabular "
                f"classification task with {n_rows:,} rows. With fewer than 200 rows, "
                f"a simpler linear model (Logistic Regression) is recommended to avoid "
                f"overfitting with the available {feat_str}."
            )
        return (
            f"Selected by the current dataset heuristic because this is a tabular "
            f"classification task with {n_rows:,} rows and {feat_str}. With sufficient "
            f"data, a tree-based model handles mixed feature types and non-linear "
            f"relationships effectively."
        )
    else:
        if n_rows < 200:
            return (
                f"Selected by the current dataset heuristic because the target is "
                f"continuous and the dataset has {n_rows:,} rows. With fewer than 200 "
                f"rows, Linear Regression is recommended to reduce overfitting risk with "
                f"the available {feat_str}."
            )
        return (
            f"Selected by the current dataset heuristic because the target is continuous "
            f"and the tabular feature structure ({feat_str}, {n_rows:,} rows) is "
            f"compatible with a tree-based regression workflow."
        )


def _build_workflow_notes(df, target_column, problem_type,
                          numeric_cols, categorical_cols, text_cols) -> list[str]:
    """Generate Unit-5-style educational context notes based on the dataset."""
    notes = []

    # Classification + imbalance
    if problem_type == "classification" and target_column in df.columns:
        counts = df[target_column].dropna().astype(str).value_counts()
        if len(counts) >= 2:
            ratio = int(counts.iloc[-1]) / max(int(counts.iloc[0]), 1)
            if ratio < 0.3:
                dist_str = ":".join(str(int(c)) for c in counts.to_numpy()[:5])
                notes.append(
                    f"The Finance workflow demonstrates using SMOTE for training-set "
                    f"class imbalance. The current dataset has a class distribution of "
                    f"{dist_str}, so imbalance handling may be relevant."
                )

    # Categorical features
    if categorical_cols:
        n_cat = len([c for c in categorical_cols if c not in text_cols])
        if n_cat > 0:
            notes.append(
                f"The Healthcare workflow demonstrates one-hot encoding and train/test "
                f"feature alignment for categorical columns. Similar preprocessing is "
                f"relevant to this dataset because it contains {n_cat} categorical "
                f"feature{'s' if n_cat != 1 else ''}."
            )

    # Regression
    if problem_type == "regression":
        notes.append(
            "The Agriculture workflow demonstrates Random Forest Regression for a "
            "continuous target and evaluates it with MAE and R². These metrics are "
            "appropriate for the current regression task."
        )

    # Text features
    if text_cols:
        notes.append(
            "The Media workflow demonstrates TF-IDF and cosine similarity for "
            "text-based similarity. Text processing is relevant here because the "
            f"current dataset contains {len(text_cols)} text feature(s): "
            f"{', '.join(text_cols[:3])}."
        )

    # Preprocessing note
    notes.append(
        "Categorical preprocessing is applied automatically using one-hot encoding. "
        "Numeric features are imputed (median) and scaled (StandardScaler)."
    )

    # Missing values note
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        notes.append(
            f"Missing values ({total_missing:,} cells) will be automatically imputed "
            "before training."
        )

    return notes


# ── Correlation data for the frontend ─────────────────────────────────────

def feature_correlation(dataset_id: str, x_col: str = "", y_col: str = "") -> dict:
    """Correlation between two specific columns, or the full matrix."""
    _, df = require_dataset(dataset_id)
    numeric_cols = get_numeric_columns(df)

    if x_col and y_col and x_col != y_col:
        if x_col in numeric_cols and y_col in numeric_cols:
            pair = df[[x_col, y_col]].dropna()
            if len(pair) > 1:
                corr_val = float(pair[x_col].corr(pair[y_col]))
                return {
                    "available": True,
                    "x_column": x_col,
                    "y_column": y_col,
                    "correlation": round(corr_val, 4) if not math.isnan(corr_val) else None,
                }
        return {"available": False, "message": "Both columns must be numeric."}

    # Full matrix
    if len(numeric_cols) < 2:
        return {"available": False, "message": "Need at least 2 numeric columns."}

    used = numeric_cols[:30]
    corr = df[used].corr()
    return {
        "available": True,
        "columns": [str(c) for c in corr.columns],
        "matrix": [[to_jsonable(v) for v in row] for row in corr.to_numpy()],
    }
