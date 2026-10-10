"""Exploratory data analysis.

The original project built Plotly figures in Python and shipped them to the
browser. Here the same analyses are computed server-side and returned as
compact chart data, which the React client renders with Plotly.js. The
applicability rules are identical to the originals:

  * box plot  -> numeric columns only, otherwise None
  * heatmap   -> needs at least two numeric columns, otherwise None
  * missing   -> None when nothing is missing
  * scatter   -> None when x and y are the same column

Row-level charts (scatter) use a deterministic sample so the browser never
receives 100k+ points. Aggregate charts (histogram, box, correlation,
missing) are computed over the complete dataset — binning and quartiles are
cheap in pandas and exact answers cost nothing extra.
"""

import numpy as np
import pandas as pd

from app.core.config import (
    MAX_CATEGORY_BARS,
    MAX_HEATMAP_COLUMNS,
    MAX_HISTOGRAM_BINS,
    SCATTER_SAMPLE_LIMIT,
    SCATTER_SAMPLE_SEED,
)
from app.ml.helpers import get_numeric_columns, get_text_category_columns
from app.utils.serialization import to_jsonable


def compute_overview_stats(df: pd.DataFrame) -> dict:
    """Dataset overview stats (row/column counts, numeric/categorical column
    lists, missing values, duplicates). Same six figures the original
    Data Analysis page showed."""
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_text_category_columns(df)
    return {
        "total_rows": int(len(df)),
        "total_cols": int(len(df.columns)),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def compute_dataset_metadata(df: pd.DataFrame) -> dict:
    """Per-column metadata for the Upload page: dtypes, per-column missing
    counts and the duplicate-row count."""
    return {
        "dtypes": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
        "missing_summary": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
        "duplicate_count": int(df.duplicated().sum()),
    }


def _deterministic_sample(df: pd.DataFrame, limit: int) -> tuple:
    """Sample rows reproducibly (fixed seed, original row order preserved).
    Returns (frame, was_sampled)."""
    if len(df) <= limit:
        return df, False
    return df.sample(n=limit, random_state=SCATTER_SAMPLE_SEED).sort_index(), True


def histogram(df: pd.DataFrame, column: str):
    """Distribution of one column.

    Numeric columns are binned; text/category/bool columns are counted by
    value, matching what px.histogram produced for each case.
    """
    if column not in df.columns:
        return None

    series = df[column].dropna()
    if series.empty:
        return None

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        values = series.to_numpy(dtype="float64", na_value=np.nan)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        lo, hi = float(values.min()), float(values.max())
        if lo == hi:
            return {
                "kind": "numeric",
                "column": column,
                "bins": [f"{lo:g}"],
                "counts": [int(values.size)],
                "bin_edges": [lo, hi],
            }
        distinct = int(np.unique(values).size)
        n_bins = int(min(MAX_HISTOGRAM_BINS, max(5, distinct)))
        counts, edges = np.histogram(values, bins=n_bins)
        labels = [f"{edges[i]:.3g} – {edges[i + 1]:.3g}" for i in range(len(counts))]
        return {
            "kind": "numeric",
            "column": column,
            "bins": labels,
            "counts": [int(c) for c in counts],
            "bin_edges": [float(e) for e in edges],
        }

    counts = series.astype(str).value_counts()
    truncated = len(counts) > MAX_CATEGORY_BARS
    counts = counts.head(MAX_CATEGORY_BARS)
    return {
        "kind": "categorical",
        "column": column,
        "bins": [str(i) for i in counts.index],
        "counts": [int(c) for c in counts.to_numpy()],
        "truncated": truncated,
        "distinct": int(series.nunique()),
    }


def boxplot(df: pd.DataFrame, column: str):
    """Five-number summary plus Tukey outliers. Returns None for
    non-numeric columns, matching the original box-plot guard."""
    if column not in get_numeric_columns(df):
        return None

    series = df[column].dropna()
    if series.empty:
        return None

    values = series.to_numpy(dtype="float64", na_value=np.nan)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    q1, median, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    lower_fence = float(values[values >= q1 - 1.5 * iqr].min()) if iqr > 0 else float(values.min())
    upper_fence = float(values[values <= q3 + 1.5 * iqr].max()) if iqr > 0 else float(values.max())

    outliers = values[(values < lower_fence) | (values > upper_fence)]
    if outliers.size > 500:
        step = int(np.ceil(outliers.size / 500))
        outliers = outliers[::step]

    return {
        "column": column,
        "q1": q1,
        "median": median,
        "q3": q3,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "count": int(values.size),
        "outliers": [float(v) for v in outliers],
    }


def correlation(df: pd.DataFrame):
    """Pearson correlation matrix over numeric columns. Returns None when
    fewer than two numeric columns exist, so the client can show a message
    instead of an error."""
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        return None

    truncated = len(numeric_cols) > MAX_HEATMAP_COLUMNS
    used = numeric_cols[:MAX_HEATMAP_COLUMNS]
    corr = df[used].corr()

    return {
        "columns": [str(c) for c in corr.columns],
        "matrix": [[to_jsonable(v) for v in row] for row in corr.to_numpy()],
        "truncated": truncated,
        "total_numeric": len(numeric_cols),
    }


def missing_values(df: pd.DataFrame):
    """Per-column missing counts, restricted to columns that actually have
    missing values. Returns None when the dataset is complete."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None
    total_rows = max(len(df), 1)
    return {
        "columns": [str(c) for c in missing.index],
        "counts": [int(v) for v in missing.to_numpy()],
        "percentages": [round(float(v) / total_rows * 100, 2) for v in missing.to_numpy()],
        "total_missing": int(missing.sum()),
    }


def missing_analysis(df: pd.DataFrame) -> dict:
    """Detailed missing-value analysis with dataset-agnostic detection,
    per-column severity thresholds, handling strategies, before/after metrics,
    and explanation messages."""
    n_rows = len(df)
    n_cols = len(df.columns)
    total_cells = n_rows * n_cols
    col_missing = df.isnull().sum()
    total_missing = int(col_missing.sum())

    overall_missing_pct = round((total_missing / max(total_cells, 1)) * 100, 2)
    rows_with_missing = int(df.isnull().any(axis=1).sum())
    rows_with_missing_pct = round((rows_with_missing / max(n_rows, 1)) * 100, 2)
    affected_cols = [str(c) for c in col_missing[col_missing > 0].index]
    affected_cols_count = len(affected_cols)

    numeric_cols = set(get_numeric_columns(df))
    numeric_imputed = 0
    categorical_imputed = 0
    columns_100_pct_missing = []

    highest_col = None
    if total_missing > 0:
        top_col = col_missing.idxmax()
        top_count = int(col_missing.max())
        highest_col = {
            "column": str(top_col),
            "count": top_count,
            "percentage": round((top_count / max(n_rows, 1)) * 100, 2),
        }

    details = []
    for col in df.columns:
        col_str = str(col)
        missing_count = int(col_missing[col])
        missing_pct = round((missing_count / max(n_rows, 1)) * 100, 2)
        is_num = col in numeric_cols
        col_type = "Numeric" if is_num else "Categorical"

        if missing_count == n_rows and n_rows > 0:
            severity = "Unusable (100% Missing)"
            handling = "Excluded from features (no information available)"
            final_status = "Excluded"
            columns_100_pct_missing.append(col_str)
        elif missing_count == 0:
            severity = "None"
            handling = "No imputation needed"
            final_status = "Clean"
        elif missing_pct <= 5.0:
            severity = "Low"
            handling = "Median imputation" if is_num else "Most frequent"
            final_status = "Imputed"
            if is_num:
                numeric_imputed += 1
            else:
                categorical_imputed += 1
        elif missing_pct <= 20.0:
            severity = "Moderate"
            handling = "Median imputation" if is_num else "Most frequent"
            final_status = "Imputed"
            if is_num:
                numeric_imputed += 1
            else:
                categorical_imputed += 1
        elif missing_pct <= 40.0:
            severity = "High"
            handling = "Median imputation" if is_num else "Most frequent"
            final_status = "Imputed"
            if is_num:
                numeric_imputed += 1
            else:
                categorical_imputed += 1
        else:
            severity = "Very High"
            handling = "Median imputation" if is_num else "Most frequent"
            final_status = "Imputed"
            if is_num:
                numeric_imputed += 1
            else:
                categorical_imputed += 1

        details.append({
            "column": col_str,
            "type": col_type,
            "dtype": str(df[col].dtype),
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "severity": severity,
            "handling": handling,
            "final_status": final_status,
        })

    explanation_message = (
        "Missing values were detected in this dataset. Numeric missing values are filled "
        "using the median, while categorical missing values are filled using the most frequent value. "
        "The original uploaded data is preserved, and the learned cleaning rules are reused during prediction."
    )

    if total_missing > 0:
        dynamic_explanation = (
            f"{total_missing:,} missing values detected across {affected_cols_count} columns. "
            "Numeric fields will use median imputation and categorical fields will use most-frequent imputation."
        )
    else:
        dynamic_explanation = "No missing values detected. All columns are complete and ready for model training."

    quality_status = "Needs Cleaning" if total_missing > 0 else "Valid"
    quality_label = f"⚠ {total_missing:,} missing cells detected" if total_missing > 0 else "✓ Valid"

    return {
        "total_rows": n_rows,
        "total_cols": n_cols,
        "total_cells": total_cells,
        "total_missing_cells": total_missing,
        "overall_missing_pct": overall_missing_pct,
        "rows_with_missing": rows_with_missing,
        "rows_with_missing_pct": rows_with_missing_pct,
        "affected_columns_count": affected_cols_count,
        "highest_missing_column": highest_col,
        "quality_status": quality_status,
        "quality_label": quality_label,
        "numeric_columns_imputed_count": numeric_imputed,
        "categorical_columns_imputed_count": categorical_imputed,
        "columns_100_pct_missing": columns_100_pct_missing,
        "details": details,
        "before_after": {
            "before_missing_cells": total_missing,
            "after_projected_feature_missing": 0,
        },
        "explanation_message": explanation_message,
        "dynamic_explanation": dynamic_explanation,
    }


def generate_cleaning_report_csv(df: pd.DataFrame) -> str:
    """Generate CSV text for the Download Cleaning Report feature."""
    analysis = missing_analysis(df)
    import io, csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Column", "Data Type", "Original Missing Count", "Missing %", "Cleaning Method", "Final Status"])
    for row in analysis["details"]:
        writer.writerow([
            row["column"],
            row["type"],
            row["missing_count"],
            f"{row['missing_pct']:.2f}%",
            row["handling"],
            row["final_status"],
        ])
    return output.getvalue()


def scatter(df: pd.DataFrame, x_col: str, y_col: str):
    """Two numeric columns plotted against each other. Returns None when the
    two columns are the same, preserving the original validation."""
    if x_col == y_col:
        return None
    numeric_cols = get_numeric_columns(df)
    if x_col not in numeric_cols or y_col not in numeric_cols:
        return None

    pair = df[[x_col, y_col]].dropna()
    if pair.empty:
        return None

    sampled, was_sampled = _deterministic_sample(pair, SCATTER_SAMPLE_LIMIT)

    return {
        "x_column": x_col,
        "y_column": y_col,
        "x": [to_jsonable(v) for v in sampled[x_col].to_numpy()],
        "y": [to_jsonable(v) for v in sampled[y_col].to_numpy()],
        "sampled": was_sampled,
        "plotted_points": int(len(sampled)),
        "total_points": int(len(pair)),
    }


def column_profiles(df: pd.DataFrame, metadata: dict) -> list:
    """Per-column card data for the Upload page: dtype, missing count,
    distinct count, example value, and enriched statistics."""
    dtypes = metadata["dtypes"]
    missing = metadata["missing_summary"]
    numeric_cols = set(get_numeric_columns(df))
    total_rows = max(len(df), 1)

    profiles = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        example = to_jsonable(non_null.iloc[0]) if len(non_null) else None
        missing_count = int(missing.get(str(col), 0))
        profile = {
            "name": str(col),
            "dtype": dtypes.get(str(col), str(series.dtype)),
            "kind": "numeric" if col in numeric_cols else "categorical",
            "missing": missing_count,
            "missing_pct": round(missing_count / total_rows * 100, 1),
            "distinct": int(series.nunique(dropna=True)),
            "example": example,
        }
        if col in numeric_cols and len(non_null):
            arr = non_null.to_numpy(dtype="float64", na_value=np.nan)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                profile["min"] = to_jsonable(float(arr.min()))
                profile["max"] = to_jsonable(float(arr.max()))
                profile["mean"] = to_jsonable(float(arr.mean()))
                profile["median"] = to_jsonable(float(np.median(arr)))
                profile["std"] = to_jsonable(float(arr.std()))
        elif col not in numeric_cols and len(non_null):
            top = non_null.astype(str).value_counts().head(5)
            profile["top_categories"] = [
                {"value": str(v), "count": int(c)}
                for v, c in top.items()
            ]
        profiles.append(profile)
    return profiles


def filter_options(df: pd.DataFrame) -> dict:
    """Ranges and category values that drive the exploratory filter panel."""
    numeric = []
    for col in get_numeric_columns(df):
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        numeric.append({
            "name": str(col),
            "min": to_jsonable(float(non_null.min())),
            "max": to_jsonable(float(non_null.max())),
        })

    categorical = []
    for col in get_text_category_columns(df):
        values = df[col].dropna().unique()
        categorical.append({
            "name": str(col),
            "values": [str(v) for v in sorted(values, key=str)[:200]],
            "truncated": len(values) > 200,
        })

    return {"numeric": numeric, "categorical": categorical}
