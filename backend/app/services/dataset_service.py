"""Dataset ingestion, inspection, paginated preview and exploratory filtering.

The browser never receives a whole dataset: previews are sliced server-side
and returned one page at a time, and filtering is evaluated here against the
stored frame.

FILTERING IS EXPLORATORY ONLY. Nothing in this module writes a filtered frame
back over the stored dataset, so the data used for training and prediction is
untouched no matter what the analysis page does — the same guarantee the
original page comment described.
"""

import pandas as pd

from app.core.errors import AppError, dataset_not_found
from app.core.timing import track
from app.ml import eda
from app.ml.helpers import (
    detect_target_candidates,
    get_numeric_columns,
    get_text_category_columns,
)
from app.services.store import DatasetRecord, new_id, store
from app.utils.file_utils import (
    parse_tabular_bytes,
    safe_display_name,
    storage_path_for,
    validate_frame,
    validate_upload,
)
from app.utils.serialization import frame_to_records, to_jsonable


def create_dataset(file_bytes: bytes, filename: str, session_id: str) -> DatasetRecord:
    """Validate, parse, profile and register an uploaded dataset."""
    with track("Dataset Processing", meta={"filename": filename}):
        display_name = safe_display_name(filename)
        extension = validate_upload(display_name, len(file_bytes))

        df = parse_tabular_bytes(file_bytes, extension)
        validate_frame(df)

        dataset_id = new_id("ds")
        path = storage_path_for(dataset_id, extension)
        with open(path, "wb") as handle:
            handle.write(file_bytes)

        metadata = eda.compute_dataset_metadata(df)
        total_missing = sum(metadata["missing_summary"].values())
        record = DatasetRecord(
            id=dataset_id,
            filename=display_name,
            extension=extension,
            storage_path=path,
            file_size=len(file_bytes),
            num_rows=int(len(df)),
            num_cols=int(len(df.columns)),
            column_names=[str(c) for c in df.columns],
            dtypes=metadata["dtypes"],
            missing_summary=metadata["missing_summary"],
            duplicate_count=metadata["duplicate_count"],
            detected_targets=detect_target_candidates(df),
            refined=(total_missing == 0),
            session_id=session_id,
        )

        store.add_dataset(record, df)
        # Registering a new active dataset clears the active model.
        store.set_active_dataset(session_id, dataset_id)
        store.touch(session_id, f"Uploaded {display_name}", stage="analyze")
        return record


def require_dataset(dataset_id: str):
    """Fetch a dataset record and its frame, or raise a clean error."""
    record = store.get_dataset(dataset_id)
    if record is None:
        raise dataset_not_found(dataset_id)
    df = store.get_frame(dataset_id)
    if df is None:
        raise dataset_not_found(dataset_id)
    return record, df


def dataset_summary(dataset_id: str) -> dict:
    record, df = require_dataset(dataset_id)
    summary = record.to_summary()
    summary["column_profiles"] = _cached(
        dataset_id,
        "column_profiles",
        lambda: eda.column_profiles(df, {
            "dtypes": record.dtypes,
            "missing_summary": record.missing_summary,
        }),
    )
    summary["overview"] = _cached(dataset_id, "overview", lambda: eda.compute_overview_stats(df))
    summary["total_missing"] = summary["overview"]["total_missing"]
    summary["missing_values_detected"] = summary["total_missing"]
    summary["missing_values_handled"] = summary["total_missing"] if record.refined else 0
    summary["missing_values_remaining"] = 0 if record.refined else summary["total_missing"]
    summary["refined"] = record.refined
    summary["models_trained"] = len(store.models_for_dataset(dataset_id))
    return summary


def refine_dataset(dataset_id: str) -> dict:
    """Mark dataset as refined and return dynamic cleaning metrics."""
    record, df = require_dataset(dataset_id)
    record.refined = True
    store._write_sidecar(record)

    missing_info = eda.missing_analysis(df)
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_text_category_columns(df)

    target_rows_excluded = 0
    if record.detected_targets:
        for t_col in record.detected_targets:
            if t_col in df.columns:
                target_rows_excluded = int(df[t_col].isnull().sum())
                break

    return {
        "status": "Complete",
        "dataset_id": dataset_id,
        "refined": True,
        "rows": record.num_rows,
        "cols": record.num_cols,
        "total_missing": missing_info["total_missing_cells"],
        "missing_values_detected": missing_info["total_missing_cells"],
        "missing_values_handled": missing_info["total_missing_cells"],
        "missing_values_remaining": 0,
        "columns_affected": missing_info["affected_columns_count"],
        "numeric_cols_count": len(numeric_cols),
        "categorical_cols_count": len(categorical_cols),
        "target_rows_excluded": target_rows_excluded,
        "ready": True,
        "message": "Data refined. Ready for training.",
        "download_url": f"/api/dataset/{dataset_id}/download-cleaned",
    }


def dataset_preview(dataset_id: str, page: int, page_size: int, search: str = "") -> dict:
    """One page of rows. Search matches any cell as text, which is what a
    user scanning a preview table expects."""
    _, df = require_dataset(dataset_id)

    view = df
    if search:
        needle = search.strip().lower()
        if needle:
            mask = pd.Series(False, index=df.index)
            for col in df.columns:
                mask |= df[col].astype(str).str.lower().str.contains(needle, regex=False, na=False)
            view = df[mask]

    total = int(len(view))
    page_size = max(1, min(page_size, 200))
    total_pages = max(1, -(-total // page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    window = view.iloc[start:start + page_size]

    return {
        "columns": [str(c) for c in df.columns],
        "rows": frame_to_records(window),
        "page": page,
        "page_size": page_size,
        "total_rows": total,
        "total_pages": total_pages,
        "dataset_rows": int(len(df)),
        "search": search,
    }


def dataset_statistics(dataset_id: str) -> dict:
    """Descriptive statistics for numeric columns."""
    _, df = require_dataset(dataset_id)

    def _build():
        numeric_cols = get_numeric_columns(df)
        if not numeric_cols:
            return {"columns": [], "rows": []}
        described = df[numeric_cols].describe().transpose()
        rows = []
        for name, row in described.iterrows():
            rows.append({
                "column": str(name),
                "count": to_jsonable(row.get("count")),
                "mean": to_jsonable(row.get("mean")),
                "std": to_jsonable(row.get("std")),
                "min": to_jsonable(row.get("min")),
                "p25": to_jsonable(row.get("25%")),
                "p50": to_jsonable(row.get("50%")),
                "p75": to_jsonable(row.get("75%")),
                "max": to_jsonable(row.get("max")),
            })
        return {"columns": numeric_cols, "rows": rows}

    return _cached(dataset_id, "statistics", _build)


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Build the filtered view. Mirrors the original filter panel: an optional
    numeric range, an optional category selection, and an optional
    drop-missing-rows step, applied in that order.

    Returns a new frame. The stored dataset is never modified.
    """
    filtered = df

    numeric = (filters or {}).get("numeric")
    if numeric and numeric.get("column") in filtered.columns:
        col = numeric["column"]
        low = numeric.get("min")
        high = numeric.get("max")
        if low is not None:
            filtered = filtered[filtered[col] >= float(low)]
        if high is not None:
            filtered = filtered[filtered[col] <= float(high)]

    categorical = (filters or {}).get("categorical")
    if categorical and categorical.get("column") in filtered.columns:
        col = categorical["column"]
        values = categorical.get("values") or []
        if values:
            filtered = filtered[filtered[col].astype(str).isin([str(v) for v in values])]

    if (filters or {}).get("drop_missing"):
        filtered = filtered.dropna()

    return filtered


def filter_preview(dataset_id: str, filters: dict, limit: int = 50) -> dict:
    record, df = require_dataset(dataset_id)
    filtered = apply_filters(df, filters)
    return {
        "dataset_id": record.id,
        "matched_rows": int(len(filtered)),
        "total_rows": int(len(df)),
        "changed": len(filtered) != len(df),
        "columns": [str(c) for c in df.columns],
        "rows": frame_to_records(filtered.head(limit)),
        "preview_rows": int(min(limit, len(filtered))),
    }


def filter_csv(dataset_id: str, filters: dict) -> bytes:
    _, df = require_dataset(dataset_id)
    filtered = apply_filters(df, filters)
    if filtered.empty:
        raise AppError("EMPTY_FILTER_RESULT", "No rows match these filters, so there is nothing to download.")
    return filtered.to_csv(index=False).encode("utf-8")


def filter_options(dataset_id: str) -> dict:
    dataset_id_, df = require_dataset(dataset_id)
    return _cached(dataset_id, "filter_options", lambda: eda.filter_options(df))


def get_cleaned_csv(dataset_id: str) -> tuple[bytes, str]:
    """Return UTF-8 CSV bytes and filename for the cleaned/refined dataset.

    Rules:
    - Available only after refinement (or if dataset has no missing values).
    - Preserves original column names, row order, and non-missing values.
    - Imputes numeric features with column median.
    - Imputes categorical features with column mode (most frequent).
    - Target columns are never artificially imputed.
    - Original uploaded dataset remains completely untouched.
    """
    import os
    record, df = require_dataset(dataset_id)
    total_missing = sum(record.missing_summary.values()) if isinstance(record.missing_summary, dict) else 0
    if not record.refined and total_missing > 0:
        raise AppError("NOT_REFINED", "Dataset must be refined before downloading cleaned data.", status_code=400)

    df_cleaned = df.copy()
    targets = set(record.detected_targets or [])

    for col in df_cleaned.columns:
        if col in targets:
            continue  # Target column is never artificially imputed
        if df_cleaned[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                med = df_cleaned[col].median()
                if pd.notna(med):
                    df_cleaned[col] = df_cleaned[col].fillna(med)
            else:
                modes = df_cleaned[col].dropna().mode()
                fill_val = modes.iloc[0] if len(modes) > 0 else "Unknown"
                df_cleaned[col] = df_cleaned[col].fillna(fill_val)

    stem, _ = os.path.splitext(record.filename)
    if "_missing" in stem:
        cleaned_stem = stem.replace("_missing", "_cleaned")
    else:
        cleaned_stem = f"{stem}_cleaned"
    cleaned_filename = f"{cleaned_stem}.csv"

    csv_bytes = df_cleaned.to_csv(index=False).encode("utf-8")
    return csv_bytes, cleaned_filename


def _cached(dataset_id: str, key: str, builder):
    """Memoise a derived result per dataset. Keys are scoped by dataset_id,
    so results can never leak from one dataset to another."""
    hit = store.cached_analysis(dataset_id, key)
    if hit is not None:
        return hit
    return store.cache_analysis(dataset_id, key, builder())
