"""Student-level views over a stored batch prediction result:

  * At-Risk Student Detection (18)
  * Individual Student Profile (19)
  * Interactive Prediction Explorer (20)
  * Student Comparison (34)
  * Confidence Analysis (37) / Low-Confidence Review Queue (52)

Everything here reads the DataFrame a batch prediction already produced
(app.services.prediction_service / app.services.store) and never re-scores,
retrains or mutates it. Filtering, sorting and pagination all happen on the
backend against the stored frame, so a 100,000-row prediction run never
ships to the browser as one payload — only the current page does.
"""

import pandas as pd

from app.core.errors import AppError
from app.ml import risk
from app.services.store import store
from app.utils.serialization import frame_to_records, to_jsonable

_RISK_ORDER = {"high": 0, "medium": 1, "low": 2, None: 3}
_SEARCHABLE_EXTRA = ("Prediction", "Predicted Class")


def _context(prediction_id: str) -> dict:
    record = store.get_prediction(prediction_id)
    df = store.get_prediction_frame(prediction_id)
    if record is None or df is None:
        raise AppError(
            "PREDICTION_NOT_FOUND",
            "Those prediction results are no longer available. Run the batch prediction again.",
            status_code=404,
        )
    if record.kind != "batch":
        raise AppError(
            "NOT_A_BATCH_PREDICTION",
            "Student-level views need a batch prediction run, not a single prediction.",
            status_code=400,
        )

    capability = risk.risk_capability(record.problem_type, df, record.has_confidence)
    risk_series = risk.compute_risk_series(record.problem_type, df) if capability["supported"] else None
    identifier_column = risk.find_identifier_column(
        [c for c in df.columns if c not in ("Prediction", "Confidence", "Predicted Class")]
    )
    model = store.get_model(record.model_id)

    return {
        "record": record,
        "df": df,
        "capability": capability,
        "risk_series": risk_series,
        "identifier_column": identifier_column,
        "model": model,
    }


def capability(prediction_id: str) -> dict:
    ctx = _context(prediction_id)
    record, df = ctx["record"], ctx["df"]
    labels = []
    if "Prediction" in df.columns:
        labels = sorted({str(v) for v in df["Prediction"].dropna().unique().tolist()})[:50]
    return {
        "prediction_id": record.id,
        "problem_type": record.problem_type,
        "has_confidence": record.has_confidence,
        "identifier_column": ctx["identifier_column"],
        "row_count": record.row_count,
        "risk": ctx["capability"],
        "prediction_labels": labels,
        "model_id": record.model_id,
        "target_column": ctx["model"].target_column if ctx["model"] else None,
        "model_choice": ctx["model"].model_choice if ctx["model"] else None,
    }


def _apply_filters(ctx, search, risk_filter, prediction_filter, confidence_min, confidence_max):
    df = ctx["df"]
    mask = pd.Series(True, index=df.index)

    if search:
        needle = str(search).strip().lower()
        if needle:
            cols = [c for c in ([ctx["identifier_column"]] if ctx["identifier_column"] else []) + list(_SEARCHABLE_EXTRA) if c and c in df.columns]
            if cols:
                found = pd.Series(False, index=df.index)
                for col in cols:
                    found = found | df[col].astype(str).str.lower().str.contains(needle, na=False, regex=False)
                mask &= found
            else:
                mask &= False

    if risk_filter and ctx["risk_series"] is not None:
        mask &= ctx["risk_series"].isin(risk_filter)
    elif risk_filter and ctx["risk_series"] is None:
        mask &= False

    if prediction_filter and "Prediction" in df.columns:
        mask &= df["Prediction"].astype(str) == str(prediction_filter)

    if ctx["record"].has_confidence and "Confidence" in df.columns:
        if confidence_min is not None:
            mask &= df["Confidence"] >= confidence_min
        if confidence_max is not None:
            mask &= df["Confidence"] <= confidence_max

    filtered = df.loc[mask]
    filtered_risk = ctx["risk_series"].loc[mask] if ctx["risk_series"] is not None else None
    return filtered, filtered_risk


def _rows_with_extras(window_df, window_risk, has_confidence):
    records = frame_to_records(window_df)
    for i, rec in enumerate(records):
        idx = window_df.index[i]
        rec["_row_index"] = int(idx)
        conf = None
        if has_confidence and "Confidence" in window_df.columns:
            raw = window_df["Confidence"].iloc[i]
            conf = float(raw) if pd.notna(raw) else None
        rec["_confidence_tier"] = risk.confidence_tier(conf) if conf is not None else None
        if window_risk is not None:
            r = window_risk.iloc[i]
            rec["_risk"] = r if isinstance(r, str) else None
        else:
            rec["_risk"] = None
    return records


def explorer_page(
    prediction_id: str,
    page: int = 1,
    page_size: int = 25,
    search: str = "",
    risk_filter: list[str] | None = None,
    prediction_filter: str | None = None,
    confidence_min: float | None = None,
    confidence_max: float | None = None,
    sort_by: str | None = None,
    sort_dir: str = "asc",
) -> dict:
    ctx = _context(prediction_id)
    filtered, filtered_risk = _apply_filters(ctx, search, risk_filter, prediction_filter, confidence_min, confidence_max)

    if sort_by == "_risk" and filtered_risk is not None:
        order = filtered_risk.map(_RISK_ORDER)
        sorted_index = order.sort_values(ascending=(sort_dir == "asc"), kind="stable").index
        filtered = filtered.loc[sorted_index]
        filtered_risk = filtered_risk.loc[sorted_index]
    elif sort_by and sort_by in filtered.columns:
        ascending = sort_dir != "desc"
        filtered = filtered.sort_values(sort_by, ascending=ascending, kind="stable")
        if filtered_risk is not None:
            filtered_risk = filtered_risk.loc[filtered.index]

    page_size = max(1, min(page_size, 200))
    total_rows = len(filtered)
    total_pages = max(1, -(-total_rows // page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    window = filtered.iloc[start:start + page_size]
    window_risk = filtered_risk.iloc[start:start + page_size] if filtered_risk is not None else None

    return {
        "prediction_id": prediction_id,
        "columns": [str(c) for c in ctx["df"].columns],
        "identifier_column": ctx["identifier_column"],
        "rows": _rows_with_extras(window, window_risk, ctx["record"].has_confidence),
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "problem_type": ctx["record"].problem_type,
        "has_confidence": ctx["record"].has_confidence,
        "risk": ctx["capability"],
    }


def explorer_csv(
    prediction_id: str,
    search: str = "",
    risk_filter: list[str] | None = None,
    prediction_filter: str | None = None,
    confidence_min: float | None = None,
    confidence_max: float | None = None,
) -> bytes:
    ctx = _context(prediction_id)
    filtered, filtered_risk = _apply_filters(ctx, search, risk_filter, prediction_filter, confidence_min, confidence_max)
    out = filtered.copy()
    if filtered_risk is not None:
        out["Risk"] = filtered_risk.map(lambda v: v.capitalize() if isinstance(v, str) else "")
    return out.to_csv(index=False).encode("utf-8")


def _sections_for_row(row: pd.Series) -> dict:
    sections = {"academic": [], "behavioral": [], "demographic": [], "other": []}
    for col, value in row.items():
        if col in ("Prediction", "Confidence", "Predicted Class"):
            continue
        bucket = risk.categorize_column(col)
        sections[bucket].append({"name": col, "value": to_jsonable(value)})
    return sections


def student_profile(prediction_id: str, row_index: int) -> dict:
    ctx = _context(prediction_id)
    df = ctx["df"]
    if row_index not in df.index:
        raise AppError(
            "STUDENT_NOT_FOUND",
            "That student row is not part of this prediction run.",
            status_code=404,
        )
    row = df.loc[row_index]
    identifier_column = ctx["identifier_column"]
    identifier_value = to_jsonable(row[identifier_column]) if identifier_column else None

    risk_value = None
    if ctx["risk_series"] is not None:
        r = ctx["risk_series"].loc[row_index]
        risk_value = r if isinstance(r, str) else None

    confidence_value = None
    if ctx["record"].has_confidence and "Confidence" in df.columns:
        raw = row.get("Confidence")
        confidence_value = float(raw) if pd.notna(raw) else None

    return {
        "prediction_id": prediction_id,
        "row_index": int(row_index),
        "identifier_column": identifier_column,
        "identifier_value": identifier_value,
        "display_label": identifier_value if identifier_value not in (None, "") else f"Row {row_index + 1}",
        "sections": _sections_for_row(row),
        "prediction": {
            "value": to_jsonable(row.get("Prediction")),
            "target_column": ctx["model"].target_column if ctx["model"] else None,
            "problem_type": ctx["record"].problem_type,
        },
        "confidence": {"value": confidence_value, "tier": risk.confidence_tier(confidence_value)},
        "risk": {"value": risk_value, "capability": ctx["capability"]},
    }


def student_compare(prediction_id: str, row_indices: list[int]) -> dict:
    unique_indices = list(dict.fromkeys(row_indices))
    if len(unique_indices) < 2 or len(unique_indices) > 4:
        raise AppError("INVALID_SELECTION", "Select between 2 and 4 students to compare.", status_code=400)
    return {
        "prediction_id": prediction_id,
        "students": [student_profile(prediction_id, idx) for idx in unique_indices],
    }


def confidence_summary(prediction_id: str) -> dict:
    ctx = _context(prediction_id)
    record, df = ctx["record"], ctx["df"]
    if not record.has_confidence or "Confidence" not in df.columns:
        return {
            "prediction_id": prediction_id,
            "available": False,
            "reason": "This model does not produce prediction probabilities.",
        }
    conf = pd.to_numeric(df["Confidence"], errors="coerce").dropna()
    tiers = conf.apply(risk.confidence_tier)
    counts = tiers.value_counts()
    return {
        "prediction_id": prediction_id,
        "available": True,
        "total": int(conf.size),
        "mean": float(conf.mean()) if conf.size else None,
        "min": float(conf.min()) if conf.size else None,
        "max": float(conf.max()) if conf.size else None,
        "high": int(counts.get("high", 0)),
        "medium": int(counts.get("medium", 0)),
        "low": int(counts.get("low", 0)),
    }
