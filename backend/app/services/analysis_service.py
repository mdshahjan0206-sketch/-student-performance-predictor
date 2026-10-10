"""Chart data for the Data Analysis dashboard.

Every result is cached per dataset under a key that includes the chart's own
parameters, so switching between tabs, columns or chart types re-uses work
instead of recomputing it. Cache keys are scoped by dataset_id, so a chart
computed for one dataset can never be served for another.
"""

from app.core.timing import track
from app.ml import eda
from app.services.dataset_service import require_dataset, _cached


def overview(dataset_id: str) -> dict:
    _, df = require_dataset(dataset_id)
    with track("EDA Overview", meta={"dataset_id": dataset_id}):
        return _cached(dataset_id, "overview", lambda: eda.compute_overview_stats(df))


def histogram(dataset_id: str, column: str) -> dict:
    _, df = require_dataset(dataset_id)
    data = _cached(dataset_id, f"hist::{column}", lambda: eda.histogram(df, column))
    if data is None:
        return {
            "available": False,
            "message": "Histogram not applicable for this column type.",
        }
    return {"available": True, "chart": data}


def boxplot(dataset_id: str, column: str) -> dict:
    _, df = require_dataset(dataset_id)
    data = _cached(dataset_id, f"box::{column}", lambda: eda.boxplot(df, column))
    if data is None:
        return {"available": False, "message": "Box plot requires a numeric column."}
    return {"available": True, "chart": data}


def correlation(dataset_id: str) -> dict:
    _, df = require_dataset(dataset_id)
    with track("Correlation", meta={"dataset_id": dataset_id}):
        data = _cached(dataset_id, "corr", lambda: eda.correlation(df))
    if data is None:
        return {
            "available": False,
            "message": "Correlation heatmap requires at least two numeric columns.",
        }
    return {"available": True, "chart": data}


def missing(dataset_id: str) -> dict:
    _, df = require_dataset(dataset_id)
    chart = _cached(dataset_id, "missing", lambda: eda.missing_values(df))
    analysis = _cached(dataset_id, "missing_analysis", lambda: eda.missing_analysis(df))
    return {
        "available": True,
        "chart": chart,
        "analysis": analysis,
        "message": None if chart is not None else "No missing values detected. All columns are complete.",
    }


def cleaning_report(dataset_id: str) -> str:
    _, df = require_dataset(dataset_id)
    return _cached(dataset_id, "cleaning_report_csv", lambda: eda.generate_cleaning_report_csv(df))


def scatter(dataset_id: str, x_col: str, y_col: str) -> dict:
    _, df = require_dataset(dataset_id)
    if x_col == y_col:
        return {"available": False, "message": "Choose two different columns to compare."}
    data = _cached(dataset_id, f"scatter::{x_col}::{y_col}", lambda: eda.scatter(df, x_col, y_col))
    if data is None:
        return {"available": False, "message": "Scatter plot requires two different numeric columns."}
    return {"available": True, "chart": data}
