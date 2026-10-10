"""Turn pandas / numpy values into JSON-safe Python values.

FastAPI's default encoder chokes on numpy scalars and silently emits invalid
JSON for NaN/Infinity, so every value that leaves the API passes through here
first. NaN and +/-Infinity become null, which the frontend renders as an
em-space placeholder rather than the string "NaN".
"""

import math

import numpy as np
import pandas as pd


def to_jsonable(value):
    """Recursively convert a value to something json.dumps can handle."""
    if value is None:
        return None

    if isinstance(value, (np.generic,)):
        value = value.item()

    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value

    if isinstance(value, (bool, int, str)):
        return value

    if value is pd.NaT:
        return None

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]

    if isinstance(value, pd.Series):
        return [to_jsonable(v) for v in value.tolist()]

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def frame_to_records(df: pd.DataFrame) -> list:
    """Convert a DataFrame to a list of JSON-safe row dicts, preserving
    column order. Only ever called on a page-sized slice, never on a full
    100k-row dataset."""
    columns = [str(c) for c in df.columns]
    records = []
    for row in df.itertuples(index=False, name=None):
        records.append({columns[i]: to_jsonable(v) for i, v in enumerate(row)})
    return records
