"""Upload security and tabular parsing.

Uploads are never trusted: the extension is checked against an allow-list,
the size is capped, and the file is written under a server-generated name
inside a fixed directory, so a crafted filename can neither escape the
uploads folder nor be interpreted as anything executable. The original
filename is kept only as a display label.
"""

import io
import os
import re

import pandas as pd

from app.core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, UPLOADS_DIR
from app.core.errors import AppError

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def safe_display_name(filename: str) -> str:
    """Strip any path component and unusual characters from a user-supplied
    filename before it is stored or echoed back."""
    base = os.path.basename(filename or "").replace("\\", "/").split("/")[-1]
    cleaned = _SAFE_NAME.sub("_", base).strip("._-")
    return cleaned[:120] or "dataset"


def file_extension(filename: str) -> str:
    return (filename or "").lower().rsplit(".", 1)[-1] if "." in (filename or "") else ""


def validate_upload(filename: str, size: int) -> str:
    """Check extension and size. Returns the validated lowercase extension."""
    ext = file_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise AppError(
            "UNSUPPORTED_FORMAT",
            "Unsupported file format. Upload a .csv or .xlsx file.",
        )
    if size <= 0:
        raise AppError("EMPTY_FILE", "The uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise AppError(
            "FILE_TOO_LARGE",
            f"The file is larger than the {limit_mb} MB limit.",
            status_code=413,
        )
    return ext


def storage_path_for(record_id: str, extension: str) -> str:
    """Build a path inside the uploads directory from a server-generated id,
    so no part of it comes from the client."""
    return os.path.join(UPLOADS_DIR, f"{record_id}.{extension}")


MISSING_SENTINELS = [
    "NA", "N/A", "na", "n/a",
    "null", "NULL", "Null",
    "-", "?",
    "missing", "MISSING", "Missing",
    "none", "NONE", "None",
    "nan", "NAN", "NaN",
    "",
]


def normalize_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize sentinel strings, whitespace-only cells, and mixed representations
    into consistent np.nan without altering valid data."""
    import numpy as np

    df = df.copy()
    sentinels = {"", "na", "n/a", "null", "none", "-", "?", "missing", "nan"}
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            cleaned = df[col].apply(
                lambda x: np.nan
                if (pd.isna(x) or (isinstance(x, str) and x.strip().lower() in sentinels))
                else (x.strip() if isinstance(x, str) else x)
            )
            # If all non-null values can cleanly parse as numbers, convert to numeric
            non_null_clean = cleaned.dropna()
            if len(non_null_clean) > 0:
                coerced = pd.to_numeric(cleaned, errors="coerce")
                if len(non_null_clean) == len(coerced.dropna()):
                    df[col] = coerced
                    continue
            df[col] = cleaned
    return df


def parse_tabular_bytes(file_bytes: bytes, file_ext: str) -> pd.DataFrame:
    """Parse CSV/Excel bytes into a DataFrame, translating pandas' parser
    failures into clean, user-facing messages and normalizing missing markers."""
    buffer = io.BytesIO(file_bytes)
    try:
        if file_ext == "csv":
            df = pd.read_csv(buffer, na_values=MISSING_SENTINELS, keep_default_na=True)
        else:
            df = pd.read_excel(buffer, na_values=MISSING_SENTINELS, keep_default_na=True)
        return normalize_missing_markers(df)
    except pd.errors.EmptyDataError as exc:
        raise AppError("EMPTY_FILE", "The file is empty or corrupted.") from exc
    except pd.errors.ParserError as exc:
        raise AppError("INVALID_CSV", "The file is not a valid CSV.") from exc
    except ValueError as exc:
        raise AppError(
            "INVALID_FILE",
            "The file could not be read. Check that it is a valid .csv or .xlsx file.",
        ) from exc


def validate_frame(df: pd.DataFrame) -> None:
    """Apply the dataset-level checks the original loader performed."""
    if df is None or df.empty:
        raise AppError("EMPTY_DATASET", "Uploaded file is empty.")

    if df.columns.duplicated().any():
        dupes = sorted(set(str(c) for c in df.columns[df.columns.duplicated()]))
        raise AppError(
            "DUPLICATE_COLUMNS",
            f"Duplicate column names found: {', '.join(dupes)}. Fix the file and upload it again.",
        )
