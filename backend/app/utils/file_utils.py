"""Upload security and tabular parsing.

Uploads are never trusted: the extension is checked against an allow-list,
the size is capped, and the file is written under a server-generated name
inside a fixed directory, so a crafted filename can neither escape the
uploads folder nor be interpreted as anything executable. The original
filename is kept only as a display label.
"""

import codecs
import csv
import io
import os
import re
import zipfile

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


# --- CSV text handling -------------------------------------------------
# Spreadsheet exports arrive in more shapes than "UTF-8, comma separated":
# Excel on Windows writes UTF-8 with a BOM or cp1252, European locales use
# semicolons, and "Unicode text" exports are UTF-16. Each is decoded or
# detected here so a legitimate file is read as written, and a file that is
# genuinely broken is rejected with the line that is wrong.

_CANDIDATE_DELIMITERS = (",", ";", "\t", "|")


def decode_csv_bytes(file_bytes: bytes) -> str:
    """Decode uploaded CSV bytes to text without altering any value.

    A byte-order mark selects UTF-32/UTF-16/UTF-8 (the BOM itself is dropped).
    Without one, UTF-8 is tried first, then Windows-1252 and finally Latin-1,
    which is how spreadsheet tools write legacy "ANSI" exports.
    """
    if file_bytes.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        encoding = "utf-32"
    elif file_bytes.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding = "utf-16"
    else:
        encoding = "utf-8-sig"

    if encoding == "utf-8-sig":
        # A real text file has no NUL bytes. Their presence (or a zip header)
        # means a binary file, for example an .xlsx saved with a .csv name.
        if file_bytes.startswith(b"PK\x03\x04") or b"\x00" in file_bytes[:8192]:
            raise AppError(
                "NOT_A_CSV",
                "This file is binary, not text. If it is an Excel workbook, upload it with the "
                ".xlsx extension or use File > Save As > CSV first.",
            )
        try:
            return file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            pass
        try:
            return file_bytes.decode("cp1252")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")

    try:
        return file_bytes.decode(encoding)
    except UnicodeDecodeError as exc:
        raise AppError(
            "INVALID_FILE",
            "The file starts like a Unicode text file but could not be decoded. "
            "Re-save it as CSV (UTF-8) and upload it again.",
        ) from exc


def detect_delimiter(text: str) -> str:
    """Pick the delimiter that splits the header line into the most columns.

    Falls back to a comma. Quoted fields are respected, so a comma inside
    "Name, Surname" does not count as a separator.
    """
    header = ""
    for line in io.StringIO(text):
        if line.strip():
            header = line
            break
    best, best_count = ",", 1
    for candidate in _CANDIDATE_DELIMITERS:
        try:
            fields = next(csv.reader([header], delimiter=candidate), [])
        except csv.Error:
            continue
        if len(fields) > best_count:
            best, best_count = candidate, len(fields)
    return best


def _check_csv_structure(text: str, delimiter: str) -> None:
    """Reject rows that carry more values than the header has columns.

    pandas would either raise an unhelpful error for these or, when it is the
    first data row, silently treat the leading columns as an index and shift
    every value one column to the left. A row with *fewer* values is allowed
    and read as missing, exactly as before.
    """
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    width = None
    try:
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue  # blank line
            if width is None:
                width = len(row)
                continue
            if len(row) > width:
                raise AppError(
                    "INVALID_CSV",
                    f"Line {reader.line_num} has {len(row)} values but the header has {width} columns. "
                    "Check for an extra comma, an unquoted comma inside a value, or a trailing separator, "
                    "then upload the file again.",
                    details={"line": reader.line_num, "expected_columns": width, "found_columns": len(row)},
                )
    except csv.Error as exc:
        raise AppError(
            "INVALID_CSV",
            f"The file could not be read as CSV near line {reader.line_num}. "
            "Check for an unclosed quotation mark or a corrupted row.",
            details={"line": reader.line_num},
        ) from exc


def _parse_csv(file_bytes: bytes) -> pd.DataFrame:
    text = decode_csv_bytes(file_bytes)
    if not text.strip():
        raise AppError("EMPTY_FILE", "The file is empty or corrupted.")

    if text.lstrip()[:200].lower().startswith(("<!doctype html", "<html", "<?xml")):
        raise AppError(
            "NOT_A_CSV",
            "This file contains a web page or XML, not CSV data. Export the dataset as a "
            "real CSV file and upload that instead.",
        )

    delimiter = detect_delimiter(text)
    _check_csv_structure(text, delimiter)
    # index_col=False stops pandas from promoting leading columns to an index.
    return pd.read_csv(
        io.StringIO(text),
        sep=delimiter,
        index_col=False,
        na_values=MISSING_SENTINELS,
        keep_default_na=True,
    )


def parse_tabular_bytes(file_bytes: bytes, file_ext: str) -> pd.DataFrame:
    """Parse CSV/Excel bytes into a DataFrame, translating pandas' parser
    failures into clean, user-facing messages and normalizing missing markers."""
    try:
        if file_ext == "csv":
            df = _parse_csv(file_bytes)
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), na_values=MISSING_SENTINELS, keep_default_na=True)
        # Header text only: stray spaces around a column name must not make
        # "attendance " and "attendance" different columns. Values and casing
        # are left exactly as uploaded.
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        return normalize_missing_markers(df)
    except AppError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise AppError("EMPTY_FILE", "The file is empty or corrupted.") from exc
    except pd.errors.ParserError as exc:
        raise AppError(
            "INVALID_CSV",
            "The file could not be parsed as CSV. Check that every row has the same number of "
            "values as the header and that quotation marks are closed.",
        ) from exc
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise AppError(
            "INVALID_FILE",
            "The Excel file could not be opened. It may be corrupted or password protected.",
        ) from exc
    except ValueError as exc:
        raise AppError(
            "INVALID_FILE",
            "The file could not be read. Check that it is a valid .csv or .xlsx file.",
        ) from exc


def validate_frame(df: pd.DataFrame) -> None:
    """Apply the dataset-level checks the original loader performed."""
    if df is not None and len(df.columns) > 0 and len(df) == 0:
        raise AppError(
            "EMPTY_DATASET",
            "The file has a header row but no data rows. Add at least one row of values and upload it again.",
        )
    if df is None or df.empty:
        raise AppError("EMPTY_DATASET", "Uploaded file is empty.")

    if len(df.columns) < 2:
        raise AppError(
            "TOO_FEW_COLUMNS",
            f"The file has only {len(df.columns)} column, so there is nothing to predict from. "
            "A dataset needs at least one input column and one outcome column. If your values are "
            "separated by something other than commas, check that every row uses the same separator.",
        )

    if df.columns.duplicated().any():
        dupes = sorted(set(str(c) for c in df.columns[df.columns.duplicated()]))
        raise AppError(
            "DUPLICATE_COLUMNS",
            f"Duplicate column names found: {', '.join(dupes)}. Fix the file and upload it again.",
        )
