"""CSV upload validation: decoding, delimiters, structure and error messages.

Self-contained: every file is generated here from a fixed seed, so the suite
needs no sample-data folder and no real student records. Run from backend/:

    python -m pytest tests/test_upload_validation.py -q
"""

import io
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import dataset_service  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

STUDENT_COLUMNS = [
    "study_hours", "attendance", "sleep_hours", "internet_usage",
    "assignments_completed", "previous_score", "exam_score", "placement_status",
]


def student_csv(rows: int = 200, seed: int = 7) -> str:
    """Synthetic data in the student schema (integers, a float target and a
    text label), generated deterministically."""
    rng = random.Random(seed)
    lines = [",".join(STUDENT_COLUMNS)]
    for _ in range(rows):
        score = round(rng.uniform(30, 100), 1)
        lines.append(",".join(map(str, [
            rng.randint(1, 10), rng.randint(40, 100), rng.randint(4, 9), rng.randint(1, 8),
            rng.randint(0, 10), rng.randint(30, 95), score,
            "Placed" if score > 55 else "Not Placed",
        ])))
    return "\n".join(lines) + "\n"


def upload(data: bytes, filename: str = "students.csv", field: str = "file", session: str = "upload-tests"):
    return client.post(
        "/api/dataset/upload",
        files={field: (filename, data, "text/csv")},
        headers={"X-Session-Id": session},
    )


def error_of(res):
    body = res.json()
    assert body["success"] is False
    return body["error"]


# --- 1 & 2. valid files ------------------------------------------------

def test_valid_generic_schema_is_accepted():
    data = b"hours,attendance,grade\n3,80,B\n5,92,A\n1,50,C\n"
    res = upload(data)
    assert res.status_code == 200, res.text
    ds = res.json()["dataset"]
    assert ds["num_rows"] == 3 and ds["column_names"] == ["hours", "attendance", "grade"]


def test_student_schema_is_accepted_and_values_are_untouched():
    text = student_csv(200)
    res = upload(text.encode())
    assert res.status_code == 200, res.text
    ds = res.json()["dataset"]
    assert ds["column_names"] == STUDENT_COLUMNS
    assert ds["num_rows"] == 200 and ds["num_cols"] == 8
    assert "exam_score" in ds["detected_targets"] or "placement_status" in ds["detected_targets"]

    first = text.splitlines()[1].split(",")
    page = client.get(f"/api/dataset/{ds['dataset_id']}/preview?page=1&page_size=1").json()
    row = page["rows"][0]
    assert str(row["study_hours"]) == first[0]
    assert float(row["exam_score"]) == float(first[6])
    assert row["placement_status"] == first[7]


# --- 3. missing columns --------------------------------------------------

def test_single_column_file_is_rejected_with_guidance():
    res = upload(b"study_hours\n3\n4\n5\n")
    assert res.status_code == 400
    err = error_of(res)
    assert err["code"] == "TOO_FEW_COLUMNS"
    assert "separator" in err["message"]


def test_missing_target_column_is_reported_when_training():
    # Upload is schema-agnostic by design (any tabular dataset may be analysed);
    # the required outcome column is enforced where it is used, at training.
    res = upload(b"height_cm,weight_kg\n170,60\n160,55\n180,80\n", filename="body.csv")
    assert res.status_code == 200
    ds = res.json()["dataset"]
    assert ds["detected_targets"] == []
    train = client.post("/api/model/train", json={
        "dataset_id": ds["dataset_id"], "target_column": "exam_score",
        "problem_type": "regression", "model_choice": "Linear Regression",
    })
    assert train.status_code == 400
    assert error_of(train)["code"] == "INVALID_TARGET"


# --- 4. empty and malformed ----------------------------------------------

@pytest.mark.parametrize("payload", [b"", b"   \n\n  "])
def test_empty_files_are_rejected(payload):
    res = upload(payload)
    assert res.status_code == 400
    assert error_of(res)["code"] in ("EMPTY_FILE", "EMPTY_DATASET")


def test_header_only_file_says_there_are_no_rows():
    res = upload(b"study_hours,attendance,exam_score\n")
    assert res.status_code == 400
    err = error_of(res)
    assert err["code"] == "EMPTY_DATASET"
    assert "no data rows" in err["message"]


def test_row_with_extra_values_names_the_line():
    lines = student_csv(20).splitlines()
    lines[5] = lines[5] + ",9,9"
    res = upload(("\n".join(lines) + "\n").encode())
    assert res.status_code == 400
    err = error_of(res)
    assert err["code"] == "INVALID_CSV"
    assert "Line 6" in err["message"]
    assert err["details"] == {"line": 6, "expected_columns": 8, "found_columns": 10}


def test_extra_values_in_first_data_row_do_not_shift_columns():
    # pandas alone would silently use the leading columns as an index here.
    lines = student_csv(10).splitlines()
    lines[1] = lines[1] + ",oops"
    res = upload(("\n".join(lines) + "\n").encode())
    assert res.status_code == 400
    assert error_of(res)["code"] == "INVALID_CSV"


def test_html_error_page_saved_as_csv_is_recognised():
    res = upload(b"<!DOCTYPE html><html><body>502 Bad Gateway</body></html>")
    assert res.status_code == 400
    assert error_of(res)["code"] == "NOT_A_CSV"


def test_binary_file_with_csv_name_is_recognised():
    res = upload(b"PK\x03\x04\x14\x00\x06\x00" + b"\x00" * 64)
    assert res.status_code == 400
    assert error_of(res)["code"] == "NOT_A_CSV"


def test_unsupported_extension_still_rejected():
    res = upload(b"a,b\n1,2\n", filename="notes.txt")
    assert res.status_code == 400
    assert error_of(res)["code"] == "UNSUPPORTED_FORMAT"


# --- 5. encodings, delimiters, headers -------------------------------------

def test_utf8_bom_header_is_clean():
    res = upload(b"\xef\xbb\xbf" + student_csv(50).encode())
    assert res.status_code == 200, res.text
    names = res.json()["dataset"]["column_names"]
    assert names == STUDENT_COLUMNS
    assert not names[0].startswith("﻿")


def test_utf16_csv_is_decoded():
    res = upload(student_csv(50).encode("utf-16"))
    assert res.status_code == 200, res.text
    assert res.json()["dataset"]["column_names"] == STUDENT_COLUMNS


def test_windows_1252_text_is_decoded_without_changing_values():
    data = "name,score\nJosé,90\nZoë,85\n".encode("cp1252")
    res = upload(data)
    assert res.status_code == 200, res.text
    ds = res.json()["dataset"]
    page = client.get(f"/api/dataset/{ds['dataset_id']}/preview?page=1&page_size=5").json()
    assert [r["name"] for r in page["rows"]] == ["José", "Zoë"]


@pytest.mark.parametrize("sep", [";", "\t", "|"])
def test_other_delimiters_are_detected(sep):
    res = upload(student_csv(30).replace(",", sep).encode())
    assert res.status_code == 200, res.text
    ds = res.json()["dataset"]
    assert ds["column_names"] == STUDENT_COLUMNS and ds["num_rows"] == 30


def test_quoted_comma_is_not_a_delimiter():
    res = upload(b'name,score\n"Doe, Jane",90\n"Roe, Rick",80\n')
    assert res.status_code == 200, res.text
    assert res.json()["dataset"]["num_rows"] == 2


def test_header_whitespace_is_trimmed_but_casing_is_kept():
    res = upload(b" Study_Hours , attendance \n3,80\n5,90\n")
    assert res.status_code == 200, res.text
    assert res.json()["dataset"]["column_names"] == ["Study_Hours", "attendance"]


def test_crlf_line_endings():
    res = upload(student_csv(25).replace("\n", "\r\n").encode())
    assert res.status_code == 200, res.text
    assert res.json()["dataset"]["num_rows"] == 25


# --- 6. request shape ---------------------------------------------------

def test_wrong_multipart_field_name_names_the_field():
    res = upload(student_csv(10).encode(), field="dataset")
    assert res.status_code == 422
    err = error_of(res)
    assert err["code"] == "INVALID_REQUEST" and "file" in err["message"]


def test_request_without_any_file_is_rejected():
    res = client.post("/api/dataset/upload", headers={"X-Session-Id": "upload-tests"})
    assert res.status_code == 422
    assert error_of(res)["code"] == "INVALID_REQUEST"


# --- 7. processing failure ----------------------------------------------

def test_processing_exception_returns_clean_error_without_leaking_details(monkeypatch):
    def boom(_df):
        raise RuntimeError("secret /srv/internal/path exploded")

    monkeypatch.setattr(dataset_service.eda, "compute_dataset_metadata", boom)
    res = upload(student_csv(20).encode())
    assert res.status_code == 500
    err = error_of(res)
    assert err["code"] == "INTERNAL_ERROR"
    assert "reference" in err["message"]
    assert "secret" not in res.text and "/srv/internal" not in res.text and "Traceback" not in res.text


def test_parse_bytes_directly_for_non_csv_exceptions():
    from app.core.errors import AppError
    from app.utils.file_utils import parse_tabular_bytes

    with pytest.raises(AppError) as info:
        parse_tabular_bytes(io.BytesIO(b"not a zip").getvalue(), "xlsx")
    assert info.value.code in ("INVALID_FILE",)
