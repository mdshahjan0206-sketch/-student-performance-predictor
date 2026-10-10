"""Measured timings against a running backend, using a 120,000-row dataset.

Every number printed is a real wall-clock measurement of an HTTP round trip to
uvicorn on localhost. Nothing here is estimated.

Usage:  python tests/benchmark.py
"""

import json
import os
import statistics
import sys
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
DATA_DIR = os.environ.get("DATA_DIR", "/home/claude/testdata")
SESSION = "benchmark"


def call(method, path, body=None, headers=None, raw=False):
    request = urllib.request.Request(f"{BASE}{path}", method=method)
    request.add_header("X-Session-Id", SESSION)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    data = None
    if body is not None and not raw:
        data = json.dumps(body).encode()
        request.add_header("Content-Type", "application/json")
    elif raw:
        data = body
    started = time.perf_counter()
    with urllib.request.urlopen(request, data) as response:
        payload = response.read()
    elapsed = (time.perf_counter() - started) * 1000
    try:
        return json.loads(payload), elapsed
    except json.JSONDecodeError:
        return payload, elapsed


def multipart(fields, files):
    boundary = "----edupredictbenchmark"
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    for name, (filename, content) in files.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: text/csv\r\n\r\n".encode()
            + content
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def timed(label, fn, repeats=3):
    timings = [fn()[1] for _ in range(repeats)]
    print(f"  {label:<44} {statistics.median(timings):8.1f} ms   (n={repeats})")
    return statistics.median(timings)


def main():
    path = os.path.join(DATA_DIR, "students_large.csv")
    with open(path, "rb") as handle:
        content = handle.read()

    rows = content.count(b"\n") - 1
    print(f"\nDataset: students_large.csv — {rows:,} rows, {len(content) / 1024 / 1024:.1f} MB\n")

    print("Upload and inspection")
    body, content_type = multipart({}, {"file": ("students_large.csv", content)})
    result, upload_ms = call("POST", "/api/dataset/upload", body, {"Content-Type": content_type}, raw=True)
    dataset_id = result["dataset"]["dataset_id"]
    print(f"  {'upload + parse + profile (cold)':<44} {upload_ms:8.1f} ms   (n=1)")

    timed("dataset summary (cached)", lambda: call("GET", f"/api/dataset/{dataset_id}/summary"))
    timed("preview page, 25 rows", lambda: call("GET", f"/api/dataset/{dataset_id}/preview?page=400&page_size=25"))
    timed("preview search across all columns", lambda: call("GET", f"/api/dataset/{dataset_id}/preview?search=Yes&page_size=25"))

    print("\nAnalysis (first call computes, later calls hit the cache)")
    _, cold = call("GET", f"/api/analysis/{dataset_id}/correlation")
    print(f"  {'correlation heatmap (cold)':<44} {cold:8.1f} ms   (n=1)")
    timed("correlation heatmap (cached)", lambda: call("GET", f"/api/analysis/{dataset_id}/correlation"))

    _, cold = call("GET", f"/api/analysis/{dataset_id}/histogram?column=exam_score")
    print(f"  {'histogram (cold)':<44} {cold:8.1f} ms   (n=1)")
    timed("histogram (cached)", lambda: call("GET", f"/api/analysis/{dataset_id}/histogram?column=exam_score"))

    _, cold = call("GET", f"/api/analysis/{dataset_id}/scatter?x=hours_studied&y=exam_score")
    print(f"  {'scatter, sampled to 5,000 points (cold)':<44} {cold:8.1f} ms   (n=1)")
    timed("box plot (cached)", lambda: call("GET", f"/api/analysis/{dataset_id}/boxplot?column=exam_score"))
    timed("overview stats (cached)", lambda: call("GET", f"/api/analysis/{dataset_id}/overview"))

    print("\nFiltering")
    timed(
        "filter preview, numeric + drop missing",
        lambda: call("POST", f"/api/dataset/{dataset_id}/filter", {
            "numeric": {"column": "attendance", "min": 70, "max": 100},
            "drop_missing": True,
            "limit": 50,
        }),
    )

    print("\nTraining (full dataset, never sampled)")
    for model_name in ("Linear Regression", "Random Forest Regressor"):
        started = time.perf_counter()
        job, _ = call("POST", "/api/model/train", {
            "dataset_id": dataset_id,
            "target_column": "exam_score",
            "problem_type": "regression",
            "model_choice": model_name,
        })
        job_id = job["job_id"]
        while True:
            state, _ = call("GET", f"/api/jobs/{job_id}")
            if state["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)
        total = (time.perf_counter() - started) * 1000
        if state["status"] != "completed":
            print(f"  {model_name:<44} FAILED: {state['error']['message']}")
            continue
        model_id = state["model_id"]
        model, _ = call("GET", f"/api/model/{model_id}")
        print(
            f"  {model_name + ' (end to end)':<44} {total:8.1f} ms   "
            f"fit {model['model']['training_time'] * 1000:.0f} ms, R² {model['model']['metrics']['r2']:.4f}"
        )

    print("\nPrediction")
    timed("feature schema for the form", lambda: call("GET", f"/api/model/{model_id}/feature-schema"))

    schema, _ = call("GET", f"/api/model/{model_id}/feature-schema")
    inputs = {field["name"]: field["default"] for field in schema["fields"]}
    timed(
        "single student prediction",
        lambda: call("POST", "/api/predict/single", {"model_id": model_id, "inputs": inputs}),
    )

    # score the whole 120k-row file as a batch
    import csv
    import io

    source = io.StringIO(content.decode())
    reader = csv.DictReader(source)
    features = schema["feature_columns"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=features)
    writer.writeheader()
    for row in reader:
        writer.writerow({key: row[key] for key in features})
    batch_bytes = buffer.getvalue().encode()

    body, content_type = multipart({"model_id": model_id}, {"file": ("batch_large.csv", batch_bytes)})
    result, batch_ms = call("POST", "/api/predict/batch", body, {"Content-Type": content_type}, raw=True)
    prediction_id = result["batch"]["prediction_id"]
    print(f"  {f'batch prediction, {result['batch']['row_count']:,} rows':<44} {batch_ms:8.1f} ms   (n=1)")

    timed("results page, 25 of 120,000 rows", lambda: call("GET", f"/api/predictions/{prediction_id}?page=2000&page_size=25"))
    _, download_ms = call("GET", f"/api/predictions/{prediction_id}/download")
    print(f"  {'CSV export of all rows':<44} {download_ms:8.1f} ms   (n=1)")

    print()


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError:
        print("Backend not reachable. Start it with: uvicorn app.main:app", file=sys.stderr)
        sys.exit(1)
