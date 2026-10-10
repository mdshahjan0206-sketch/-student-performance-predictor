"""Backend-managed state, replacing Streamlit's st.session_state.

Four registries live here:

  datasets     dataset_id  -> uploaded file + parsed frame + metadata
  models       model_id    -> fitted pipeline + training metadata + results
  jobs         job_id      -> background training job status
  predictions  prediction_id -> a stored batch-prediction result frame

plus a small per-client session record holding the active dataset, the active
model and the workflow stage.

THE INVALIDATION RULE, carried over from the original session_manager:
registering a new active dataset always clears the active model, because a
model fitted on the previous data is not valid for the new data. On top of
that, every model carries the dataset_id it was trained on, and the prediction
endpoints refuse to run a model whose dataset_id is not the session's active
dataset. A model trained on dataset A therefore cannot be used after dataset B
is uploaded, by any route.

Large objects are kept in bounded in-memory caches backed by disk: parsed
frames can always be rebuilt from the stored upload, and pipelines from the
stored joblib file, so the caches can evict without losing state.
"""

import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import joblib
import pandas as pd

from app.core.config import (
    DATASET_CACHE_SIZE,
    MODEL_CACHE_SIZE,
    MODELS_DIR,
    OUTPUTS_DIR,
    UPLOADS_DIR,
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class DatasetRecord:
    id: str
    filename: str
    extension: str
    storage_path: str
    file_size: int
    num_rows: int
    num_cols: int
    column_names: list
    dtypes: dict
    missing_summary: dict
    duplicate_count: int
    detected_targets: list
    refined: bool = False
    created_at: float = field(default_factory=time.time)
    session_id: str = ""

    def to_summary(self) -> dict:
        total_missing = sum(self.missing_summary.values()) if isinstance(self.missing_summary, dict) else 0
        return {
            "dataset_id": self.id,
            "filename": self.filename,
            "file_size": self.file_size,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "column_names": self.column_names,
            "dtypes": self.dtypes,
            "missing_summary": self.missing_summary,
            "duplicate_count": self.duplicate_count,
            "detected_targets": self.detected_targets,
            "refined": self.refined,
            "missing_values_detected": total_missing,
            "missing_values_handled": total_missing if self.refined else 0,
            "missing_values_remaining": 0 if self.refined else total_missing,
            "uploaded_at": self.created_at,
        }


@dataclass
class ModelRecord:
    id: str
    dataset_id: str
    dataset_filename: str
    target_column: str
    problem_type: str
    model_choice: str
    estimator_name: str
    feature_columns: list
    training_time: float
    n_train: int
    n_test: int
    metrics: dict
    charts: dict
    warnings: list
    storage_path: str
    target_missing_excluded: int = 0
    created_at: float = field(default_factory=time.time)

    def to_info(self) -> dict:
        return {
            "model_id": self.id,
            "dataset_id": self.dataset_id,
            "dataset_filename": self.dataset_filename,
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "model_choice": self.model_choice,
            "estimator_name": self.estimator_name,
            "feature_columns": self.feature_columns,
            "feature_count": len(self.feature_columns),
            "training_time": self.training_time,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "target_missing_excluded": self.target_missing_excluded,
            "trained_at": self.created_at,
        }


@dataclass
class JobRecord:
    id: str
    kind: str
    status: str = "queued"
    dataset_id: str = ""
    model_id: str = ""
    error_code: str = ""
    error_message: str = ""
    warnings: list = field(default_factory=list)
    recommended_models: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        body = {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed": round(self.updated_at - self.created_at, 3),
        }
        if self.model_id:
            body["model_id"] = self.model_id
        if self.warnings:
            body["warnings"] = self.warnings
        if self.status == "failed":
            body["error"] = {"code": self.error_code, "message": self.error_message}
            if self.recommended_models:
                body["error"]["recommended_models"] = self.recommended_models
        return body


@dataclass
class PredictionRecord:
    id: str
    model_id: str
    dataset_id: str
    kind: str
    row_count: int
    columns: list
    storage_path: str
    has_confidence: bool
    problem_type: str
    warnings: list = field(default_factory=list)
    distribution: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionRecord:
    id: str
    active_dataset_id: str = ""
    active_model_id: str = ""
    stage: str = "upload"
    last_operation: str = ""
    last_operation_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "session_id": self.id,
            "active_dataset_id": self.active_dataset_id,
            "active_model_id": self.active_model_id,
            "stage": self.stage,
            "last_operation": self.last_operation,
            "last_operation_at": self.last_operation_at or None,
        }


class _LRU(OrderedDict):
    """Smallest possible bounded cache. Values are always reconstructible
    from disk, so eviction is safe."""

    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        while len(self) > self.maxsize:
            self.popitem(last=False)

    def take(self, key):
        if key not in self:
            return None
        self.move_to_end(key)
        return self[key]


class Store:
    """Thread-safe registry. Training runs on a worker thread, so every
    mutation is guarded."""

    def __init__(self):
        self._lock = threading.RLock()
        self.datasets: dict[str, DatasetRecord] = {}
        self.models: dict[str, ModelRecord] = {}
        self.jobs: dict[str, JobRecord] = {}
        self.predictions: dict[str, PredictionRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self._frames = _LRU(DATASET_CACHE_SIZE)
        self._pipelines = _LRU(MODEL_CACHE_SIZE)
        self._model_extras: dict[str, dict] = {}
        self._analysis_cache: dict[str, dict] = {}
        self._performance_log: list[dict] = []
        self._load_datasets_from_disk()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def session(self, session_id: str) -> SessionRecord:
        with self._lock:
            record = self.sessions.get(session_id)
            if record is None:
                record = SessionRecord(id=session_id)
                self.sessions[session_id] = record
            return record

    def touch(self, session_id: str, operation: str, stage: str = ""):
        with self._lock:
            record = self.session(session_id)
            record.last_operation = operation
            record.last_operation_at = time.time()
            if stage:
                record.stage = stage
            return record

    def set_active_dataset(self, session_id: str, dataset_id: str):
        """Register the active dataset. ALWAYS clears the active model, since
        it was fitted against different data — the same rule the original
        session_manager.set_dataset() enforced."""
        with self._lock:
            record = self.session(session_id)
            record.active_dataset_id = dataset_id
            record.active_model_id = ""
            record.stage = "analyze"
            return record

    def set_active_model(self, session_id: str, model_id: str):
        with self._lock:
            record = self.session(session_id)
            record.active_model_id = model_id
            record.stage = "predict"
            return record

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def add_dataset(self, record: DatasetRecord, df: pd.DataFrame):
        with self._lock:
            self.datasets[record.id] = record
            self._frames.put(record.id, df)
            self._write_sidecar(record)

    def get_dataset(self, dataset_id: str):
        with self._lock:
            return self.datasets.get(dataset_id)

    def get_frame(self, dataset_id: str):
        """Return the parsed DataFrame, re-reading the stored upload only if
        it has been evicted from the cache."""
        with self._lock:
            cached = self._frames.take(dataset_id)
            if cached is not None:
                return cached
            record = self.datasets.get(dataset_id)
            if record is None or not os.path.exists(record.storage_path):
                return None
            df = _read_frame(record.storage_path, record.extension)
            self._frames.put(dataset_id, df)
            return df

    # ------------------------------------------------------------------
    # Analysis cache (per dataset, cleared when the dataset goes away)
    # ------------------------------------------------------------------
    def cached_analysis(self, dataset_id: str, key: str):
        with self._lock:
            return self._analysis_cache.get(dataset_id, {}).get(key)

    def cache_analysis(self, dataset_id: str, key: str, value):
        with self._lock:
            self._analysis_cache.setdefault(dataset_id, {})[key] = value
            return value

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def add_model(self, record: ModelRecord, pipeline, extras: dict):
        with self._lock:
            self.models[record.id] = record
            self._pipelines.put(record.id, pipeline)
            self._model_extras[record.id] = extras
            joblib.dump({"pipeline": pipeline, **extras}, record.storage_path)

    def get_model(self, model_id: str):
        with self._lock:
            return self.models.get(model_id)

    def get_pipeline(self, model_id: str):
        with self._lock:
            cached = self._pipelines.take(model_id)
            if cached is not None:
                return cached
            record = self.models.get(model_id)
            if record is None or not os.path.exists(record.storage_path):
                return None
            payload = joblib.load(record.storage_path)
            pipeline = payload["pipeline"]
            self._pipelines.put(model_id, pipeline)
            self._model_extras.setdefault(model_id, {
                k: v for k, v in payload.items() if k != "pipeline"
            })
            return pipeline

    def get_model_extras(self, model_id: str) -> dict:
        with self._lock:
            if model_id in self._model_extras:
                return self._model_extras[model_id]
            record = self.models.get(model_id)
            if record is None or not os.path.exists(record.storage_path):
                return {}
            payload = joblib.load(record.storage_path)
            extras = {k: v for k, v in payload.items() if k != "pipeline"}
            self._model_extras[model_id] = extras
            return extras

    def models_for_dataset(self, dataset_id: str) -> list:
        with self._lock:
            return sorted(
                [m for m in self.models.values() if m.dataset_id == dataset_id],
                key=lambda m: m.created_at,
                reverse=True,
            )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    def add_job(self, record: JobRecord):
        with self._lock:
            self.jobs[record.id] = record
            return record

    def get_job(self, job_id: str):
        with self._lock:
            return self.jobs.get(job_id)

    def update_job(self, job_id: str, **changes):
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = time.time()
            return job

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def add_prediction(self, record: PredictionRecord, df: pd.DataFrame):
        with self._lock:
            df.to_csv(record.storage_path, index=False)
            self.predictions[record.id] = record

    def get_prediction(self, prediction_id: str):
        with self._lock:
            return self.predictions.get(prediction_id)

    def get_prediction_frame(self, prediction_id: str):
        record = self.get_prediction(prediction_id)
        if record is None or not os.path.exists(record.storage_path):
            return None
        return pd.read_csv(record.storage_path)

    def predictions_for_dataset(self, dataset_id: str) -> list:
        with self._lock:
            return sorted(
                [p for p in self.predictions.values() if p.dataset_id == dataset_id],
                key=lambda p: p.created_at,
                reverse=True,
            )

    # ------------------------------------------------------------------
    # Performance monitor: a bounded, in-memory log of actual measured
    # operation durations. Nothing here is estimated or synthetic — each
    # entry is written by the operation itself as it completes (see
    # app/core/timing.py), so the feature can only ever show real numbers.
    # ------------------------------------------------------------------
    PERFORMANCE_LOG_LIMIT = 300

    def log_performance(self, operation: str, duration: float, status: str = "success", meta: dict | None = None):
        with self._lock:
            entry = {
                "operation": operation,
                "duration": round(float(duration), 4),
                "status": status,
                "timestamp": time.time(),
            }
            if meta:
                entry["meta"] = meta
            self._performance_log.append(entry)
            if len(self._performance_log) > self.PERFORMANCE_LOG_LIMIT:
                self._performance_log = self._performance_log[-self.PERFORMANCE_LOG_LIMIT:]

    def get_performance_log(self, limit: int = 100) -> list:
        with self._lock:
            return list(reversed(self._performance_log[-limit:]))

    # ------------------------------------------------------------------
    def _write_sidecar(self, record: DatasetRecord):
        """Persist dataset metadata next to the upload so a server restart
        can still describe files that are on disk."""
        path = os.path.join(UPLOADS_DIR, f"{record.id}.meta.json")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(record.to_summary(), handle)
        except OSError:
            pass

    def _load_datasets_from_disk(self):
        """Restore dataset records from persisted sidecar metadata files."""
        try:
            if not os.path.isdir(UPLOADS_DIR):
                return
            for fname in os.listdir(UPLOADS_DIR):
                if fname.endswith(".meta.json"):
                    meta_path = os.path.join(UPLOADS_DIR, fname)
                    with open(meta_path, "r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    did = data.get("dataset_id")
                    if not did or did in self.datasets:
                        continue
                    csv_path = os.path.join(UPLOADS_DIR, f"{did}.csv")
                    xlsx_path = os.path.join(UPLOADS_DIR, f"{did}.xlsx")
                    if os.path.isfile(csv_path):
                        storage_path = csv_path
                        ext = "csv"
                    elif os.path.isfile(xlsx_path):
                        storage_path = xlsx_path
                        ext = "xlsx"
                    else:
                        continue
                    missing_summary = data.get("missing_summary", {})
                    total_missing = sum(missing_summary.values()) if isinstance(missing_summary, dict) else 0
                    rec = DatasetRecord(
                        id=did,
                        filename=data.get("filename", f"{did}.{ext}"),
                        extension=ext,
                        storage_path=storage_path,
                        file_size=data.get("file_size", 0),
                        num_rows=data.get("num_rows", 0),
                        num_cols=data.get("num_cols", 0),
                        column_names=data.get("column_names", []),
                        dtypes=data.get("dtypes", {}),
                        missing_summary=missing_summary,
                        duplicate_count=data.get("duplicate_count", 0),
                        detected_targets=data.get("detected_targets", []),
                        refined=data.get("refined", total_missing == 0),
                        created_at=data.get("uploaded_at", time.time()),
                    )
                    self.datasets[did] = rec
        except Exception:
            pass


def _read_frame(path: str, extension: str) -> pd.DataFrame:
    if extension == "csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def model_path(model_id: str) -> str:
    return os.path.join(MODELS_DIR, f"{model_id}.joblib")


def output_path(prediction_id: str) -> str:
    return os.path.join(OUTPUTS_DIR, f"{prediction_id}.csv")


store = Store()
