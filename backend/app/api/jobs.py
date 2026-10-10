"""Background job polling, global session status, and health."""

import time

from fastapi import APIRouter, Depends

from app.api.deps import get_session_id
from app.core.errors import AppError
from app.services.store import store

router = APIRouter(prefix="/api", tags=["jobs"])

_STARTED_AT = time.time()


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = store.get_job(job_id)
    if job is None:
        raise AppError(
            "JOB_NOT_FOUND",
            "That training job is no longer available. Start training again.",
            status_code=404,
        )
    return {"success": True, **job.to_dict()}


@router.get("/session/status")
def session_status(session_id: str = Depends(get_session_id)):
    """Everything the global status bar shows: active dataset, row count,
    model status, workflow stage and last operation."""
    session = store.session(session_id)
    body = session.to_dict()

    dataset = store.get_dataset(session.active_dataset_id) if session.active_dataset_id else None
    if dataset is not None:
        body["dataset"] = {
            "dataset_id": dataset.id,
            "filename": dataset.filename,
            "num_rows": dataset.num_rows,
            "num_cols": dataset.num_cols,
        }
    else:
        body["dataset"] = None
        if session.active_dataset_id:
            # The dataset went away (server restart); stop advertising it.
            session.active_dataset_id = ""
            session.active_model_id = ""
            session.stage = "upload"

    model = store.get_model(session.active_model_id) if session.active_model_id else None
    if model is not None:
        body["model"] = {
            "model_id": model.id,
            "model_choice": model.model_choice,
            "problem_type": model.problem_type,
            "target_column": model.target_column,
            "dataset_id": model.dataset_id,
        }
    else:
        body["model"] = None
        session.active_model_id = ""

    return {"success": True, **body}


@router.get("/health")
def health():
    return {
        "success": True,
        "status": "ok",
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
        "datasets": len(store.datasets),
        "models": len(store.models),
    }
