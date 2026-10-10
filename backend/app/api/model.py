"""Model endpoints: training configuration, pre-flight validation, background
training submission, and trained-model details."""

from fastapi import APIRouter, Depends

from app.api.deps import get_session_id
from app.schemas.models import PreflightRequest, TargetAnalysisRequest, TrainRequest
from app.services import model_service

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/config/{dataset_id}")
def training_config(dataset_id: str):
    return {"success": True, **model_service.training_config(dataset_id)}


@router.post("/target-analysis")
def target_analysis(body: TargetAnalysisRequest):
    return {
        "success": True,
        "analysis": model_service.target_analysis(
            body.dataset_id, body.target_column, body.problem_override
        ),
    }


@router.post("/preflight")
def preflight(body: PreflightRequest):
    return {
        "success": True,
        "validation": model_service.preflight(
            body.dataset_id, body.target_column, body.problem_type, body.model_choice
        ),
    }


@router.post("/train")
def train(body: TrainRequest, session_id: str = Depends(get_session_id)):
    job = model_service.submit_training(
        body.dataset_id, body.target_column, body.problem_type, body.model_choice, session_id
    )
    return {"success": True, "job_id": job.id, "status": job.status}


@router.get("/history/{dataset_id}")
def history(dataset_id: str):
    return {"success": True, "models": model_service.training_history(dataset_id)}


@router.get("/{model_id}")
def model_details(model_id: str):
    return {"success": True, "model": model_service.model_details(model_id)}


@router.get("/{model_id}/metrics")
def model_metrics(model_id: str):
    return {"success": True, **model_service.model_metrics(model_id)}


@router.get("/{model_id}/feature-schema")
def feature_schema(model_id: str):
    return {"success": True, **model_service.feature_schema(model_id)}
