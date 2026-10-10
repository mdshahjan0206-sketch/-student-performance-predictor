"""Explainability endpoints (feature 25). Nothing here trains, retrains or
mutates a model or its pipeline — see app.services.explain_service.
"""

from fastapi import APIRouter

from app.schemas.models import ExplainInstanceRequest
from app.services import explain_service

router = APIRouter(prefix="/api/explain", tags=["explainability"])


@router.get("/{model_id}/capability")
def get_capability(model_id: str):
    return {"success": True, "capability": explain_service.capability(model_id)}


@router.get("/{model_id}/global")
def get_global_importance(model_id: str):
    return {"success": True, **explain_service.global_importance(model_id)}


@router.post("/{model_id}/instance")
def post_instance(model_id: str, body: ExplainInstanceRequest):
    return {"success": True, **explain_service.explain_instance(model_id, body.inputs)}


@router.get("/prediction/{prediction_id}/{row_index}")
def get_for_prediction(prediction_id: str, row_index: int):
    return {"success": True, **explain_service.explain_for_prediction(prediction_id, row_index)}
