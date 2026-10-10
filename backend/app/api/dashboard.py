"""Dashboard endpoint: everything the Student Performance Dashboard needs in
one call, aggregated from the dataset, model and prediction records that
already exist — nothing here is computed twice or invented."""

from fastapi import APIRouter, Depends

from app.api.deps import get_session_id
from app.services import dashboard_service

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/{dataset_id}")
def dashboard(dataset_id: str, session_id: str = Depends(get_session_id)):
    return {"success": True, "dashboard": dashboard_service.dashboard_summary(dataset_id, session_id)}
