"""Student-level endpoints built on top of a stored batch prediction:
capability/at-risk metadata, the paginated explorer, individual profiles,
comparison and confidence summary. See app.services.student_service for the
derivation rules.
"""

from fastapi import APIRouter, Query
from fastapi.responses import Response

from app.schemas.models import StudentCompareRequest
from app.services import student_service

router = APIRouter(prefix="/api/students", tags=["students"])


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return parts or None


@router.get("/{prediction_id}/capability")
def get_capability(prediction_id: str):
    return {"success": True, "capability": student_service.capability(prediction_id)}


@router.get("/{prediction_id}")
def get_explorer_page(
    prediction_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    search: str = Query(""),
    risk: str | None = Query(None, description="Comma-separated: high,medium,low"),
    prediction: str | None = Query(None),
    confidence_min: float | None = Query(None, ge=0, le=1),
    confidence_max: float | None = Query(None, ge=0, le=1),
    sort_by: str | None = Query(None),
    sort_dir: str = Query("asc"),
):
    result = student_service.explorer_page(
        prediction_id,
        page=page,
        page_size=page_size,
        search=search,
        risk_filter=_split(risk),
        prediction_filter=prediction,
        confidence_min=confidence_min,
        confidence_max=confidence_max,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )
    return {"success": True, **result}


@router.get("/{prediction_id}/download")
def download_explorer(
    prediction_id: str,
    search: str = Query(""),
    risk: str | None = Query(None),
    prediction: str | None = Query(None),
    confidence_min: float | None = Query(None, ge=0, le=1),
    confidence_max: float | None = Query(None, ge=0, le=1),
):
    csv_bytes = student_service.explorer_csv(
        prediction_id,
        search=search,
        risk_filter=_split(risk),
        prediction_filter=prediction,
        confidence_min=confidence_min,
        confidence_max=confidence_max,
    )
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="student_results.csv"'},
    )


@router.get("/{prediction_id}/confidence-summary")
def get_confidence_summary(prediction_id: str):
    return {"success": True, "summary": student_service.confidence_summary(prediction_id)}


@router.get("/{prediction_id}/profile/{row_index}")
def get_profile(prediction_id: str, row_index: int):
    return {"success": True, "profile": student_service.student_profile(prediction_id, row_index)}


@router.post("/{prediction_id}/compare")
def post_compare(prediction_id: str, body: StudentCompareRequest):
    return {"success": True, "comparison": student_service.student_compare(prediction_id, body.row_indices)}
