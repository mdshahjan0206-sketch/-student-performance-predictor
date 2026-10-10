"""Dataset endpoints: upload, summary, paged preview, statistics, filtering."""

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import Response

from app.api.deps import get_session_id
from app.core.config import MAX_UPLOAD_BYTES
from app.core.errors import AppError
from app.schemas.models import FilterRequest
from app.services import dataset_service
from app.services.store import store

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
):
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise AppError("FILE_TOO_LARGE", f"The file is larger than the {limit_mb} MB limit.", status_code=413)

    record = dataset_service.create_dataset(contents, file.filename or "dataset.csv", session_id)
    summary = dataset_service.dataset_summary(record.id)
    return {
        "success": True,
        "dataset": summary,
        "session": store.session(session_id).to_dict(),
    }


@router.get("/{dataset_id}/summary")
def dataset_summary(dataset_id: str):
    return {"success": True, "dataset": dataset_service.dataset_summary(dataset_id)}


@router.post("/{dataset_id}/refine")
def refine_dataset(dataset_id: str):
    return {"success": True, "refinement": dataset_service.refine_dataset(dataset_id)}


@router.get("/{dataset_id}/download-cleaned")
def download_cleaned(dataset_id: str):
    csv_bytes, filename = dataset_service.get_cleaned_csv(dataset_id)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{dataset_id}/preview")
def dataset_preview(
    dataset_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    search: str = Query("", max_length=200),
):
    return {"success": True, **dataset_service.dataset_preview(dataset_id, page, page_size, search)}


@router.get("/{dataset_id}/statistics")
def dataset_statistics(dataset_id: str):
    return {"success": True, **dataset_service.dataset_statistics(dataset_id)}


@router.get("/{dataset_id}/filter-options")
def filter_options(dataset_id: str):
    return {"success": True, **dataset_service.filter_options(dataset_id)}


@router.post("/{dataset_id}/filter")
def filter_preview(dataset_id: str, body: FilterRequest):
    """Exploratory only. The stored dataset is never replaced by the result."""
    filters = body.model_dump()
    limit = filters.pop("limit", 50)
    return {"success": True, **dataset_service.filter_preview(dataset_id, filters, limit=limit)}


@router.post("/{dataset_id}/filter/download")
def filter_download(dataset_id: str, body: FilterRequest):
    filters = body.model_dump()
    filters.pop("limit", None)
    csv_bytes = dataset_service.filter_csv(dataset_id, filters)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="filtered_preview.csv"'},
    )
