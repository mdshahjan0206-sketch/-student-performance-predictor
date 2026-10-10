"""Prediction endpoints: single record, batch file, paged results, CSV export."""

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import Response

from app.api.deps import get_session_id
from app.core.config import MAX_UPLOAD_BYTES
from app.core.errors import AppError
from app.schemas.models import SinglePredictionRequest
from app.services import prediction_service

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict/single")
def predict_single(body: SinglePredictionRequest, session_id: str = Depends(get_session_id)):
    return {
        "success": True,
        "result": prediction_service.run_single(body.model_id, body.inputs, session_id),
    }


@router.post("/predict/batch")
async def predict_batch(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
):
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise AppError("FILE_TOO_LARGE", f"The file is larger than the {limit_mb} MB limit.", status_code=413)

    result = prediction_service.run_batch(model_id, contents, file.filename or "batch.csv", session_id)
    first_page = prediction_service.prediction_page(result["prediction_id"], 1, 25)
    return {"success": True, "batch": result, "results": first_page}


@router.get("/predictions/{prediction_id}")
def prediction_results(
    prediction_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
):
    return {"success": True, **prediction_service.prediction_page(prediction_id, page, page_size)}


@router.get("/predictions/history/{dataset_id}")
def prediction_history(dataset_id: str):
    return {"success": True, "history": prediction_service.prediction_history(dataset_id)}


@router.get("/predictions/{prediction_id}/download")
def download_predictions(prediction_id: str):
    csv_bytes = prediction_service.prediction_csv(prediction_id)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="prediction_results.csv"'},
    )
