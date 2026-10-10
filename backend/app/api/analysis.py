"""Data Analysis endpoints. Each returns chart-ready data plus an
`available` flag so the client can show a useful message instead of an error
when a chart does not apply to the chosen columns."""

from fastapi import APIRouter, Query

from app.services import analysis_service

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.get("/{dataset_id}/overview")
def overview(dataset_id: str):
    return {"success": True, "overview": analysis_service.overview(dataset_id)}


@router.get("/{dataset_id}/histogram")
def histogram(dataset_id: str, column: str = Query(...)):
    return {"success": True, **analysis_service.histogram(dataset_id, column)}


@router.get("/{dataset_id}/boxplot")
def boxplot(dataset_id: str, column: str = Query(...)):
    return {"success": True, **analysis_service.boxplot(dataset_id, column)}


@router.get("/{dataset_id}/correlation")
def correlation(dataset_id: str):
    return {"success": True, **analysis_service.correlation(dataset_id)}


@router.get("/{dataset_id}/missing")
def missing(dataset_id: str):
    return {"success": True, **analysis_service.missing(dataset_id)}


@router.get("/{dataset_id}/cleaning-report")
def cleaning_report(dataset_id: str):
    from fastapi.responses import Response
    csv_content = analysis_service.cleaning_report(dataset_id)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="cleaning_report_{dataset_id}.csv"'},
    )


@router.get("/{dataset_id}/scatter")
def scatter(dataset_id: str, x: str = Query(...), y: str = Query(...)):
    return {"success": True, **analysis_service.scatter(dataset_id, x, y)}
