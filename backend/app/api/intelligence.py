"""Dataset Intelligence and Enhanced Recommendation API endpoints.

Serves the post-upload dataset intelligence, feature detail, data quality,
correlation/relationship, and transparent model recommendation payloads.
All endpoints reuse existing dataset records and caching."""

from fastapi import APIRouter, Query

from app.services import intelligence_service

router = APIRouter(prefix="/api/intelligence", tags=["intelligence"])


@router.get("/{dataset_id}")
def dataset_intelligence(dataset_id: str):
    """Full dataset intelligence payload for the upload page."""
    return {"success": True, "intelligence": intelligence_service.dataset_intelligence(dataset_id)}


@router.get("/{dataset_id}/recommendation")
def model_recommendation(
    dataset_id: str,
    target: str = Query(...),
    problem_type: str = Query(...),
):
    """Enhanced model recommendation with evidence and workflow notes."""
    return {
        "success": True,
        "recommendation": intelligence_service.enhanced_recommendation(
            dataset_id, target, problem_type
        ),
    }


@router.get("/{dataset_id}/correlation")
def feature_correlation(
    dataset_id: str,
    x: str = Query(""),
    y: str = Query(""),
):
    """Pairwise or full correlation data."""
    return {
        "success": True,
        **intelligence_service.feature_correlation(dataset_id, x, y),
    }
