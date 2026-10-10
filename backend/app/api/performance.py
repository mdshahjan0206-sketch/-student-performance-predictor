"""Performance/processing-time monitor endpoint (feature 16).

Returns the real, measured durations logged by app.core.timing.track() as
operations actually run. There is no synthetic or estimated data here — an
empty log just means nothing has been processed yet in this server session.
"""

from fastapi import APIRouter, Query

from app.services.store import store

router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.get("")
def performance_log(limit: int = Query(50, ge=1, le=300)):
    return {"success": True, "entries": store.get_performance_log(limit)}
