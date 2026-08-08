"""
GET /analytics -- token usage, latency, and cost analytics.
"""
from fastapi import APIRouter, HTTPException

from app.core.logging_config import get_logger
from app.models.schemas import AnalyticsSummary, ErrorResponse
from app.services import analytics_service

logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get(
    "",
    response_model=AnalyticsSummary,
    responses={500: {"model": ErrorResponse}},
    summary="Get token usage, latency, and cost analytics across all LLM calls",
)
async def get_analytics() -> AnalyticsSummary:
    try:
        return await analytics_service.get_analytics_summary()
    except Exception as exc:
        logger.exception("Unexpected error retrieving analytics")
        raise HTTPException(status_code=500, detail=f"Internal error retrieving analytics: {exc}") from exc
