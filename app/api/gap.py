"""
POST /gap -- research gap analysis for an uploaded paper.
"""
from fastapi import APIRouter, HTTPException

from app.core.logging_config import get_logger
from app.models.schemas import GapRequest, GapResponse, ErrorResponse
from app.services import gap_service

logger = get_logger(__name__)

router = APIRouter(prefix="/gap", tags=["Gap Analysis"])


@router.post(
    "",
    response_model=GapResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Generate research gap analysis for an uploaded paper",
)
async def gap_analysis(request: GapRequest) -> GapResponse:
    try:
        return await gap_service.analyze_gap(
            request.paper_id, request.include_related_papers, request.related_paper_limit
        )
    except gap_service.GapAnalysisError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during gap analysis")
        raise HTTPException(status_code=500, detail=f"Internal error during gap analysis: {exc}") from exc
