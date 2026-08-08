"""
POST /compare -- compare structured profiles of multiple uploaded papers or compare a recommended paper.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException

from app.core.logging_config import get_logger
from app.models.schemas import (
    CompareRequest, CompareResponse, ErrorResponse,
    CompareRecommendedRequest, CompareRecommendedResponse
)
from app.services import compare_service

logger = get_logger(__name__)

router = APIRouter(prefix="/compare", tags=["Compare"])


@router.post(
    "",
    response_model=CompareResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Compare two or more uploaded papers using cached structured data",
)
async def compare_papers(request: CompareRequest) -> CompareResponse:
    try:
        return await compare_service.compare_papers(request.paper_ids)
    except compare_service.CompareError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during comparison")
        raise HTTPException(status_code=500, detail=f"Internal error during comparison: {exc}") from exc


@router.post(
    "/recommendation",
    response_model=CompareRecommendedResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Compare a recommended paper with an uploaded paper (3-level comparison strategy)"
)
async def compare_recommendation(request: CompareRecommendedRequest) -> CompareRecommendedResponse:
    try:
        # recommended_paper is a SearchResultItem, convert it to dict for service
        rec_paper_dict = request.recommended_paper.model_dump()
        result = await compare_service.compare_uploaded_with_recommended(
            session_id=request.session_id,
            paper_id=request.paper_id,
            recommended_paper=rec_paper_dict
        )
        return CompareRecommendedResponse(
            session_id=request.session_id,
            paper_id=request.paper_id,
            recommended_title=result["recommended_title"],
            comparison_level=result["comparison_level"],
            comparison_report=result["comparison_report"],
            disclaimer=result["disclaimer"],
            generated_at=datetime.now(timezone.utc)
        )
    except compare_service.CompareError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during recommended comparison")
        raise HTTPException(status_code=500, detail=f"Internal error during comparison: {exc}") from exc
