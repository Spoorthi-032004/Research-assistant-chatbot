"""
POST /search -- search external research databases for a topic.
"""
from fastapi import APIRouter, HTTPException

from app.core.logging_config import get_logger
from app.models.schemas import SearchRequest, SearchResponse, ErrorResponse
from app.services import search_service

logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


@router.post(
    "",
    response_model=SearchResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Search Semantic Scholar, OpenAlex, Crossref, and arXiv for a topic",
)
async def search_papers(request: SearchRequest) -> SearchResponse:
    try:
        results = await search_service.search_all_sources(request.topic, request.limit_per_source)
        return SearchResponse(topic=request.topic, total_results=len(results), results=results)
    except Exception as exc:
        logger.exception("Unexpected error during search")
        raise HTTPException(status_code=500, detail=f"Internal error during search: {exc}") from exc
