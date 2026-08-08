"""
AI Research Assistant -- FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

All functionality is exposed via Swagger UI at /docs (no frontend required).
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging_config import configure_logging, get_logger
from app.database.mongo import connect_to_mongo, close_mongo_connection
from app.services import embedding_service
from app.api import upload, chat, search, compare, gap, analytics

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s)", settings.APP_NAME, settings.APP_ENV)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
    await connect_to_mongo()
    # Load the embedding model once, up front, so it is already resident in
    # memory (and reused) for every /upload and /chat request instead of
    # being loaded lazily on the first call.
    embedding_service.preload_model()
    yield
    await close_mongo_connection()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "A CPU-friendly, LLM-usage-minimizing research assistant backend. "
        "Upload papers, chat with them via RAG, search external academic "
        "databases, compare papers, analyze research gaps, and track LLM "
        "token/cost analytics -- all through this Swagger UI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(search.router)
app.include_router(compare.router)
app.include_router(gap.router)
app.include_router(analytics.router)


@app.get("/health", tags=["Health"], summary="Health check")
async def health_check():
    return {"status": "ok", "app": settings.APP_NAME, "env": settings.APP_ENV}
