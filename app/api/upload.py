"""
POST /upload -- upload and analyze up to MAX_UPLOAD_FILES (default 3)
research paper PDFs at once.

Each PDF is processed fully and independently (hash check, parsing,
chunking, embeddings, FAISS storage, summary/profile/gap generation). If 2
or more PDFs are uploaded, their relatedness is determined from the
embeddings of their generated summaries (average pairwise cosine
similarity). If related, a combined summary is generated in addition to
each paper's individual summary; if not, only individual summaries are
returned along with a note that the papers belong to different topics.
"""
import os
import uuid
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.models.schemas import MultiUploadResponse, ErrorResponse
from app.services import paper_service
from app.services.pdf_service import InvalidPDFError

settings = get_settings()
logger = get_logger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post(
    "",
    response_model=MultiUploadResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary=f"Upload 1-{settings.MAX_UPLOAD_FILES} research paper PDFs for analysis",
)
async def upload_papers(
    files: List[UploadFile] = File(
        ..., description=f"Between 1 and {settings.MAX_UPLOAD_FILES} research paper PDF files"
    ),
) -> MultiUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")
    if len(files) > settings.MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {settings.MAX_UPLOAD_FILES} PDF files can be uploaded at once.",
        )

    for file in files:
        is_pdf_content_type = file.content_type in ("application/pdf", "application/octet-stream")
        is_pdf_extension = (file.filename or "").lower().endswith(".pdf")
        if not is_pdf_content_type and not is_pdf_extension:
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported ({file.filename}).")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    saved_paths: List[str] = []
    saved_files: List[tuple] = []

    try:
        for file in files:
            contents = await file.read()
            size_mb = len(contents) / (1024 * 1024)
            if size_mb > settings.MAX_UPLOAD_SIZE_MB:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} exceeds max upload size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
                )
            temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
            temp_path = os.path.join(settings.UPLOAD_DIR, temp_filename)
            with open(temp_path, "wb") as f:
                f.write(contents)
            saved_paths.append(temp_path)
            saved_files.append((temp_path, file.filename))

        result = await paper_service.process_multiple_uploads(saved_files)
        return result
    except HTTPException:
        raise
    except InvalidPDFError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except paper_service.PaperProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during upload processing")
        raise HTTPException(status_code=500, detail=f"Internal error while processing upload: {exc}") from exc
    finally:
        for temp_path in saved_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
