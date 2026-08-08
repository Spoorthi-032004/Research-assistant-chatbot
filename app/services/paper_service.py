"""
Core paper-processing pipeline used by POST /upload.

To minimize LLM API usage, the summary, structured profile, and initial
research-gap analysis are generated in a SINGLE Groq call that returns
structured JSON, rather than three separate calls.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.database import mongo, faiss_store
from app.models.schemas import MultiUploadResponse, StructuredProfile, UploadResponse
from app.services import pdf_service, embedding_service, llm_service, analytics_service
from app.utils.hashing import sha256_bytes
from app.utils.chunker import chunk_text
from app.utils.page_mapper import compute_page_word_boundaries, word_index_to_page

settings = get_settings()
logger = get_logger(__name__)


class PaperProcessingError(Exception):
    pass


_ANALYSIS_SYSTEM_PROMPT = """You are a research paper analysis engine. You will be given \
excerpts from a research paper (abstract, introduction, methodology, results, and conclusion \
sections when available). Respond with ONLY a single valid JSON object (no markdown fences, \
no commentary) with exactly this schema:

{
  "summary": "a concise 10-20 sentence summary of the paper",
  "structured_information": {
    "title": "paper title",
    "problem_statement": "the core problem being addressed",
    "methodology": "brief description of the method/approach used",
    "datasets": ["dataset1", "dataset2"],
    "models_used": ["model1", "model2"],
    "evaluation_metrics": ["metric1", "metric2"],
    "results": "brief description of key results",
    "limitations": "brief description of stated or apparent limitations",
    "future_work": "brief description of suggested future work, if any",
    "keywords": ["keyword1", "keyword2", "keyword3"]
  },
  "research_gap": "a 5-10 sentence analysis of the research gap this paper leaves open"
}

If a field cannot be determined from the text, use an empty string or empty list as appropriate. \
Do not include any text outside the JSON object."""


def _build_analysis_context(full_text: str) -> str:
    """
    Keep the analysis prompt small (and thus cheap) without relying on
    section detection. Takes a chunk from the start of the paper (usually
    covers title/abstract/introduction/methodology) and a chunk from the
    end (usually covers results/conclusion/future work), since those are
    the most informative regions for summary/profile/gap generation.
    """
    front_chars = 4500
    back_chars = 2500
    stripped = full_text.strip()

    if len(stripped) <= front_chars + back_chars:
        return stripped

    front = stripped[:front_chars]
    back = stripped[-back_chars:]
    return f"{front}\n\n...[middle of document omitted]...\n\n{back}"


def _parse_llm_json(raw_content: str) -> dict:
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to salvage the largest {...} block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


async def _generate_analysis(paper_text_context: str) -> Tuple[str, StructuredProfile, str, llm_service.LLMResult]:
    messages = [
        {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Paper excerpts:\n\n{paper_text_context}"},
    ]
    result = await llm_service.generate(messages, max_tokens=900, temperature=0.1)

    try:
        parsed = _parse_llm_json(result.content)
        summary = parsed.get("summary", "").strip()
        structured = StructuredProfile(**parsed.get("structured_information", {}))
        research_gap = parsed.get("research_gap", "").strip()
    except Exception as exc:
        logger.error("Failed to parse LLM analysis JSON: %s | raw=%s", exc, result.content[:500])
        # Degrade gracefully rather than failing the whole upload
        summary = result.content[:800]
        structured = StructuredProfile()
        research_gap = "Unable to automatically determine research gap; raw model output was not valid JSON."

    return summary, structured, research_gap, result


async def process_upload(file_path: str, original_filename: str) -> UploadResponse:
    # 1. Validate PDF
    pdf_service.validate_pdf(file_path)

    # 2. Hash for duplicate detection
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    file_hash = sha256_bytes(file_bytes)

    # 3. Duplicate check -> return cached analysis only if FAISS vectors also exist
    existing = await mongo.get_paper_by_hash(file_hash)
    if existing is not None and faiss_store.paper_has_vectors(file_hash):
        logger.info("Duplicate upload detected for hash=%s (DB and FAISS are in sync), returning cached analysis", file_hash)
        return UploadResponse(
            paper_id=file_hash,
            paper_name=existing["paper_name"],
            sha256=file_hash,
            is_duplicate=True,
            summary=existing["summary"],
            structured_information=StructuredProfile(**existing["structured_information"]),
            research_gap=existing["research_gap"],
            timestamp=existing["timestamp"],
        )

    # 4. Extract text (per-page, so chunks can be approximately mapped back
    #    to a page number for FAISS metadata)
    pages = pdf_service.extract_pages(file_path)
    full_text = "\n".join(pages)
    if not full_text.strip():
        raise PaperProcessingError("No extractable text found in PDF (it may be a scanned image; OCR not run in this pipeline stage).")

    # 5. Chunk full document text (fixed size, no section detection)
    chunks = chunk_text(full_text, settings.CHUNK_SIZE_TOKENS, settings.CHUNK_OVERLAP_TOKENS)
    if not chunks:
        raise PaperProcessingError("Unable to derive any text chunks from the uploaded PDF.")

    # 6. Generate embeddings for all chunks (batched internally by
    #    embedding_service, not one chunk at a time)
    chunk_texts = [c.text for c in chunks]
    vectors = embedding_service.embed_texts(chunk_texts)

    # 7. Store vectors in the single global FAISS index, with a metadata
    #    record per chunk (paper_id, paper_name, page_number, chunk_index,
    #    chunk_text) persisted to metadata.json. Chunks/embeddings are NEVER
    #    written to MongoDB.
    page_word_boundaries = compute_page_word_boundaries(pages)
    chunk_records = [
        {
            "page_number": word_index_to_page(c.start_word_index, page_word_boundaries),
            "chunk_index": c.chunk_index,
            "chunk_text": c.text,
        }
        for c in chunks
    ]
    faiss_store.add_vectors(file_hash, original_filename, vectors, chunk_records)

    if existing is not None:
        # DB entry exists but FAISS vectors were missing. Reuse cached analysis!
        logger.info("MongoDB entry exists for %s but FAISS vectors were missing. Re-indexed paper and reusing cached analysis.", file_hash)
        summary = existing["summary"]
        structured = StructuredProfile(**existing["structured_information"])
        research_gap = existing["research_gap"]
        timestamp = existing["timestamp"]
    else:
        # 8. Single LLM call -> summary + structured profile + research gap
        analysis_context = _build_analysis_context(full_text)
        summary, structured, research_gap, llm_result = await _generate_analysis(analysis_context)
        await analytics_service.record_llm_usage("/upload", llm_result)

        # 9. Persist only generated AI results in MongoDB (never embeddings/chunks)
        timestamp = datetime.now(timezone.utc)
        document = {
            "paper_name": original_filename,
            "sha256": file_hash,
            "summary": summary,
            "structured_information": structured.model_dump(),
            "research_gap": research_gap,
            "timestamp": timestamp,
        }
        await mongo.insert_paper(document)

    return UploadResponse(
        paper_id=file_hash,
        paper_name=original_filename,
        sha256=file_hash,
        is_duplicate=(existing is not None),
        summary=summary,
        structured_information=structured,
        research_gap=research_gap,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Multi-paper upload (up to settings.MAX_UPLOAD_FILES PDFs per request)
# ---------------------------------------------------------------------------
_COMBINED_SUMMARY_SYSTEM_PROMPT = """You are a research analyst. You will be given the \
individual summaries of multiple related research papers. Write a single combined summary \
in Markdown with three sections:
## Common Themes
## Key Differences
## Overall Findings
Be concise -- use bullet points, not long paragraphs. Do not repeat each paper's summary \
verbatim; synthesize across all of them."""


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) or 1e-9
    return float(np.dot(vec_a, vec_b) / denom)


def _average_pairwise_similarity(vectors: np.ndarray) -> float:
    n = vectors.shape[0]
    if n < 2:
        return 1.0
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(_cosine_similarity(vectors[i], vectors[j]))
    return sum(similarities) / len(similarities)


def _determine_relatedness(papers: List[UploadResponse]) -> float:
    """
    Embeds each paper's generated summary (used as a stand-in for its
    abstract) and returns the average pairwise cosine similarity across all
    papers. Papers are treated as related if this exceeds
    settings.PAPER_SIMILARITY_THRESHOLD (default 0.80).
    """
    summaries = [p.summary for p in papers]
    vectors = embedding_service.embed_texts(summaries)
    return _average_pairwise_similarity(vectors)


async def _generate_combined_summary(papers: List[UploadResponse]) -> str:
    paper_blocks = []
    for i, paper in enumerate(papers, start=1):
        paper_blocks.append(f"Paper {i}: {paper.paper_name}\nSummary: {paper.summary}")
    prompt_text = "\n\n".join(paper_blocks)

    messages = [
        {"role": "system", "content": _COMBINED_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    llm_result = await llm_service.generate(messages, max_tokens=700, temperature=0.2)
    await analytics_service.record_llm_usage("/upload", llm_result)
    return llm_result.content.strip()


async def process_multiple_uploads(files: List[Tuple[str, str]]) -> MultiUploadResponse:
    """
    files: list of (temp_file_path, original_filename) tuples, already saved
    to disk by the API layer. Max settings.MAX_UPLOAD_FILES entries.

    Each PDF is processed fully and independently (hash check, parsing,
    chunking, embeddings, FAISS storage, summary/profile/gap generation --
    see process_upload()). Once all papers are processed, relatedness is
    determined from their summary embeddings; a combined summary is only
    generated when the papers are related.
    """
    if not files:
        raise PaperProcessingError("At least one PDF file is required.")
    if len(files) > settings.MAX_UPLOAD_FILES:
        raise PaperProcessingError(
            f"A maximum of {settings.MAX_UPLOAD_FILES} PDF files can be uploaded at once."
        )

    papers: List[UploadResponse] = []
    for file_path, original_filename in files:
        result = await process_upload(file_path, original_filename)
        papers.append(result)

    if len(papers) < 2:
        return MultiUploadResponse(
            relatedness="single",
            average_similarity=None,
            combined_summary=None,
            papers=papers,
            message=None,
        )

    average_similarity = _determine_relatedness(papers)
    related = average_similarity > settings.PAPER_SIMILARITY_THRESHOLD

    if related:
        combined_summary = await _generate_combined_summary(papers)
        return MultiUploadResponse(
            relatedness="related",
            average_similarity=round(average_similarity, 4),
            combined_summary=combined_summary,
            papers=papers,
            message=None,
        )

    return MultiUploadResponse(
        relatedness="unrelated",
        average_similarity=round(average_similarity, 4),
        combined_summary=None,
        papers=papers,
        message="The uploaded papers belong to different topics.",
    )
