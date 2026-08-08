"""
Paper comparison service.
Supports both internal uploaded papers comparison and recommended papers comparison
implementing the three-level comparison strategy.
"""
import os
import tempfile
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import httpx

from app.database import mongo
from app.models.schemas import CompareResponse
from app.services import llm_service, analytics_service, pdf_service
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class CompareError(Exception):
    pass


_COMPARE_SYSTEM_PROMPT = """You are a research analyst. You will be given structured profiles \
for multiple research papers (problem statement, methodology, dataset, evaluation metrics, \
results, limitations, future work). Produce a concise, well-organized comparison report in \
Markdown covering: Problem Statement, Methodology, Dataset, Evaluation Metrics, Results, \
Limitations, and Future Work for each paper, followed by a short "Key Differences" section. \
Be concise -- use bullet points, not long paragraphs."""

_FULL_TEXT_COMPARE_SYSTEM_PROMPT = """You are a research analyst. You will be given the structured profile of an uploaded research paper, and key text excerpts from a recommended research paper. 
Produce a concise, well-organized comparison report in Markdown covering: 
1. Problem Statement Comparison (how the problems compare)
2. Methodology Comparison (how the methods compare)
3. Results & Evaluation Comparison (compare datasets, evaluation metrics, results)
4. Limitations & Future Work Comparison
5. Key Differences (bulleted list of major differences)
6. Synergies (how these papers could extend or complement each other)

Be concise -- use bullet points where possible."""

_ABSTRACT_COMPARE_SYSTEM_PROMPT = """You are a research analyst. You will be given the structured profile of an uploaded research paper, and the abstract of a recommended research paper.
Produce a concise, well-organized comparison report in Markdown covering:
1. Problem Statement Comparison
2. Methodology Comparison (based on abstract)
3. Results & Evaluation Comparison (based on abstract)
4. Key Differences
5. Potential Relatedness

Be concise -- use bullet points where possible. Note that since you only have the abstract for the recommended paper, some details might be limited."""


def _format_profile_for_prompt(paper: dict) -> str:
    info = paper.get("structured_information", {})
    lines = [f"Paper: {paper.get('paper_name', 'Unknown')} (id={paper.get('sha256')})"]
    lines.append(f"- Problem Statement: {info.get('problem_statement') or 'N/A'}")
    lines.append(f"- Methodology: {info.get('methodology') or 'N/A'}")
    lines.append(f"- Datasets: {', '.join(info.get('datasets') or []) or 'N/A'}")
    lines.append(f"- Evaluation Metrics: {', '.join(info.get('evaluation_metrics') or []) or 'N/A'}")
    lines.append(f"- Results: {info.get('results') or 'N/A'}")
    lines.append(f"- Limitations: {info.get('limitations') or 'N/A'}")
    lines.append(f"- Future Work: {info.get('future_work') or 'N/A'}")
    return "\n".join(lines)


def _build_analysis_excerpt(full_text: str) -> str:
    front_chars = 5000
    back_chars = 3000
    stripped = full_text.strip()
    if len(stripped) <= front_chars + back_chars:
        return stripped
    front = stripped[:front_chars]
    back = stripped[-back_chars:]
    return f"{front}\n\n...[middle of document omitted]...\n\n{back}"


async def _download_and_extract_pdf_text(pdf_url: str) -> Optional[str]:
    try:
        logger.info("Attempting to download recommended paper PDF: %s", pdf_url)
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(pdf_url, follow_redirects=True)
            response.raise_for_status()
            
            # Simple magic bytes check for PDF
            if not response.content.startswith(b"%PDF"):
                logger.warning("Downloaded content from %s does not start with %%PDF magic bytes", pdf_url)
                return None
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            try:
                text = pdf_service.extract_text(tmp_path)
                return text
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    except Exception as exc:
        logger.warning("Failed to download or parse PDF from %s: %s", pdf_url, exc)
        return None


async def compare_papers(paper_ids: List[str]) -> CompareResponse:
    papers = await mongo.get_papers_by_ids(paper_ids)
    found_ids = {p["sha256"] for p in papers}
    missing = [pid for pid in paper_ids if pid not in found_ids]
    if missing:
        raise CompareError(f"The following paper_ids were not found: {', '.join(missing)}. Upload them first.")

    profiles_text = "\n\n".join(_format_profile_for_prompt(p) for p in papers)
    messages = [
        {"role": "system", "content": _COMPARE_SYSTEM_PROMPT},
        {"role": "user", "content": profiles_text},
    ]

    llm_result = await llm_service.generate(messages, max_tokens=900, temperature=0.2)
    await analytics_service.record_llm_usage("/compare", llm_result)

    return CompareResponse(
        paper_ids=paper_ids,
        comparison_report=llm_result.content,
        generated_at=datetime.now(timezone.utc),
    )


async def compare_uploaded_with_recommended(
    session_id: str,
    paper_id: str,
    recommended_paper: dict
) -> dict:
    """
    Implements the three-level comparison strategy between an uploaded paper
    and a recommended/discovered paper.
    """
    uploaded_paper = await mongo.get_paper_by_id(paper_id)
    if uploaded_paper is None:
        raise CompareError(f"Uploaded paper with id={paper_id} not found in database. Upload it first.")

    title = recommended_paper.get("title", "Recommended Paper")
    abstract = recommended_paper.get("abstract")
    pdf_url = recommended_paper.get("pdf_url")

    text_content = None
    level = "none"
    disclaimer = None

    # Level 1: Full-text analysis if accessible PDF
    if pdf_url:
        text_content = await _download_and_extract_pdf_text(pdf_url)
        if text_content and text_content.strip():
            level = "full_text"
        else:
            logger.info("PDF download/parse failed. Falling back to Level 2 (abstract) comparison for: %s", title)

    # Level 2: Abstract-only comparison
    if level == "none" and abstract and abstract.strip():
        level = "abstract"
        disclaimer = "*Note: This comparison was generated using the abstract only, as the full-text PDF was not accessible.*"

    # Level 3: No comparison possible
    if level == "none":
        report = "Additional content (abstract or PDF) is required to perform comparison."
        result = {
            "recommended_title": title,
            "comparison_level": "none",
            "comparison_report": report,
            "disclaimer": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await mongo.save_comparison_result(session_id, result)
        return result

    # Format inputs for LLM comparison
    uploaded_profile_text = _format_profile_for_prompt(uploaded_paper)

    if level == "full_text":
        rec_excerpt = _build_analysis_excerpt(text_content)
        prompt_text = (
            f"Uploaded Paper Structured Profile:\n{uploaded_profile_text}\n\n"
            f"Recommended Paper Excerpts:\n{rec_excerpt}"
        )
        system_prompt = _FULL_TEXT_COMPARE_SYSTEM_PROMPT
    else:  # level == "abstract"
        prompt_text = (
            f"Uploaded Paper Structured Profile:\n{uploaded_profile_text}\n\n"
            f"Recommended Paper Title: {title}\n"
            f"Recommended Paper Abstract: {abstract}"
        )
        system_prompt = _ABSTRACT_COMPARE_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
    ]

    logger.info("Calling Groq LLM for recommended paper comparison (level=%s)", level)
    llm_result = await llm_service.generate(messages, max_tokens=1000, temperature=0.2)
    await analytics_service.record_llm_usage("/compare/recommendation", llm_result)

    report_content = llm_result.content
    from app.guardrails.output_guard import run_output_guardrails
    output_guarded = await run_output_guardrails(report_content)
    report_content = output_guarded.sanitized_text

    if disclaimer:
        report_content = f"{disclaimer}\n\n{report_content}"

    result = {
        "recommended_title": title,
        "comparison_level": level,
        "comparison_report": report_content,
        "disclaimer": disclaimer,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Save to MongoDB session
    await mongo.save_comparison_result(session_id, result)
    return result
