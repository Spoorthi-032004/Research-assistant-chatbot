"""
Research gap analysis service for POST /gap.

Builds on the cached research_gap + structured_information from the upload
pipeline, optionally enriched with a small number of related papers pulled
from the external search APIs (titles/abstracts only, to keep the prompt
small), and makes a single LLM call to produce structured gap analysis.
"""
import json
from typing import List

from app.database import mongo
from app.models.schemas import GapResponse
from app.services import llm_service, analytics_service, search_service
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class GapAnalysisError(Exception):
    pass


_GAP_SYSTEM_PROMPT = """You are a research strategy analyst. Given a paper's structured profile, \
its preliminary research gap note, and (optionally) a few related recent papers' titles and \
abstracts, respond with ONLY a valid JSON object (no markdown fences) with exactly this schema:

{
  "research_gap": "a refined 3-5 sentence research gap statement",
  "missing_work": ["item1", "item2"],
  "possible_improvements": ["item1", "item2"],
  "novel_directions": ["item1", "item2"],
  "suggested_future_work": ["item1", "item2"]
}

Keep each list item short (under 20 words). Do not include text outside the JSON object."""


def _parse_gap_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


async def analyze_gap(paper_id: str, include_related: bool, related_limit: int) -> GapResponse:
    paper = await mongo.get_paper_by_id(paper_id)
    if paper is None:
        raise GapAnalysisError(f"No paper found with paper_id={paper_id}. Upload it first via /upload.")

    info = paper.get("structured_information", {})
    context_parts = [
        f"Title: {info.get('title') or paper.get('paper_name')}",
        f"Problem Statement: {info.get('problem_statement') or 'N/A'}",
        f"Methodology: {info.get('methodology') or 'N/A'}",
        f"Limitations: {info.get('limitations') or 'N/A'}",
        f"Existing Future Work: {info.get('future_work') or 'N/A'}",
        f"Preliminary Research Gap: {paper.get('research_gap') or 'N/A'}",
    ]

    if include_related:
        query_topic = info.get("title") or paper.get("paper_name") or ""
        related_papers = await search_service.search_all_sources(query_topic, limit_per_source=max(1, related_limit // 2))
        related_papers = related_papers[:related_limit]
        if related_papers:
            related_text = "\n".join(
                f"- {p.title} ({p.year or 'n.d.'}): {(p.abstract or '')[:250]}" for p in related_papers
            )
            context_parts.append(f"Related Recent Papers:\n{related_text}")

    context_text = "\n\n".join(context_parts)
    messages = [
        {"role": "system", "content": _GAP_SYSTEM_PROMPT},
        {"role": "user", "content": context_text},
    ]

    llm_result = await llm_service.generate(messages, max_tokens=700, temperature=0.3)
    await analytics_service.record_llm_usage("/gap", llm_result)

    try:
        parsed = _parse_gap_json(llm_result.content)
    except Exception as exc:
        logger.error("Failed to parse gap analysis JSON: %s | raw=%s", exc, llm_result.content[:500])
        parsed = {
            "research_gap": llm_result.content[:600],
            "missing_work": [],
            "possible_improvements": [],
            "novel_directions": [],
            "suggested_future_work": [],
        }

    return GapResponse(
        paper_id=paper_id,
        research_gap=parsed.get("research_gap", ""),
        missing_work=parsed.get("missing_work", []),
        possible_improvements=parsed.get("possible_improvements", []),
        novel_directions=parsed.get("novel_directions", []),
        suggested_future_work=parsed.get("suggested_future_work", []),
    )
