"""
Token/latency/cost analytics tracking, persisted to MongoDB.
"""
from datetime import datetime, timezone
from typing import List

from app.database import mongo
from app.models.schemas import AnalyticsEvent, AnalyticsSummary
from app.services.llm_service import LLMResult
from app.core.logging_config import get_logger

logger = get_logger(__name__)


async def record_llm_usage(endpoint: str, result: LLMResult) -> None:
    event = {
        "endpoint": endpoint,
        "model_used": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "latency_ms": result.latency_ms,
        "estimated_cost_usd": result.estimated_cost_usd,
        "timestamp": datetime.now(timezone.utc),
    }
    await mongo.insert_analytics_event(event)


async def get_analytics_summary() -> AnalyticsSummary:
    raw_events = await mongo.get_all_analytics_events()
    events: List[AnalyticsEvent] = [AnalyticsEvent(**e) for e in raw_events]

    total_requests = len(events)
    total_input = sum(e.input_tokens for e in events)
    total_output = sum(e.output_tokens for e in events)
    total_tokens = sum(e.total_tokens for e in events)
    total_cost = round(sum(e.estimated_cost_usd for e in events), 8)
    avg_latency = round(sum(e.latency_ms for e in events) / total_requests, 2) if total_requests else 0.0

    return AnalyticsSummary(
        total_requests=total_requests,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_tokens,
        total_estimated_cost_usd=total_cost,
        average_latency_ms=avg_latency,
        events=events,
    )
