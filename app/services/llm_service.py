"""
Groq LLM client. Uses Groq's OpenAI-compatible chat completions endpoint via
httpx (async), keeping the dependency footprint small and giving full
control over retries/timeouts/error handling.
"""
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

import httpx

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.utils.token_counter import count_tokens, estimate_cost

settings = get_settings()
logger = get_logger(__name__)


class LLMError(Exception):
    pass


@dataclass
class LLMResult:
    content: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    model: str


async def generate(
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> LLMResult:
    """
    Call the Groq chat completions API. Falls back to local token counting
    when the API does not report usage explicitly.
    """
    if not settings.GROQ_API_KEY:
        raise LLMError(
            "GROQ_API_KEY is not configured. Set it in your environment or .env file."
        )

    payload = {
        "model": settings.GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
        "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
    }
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{settings.GROQ_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("Groq API error: %s - %s", exc.response.status_code, exc.response.text)
            raise LLMError(f"Groq API returned {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.RequestError as exc:
            logger.error("Groq API request failed: %s", exc)
            raise LLMError(f"Failed to reach Groq API: {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    data = response.json()

    choice = data["choices"][0]
    content = choice["message"]["content"]

    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")

    if input_tokens is None:
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        input_tokens = count_tokens(prompt_text)
    if output_tokens is None:
        output_tokens = count_tokens(content)

    total_tokens = input_tokens + output_tokens
    cost = estimate_cost(
        input_tokens, output_tokens, settings.LLM_INPUT_COST_PER_1M, settings.LLM_OUTPUT_COST_PER_1M
    )

    return LLMResult(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=round(latency_ms, 2),
        estimated_cost_usd=cost,
        model=settings.GROQ_MODEL,
    )
