"""
Token counting utilities. Uses tiktoken when available (close approximation
for most LLM tokenizers) and falls back to a simple whitespace heuristic.
"""
from app.core.logging_config import get_logger

logger = get_logger(__name__)

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback path
    _ENCODER = None
    logger.warning("tiktoken unavailable, falling back to word-count heuristic for token estimation")


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in a piece of text."""
    if not text:
        return 0
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    # Rough heuristic: ~0.75 words per token on average for English text
    return max(1, int(len(text.split()) / 0.75))


def estimate_cost(input_tokens: int, output_tokens: int, input_cost_per_1m: float, output_cost_per_1m: float) -> float:
    """Estimate USD cost given per-1M-token pricing."""
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    return round(input_cost + output_cost, 8)
