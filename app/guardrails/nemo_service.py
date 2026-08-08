"""
NVIDIA NeMo Guardrails integration.

NeMo Guardrails config lives in app/guardrails/nemo_config/ (config.yml +
rails.co). We initialize the rails app lazily and expose simple
check_input / check_output helpers used by the input/output guardrail
pipeline. If the nemoguardrails package or config is unavailable at
runtime (e.g. stripped-down deployment), we fail open with a warning so the
rest of the guardrail stack (regex + Presidio) still protects the app.
"""
import os
import threading
from typing import Tuple

from app.core.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

_rails = None
_init_lock = threading.Lock()
_unavailable = False

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "nemo_config")


def _get_rails():
    global _rails, _unavailable
    if _rails is not None or _unavailable:
        return _rails

    with _init_lock:
        if _rails is None and not _unavailable:
            try:
                from nemoguardrails import LLMRails, RailsConfig

                config = RailsConfig.from_path(_CONFIG_DIR)
                _rails = LLMRails(config)
                logger.info("NeMo Guardrails initialized from %s", _CONFIG_DIR)
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("NeMo Guardrails unavailable (%s); falling back to regex+Presidio only", exc)
                _unavailable = True

    return _rails


async def check_input(text: str) -> Tuple[bool, str]:
    """Returns (is_allowed, reason_if_blocked)."""
    if not settings.ENABLE_NEMO_GUARDRAILS:
        return True, ""

    rails = _get_rails()
    if rails is None:
        return True, ""  # fail open, regex+Presidio already ran before this

    try:
        response = await rails.generate_async(
            messages=[{"role": "user", "content": text}]
        )
        content = response.get("content", "") if isinstance(response, dict) else str(response)
        if "i can't help with that" in content.lower() or "blocked" in content.lower():
            return False, "Blocked by NeMo Guardrails input rail"
        return True, ""
    except Exception as exc:  # pragma: no cover
        logger.error("NeMo Guardrails input check failed: %s", exc)
        return True, ""  # fail open rather than breaking the app


async def check_output(text: str) -> Tuple[bool, str]:
    """Returns (is_allowed, reason_if_blocked) for LLM-generated output."""
    if not settings.ENABLE_NEMO_GUARDRAILS:
        return True, ""

    rails = _get_rails()
    if rails is None:
        return True, ""

    try:
        response = await rails.generate_async(
            messages=[
                {"role": "user", "content": "N/A"},
                {"role": "assistant", "content": text},
            ]
        )
        content = response.get("content", "") if isinstance(response, dict) else str(response)
        if "i can't help with that" in content.lower() or "blocked" in content.lower():
            return False, "Blocked by NeMo Guardrails output rail"
        return True, ""
    except Exception as exc:  # pragma: no cover
        logger.error("NeMo Guardrails output check failed: %s", exc)
        return True, ""
