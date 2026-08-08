"""
Input guardrail pipeline:

    Regex validation -> Microsoft Presidio -> NeMo Guardrails -> Groq

Raises GuardrailBlockedException when a stage blocks the request.
"""
from dataclasses import dataclass
from typing import List

from app.guardrails import regex_rules, presidio_service, nemo_service
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class GuardrailBlockedException(Exception):
    def __init__(self, stage: str, reasons: List[str]):
        self.stage = stage
        self.reasons = reasons
        super().__init__(f"Blocked at {stage}: {', '.join(reasons)}")


@dataclass
class GuardrailResult:
    sanitized_text: str
    pii_entities_found: List[str]


async def run_input_guardrails(text: str) -> GuardrailResult:
    # Stage 1: Regex validation
    is_blocked, reasons = regex_rules.run_regex_checks(text)
    if is_blocked:
        logger.warning("Input blocked at regex stage: %s", reasons)
        raise GuardrailBlockedException("regex_validation", reasons)

    # Stage 2: Microsoft Presidio (PII detection + anonymization)
    is_pii_blocked, entity_types = presidio_service.contains_blocked_pii(text)
    sanitized_text = text
    if is_pii_blocked:
        # We block rather than silently anonymize, per spec (block emails,
        # phone numbers, credit cards, etc. in user input).
        logger.warning("Input blocked at Presidio stage: %s", entity_types)
        raise GuardrailBlockedException("presidio_pii_detection", entity_types)

    # Stage 3: NeMo Guardrails (topical / jailbreak rails)
    allowed, reason = await nemo_service.check_input(sanitized_text)
    if not allowed:
        logger.warning("Input blocked at NeMo Guardrails stage: %s", reason)
        raise GuardrailBlockedException("nemo_guardrails", [reason])

    return GuardrailResult(sanitized_text=sanitized_text, pii_entities_found=entity_types)
