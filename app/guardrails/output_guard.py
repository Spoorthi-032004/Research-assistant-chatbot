"""
Output guardrail pipeline:

    Groq -> NeMo Guardrails -> Microsoft Presidio -> Regex validation

Unlike input, output PII is anonymized (masked) rather than hard-blocked
where possible, so the user still gets a useful answer with secrets redacted.
Regex-detected secrets (API keys, passwords, credit cards) are always
scrubbed since the LLM should never legitimately need to emit them for this
use case.
"""
import re
from dataclasses import dataclass
from typing import List, Tuple

from app.guardrails import regex_rules, presidio_service, nemo_service
from app.guardrails.input_guard import GuardrailBlockedException
from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class OutputGuardrailResult:
    sanitized_text: str
    redactions_made: List[str]


def _scrub_regex_secrets(text: str) -> Tuple[str, List[str]]:
    redactions = []
    scrubbed = text
    for pattern in regex_rules.API_KEY_PATTERNS:
        if pattern.search(scrubbed):
            redactions.append("API key")
            scrubbed = pattern.sub("[REDACTED_API_KEY]", scrubbed)
    for pattern in regex_rules.PASSWORD_PATTERNS:
        if pattern.search(scrubbed):
            redactions.append("Password")
            scrubbed = pattern.sub("[REDACTED_PASSWORD]", scrubbed)
    if regex_rules.check_credit_card(scrubbed):
        redactions.append("Credit card number")
        scrubbed = regex_rules.CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", scrubbed)
    return scrubbed, redactions


async def run_output_guardrails(text: str) -> OutputGuardrailResult:
    # Stage 1: NeMo Guardrails (topical / policy compliance on the response)
    allowed, reason = await nemo_service.check_output(text)
    if not allowed:
        logger.warning("Output blocked at NeMo Guardrails stage: %s", reason)
        raise GuardrailBlockedException("nemo_guardrails", [reason])

    # Stage 2: Microsoft Presidio (mask PII rather than hard-block on output)
    anonymized_text, pii_entities = presidio_service.anonymize_text(text)
    redactions = [e["entity_type"] for e in pii_entities]

    # Stage 3: Regex validation (scrub secrets/keys/cards that slipped through)
    final_text, regex_redactions = _scrub_regex_secrets(anonymized_text)
    redactions.extend(regex_redactions)

    return OutputGuardrailResult(sanitized_text=final_text, redactions_made=redactions)
