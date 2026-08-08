"""
Microsoft Presidio-based PII detection and anonymization.

Lazily initializes the AnalyzerEngine (spaCy model load is expensive) so
that app startup stays fast when guardrails aren't exercised immediately.
"""
from typing import List, Tuple
import threading

from app.core.config import get_settings
from app.core.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)

_analyzer = None
_anonymizer = None
_init_lock = threading.Lock()


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is not None and _anonymizer is not None:
        return _analyzer, _anonymizer

    with _init_lock:
        if _analyzer is None or _anonymizer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            _analyzer = AnalyzerEngine()
            _anonymizer = AnonymizerEngine()
            logger.info("Presidio analyzer/anonymizer engines initialized")

    return _analyzer, _anonymizer


def detect_pii(text: str) -> List[dict]:
    """Return a list of detected entities: {entity_type, start, end, score}."""
    if not settings.ENABLE_PRESIDIO or not text.strip():
        return []

    analyzer, _ = _get_engines()
    results = analyzer.analyze(text=text, language="en", entities=settings.BLOCKED_ENTITIES)
    return [
        {
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": r.score,
        }
        for r in results
    ]


def anonymize_text(text: str) -> Tuple[str, List[dict]]:
    """Detect and mask PII, returning (anonymized_text, entities_found)."""
    entities = detect_pii(text)
    if not entities:
        return text, []

    analyzer, anonymizer = _get_engines()
    analyzer_results = analyzer.analyze(text=text, language="en", entities=settings.BLOCKED_ENTITIES)
    anonymized = anonymizer.anonymize(text=text, analyzer_results=analyzer_results)
    return anonymized.text, entities


def contains_blocked_pii(text: str) -> Tuple[bool, List[str]]:
    """Returns (is_blocked, entity_types_found) for hard-block entity types."""
    entities = detect_pii(text)
    entity_types = sorted({e["entity_type"] for e in entities})
    return (len(entity_types) > 0, entity_types)
