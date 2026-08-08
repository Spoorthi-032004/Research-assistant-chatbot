"""
First-stage regex validation: fast, cheap pattern checks that run before the
heavier Presidio/NeMo stages. Catches obvious secrets and common prompt
injection phrasing.
"""
import re
from typing import List, Tuple

API_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                # OpenAI-style
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),                # Groq-style
    re.compile(r"AKIA[0-9A-Z]{16}"),                    # AWS access key
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),              # Google API key
    re.compile(r"ghp_[A-Za-z0-9]{36}"),                 # GitHub PAT
]

PASSWORD_PATTERNS = [
    re.compile(r"(?i)password\s*[:=]\s*\S+"),
    re.compile(r"(?i)passwd\s*[:=]\s*\S+"),
]

CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore (all|any|the)? ?previous instructions"),
    re.compile(r"(?i)disregard (all|any|the)? ?(system|prior) prompt"),
    re.compile(r"(?i)you are now (in )?(developer|dan|jailbreak) mode"),
    re.compile(r"(?i)reveal (your|the) (system prompt|instructions)"),
    re.compile(r"(?i)act as if you have no (restrictions|guardrails|filters)"),
    re.compile(r"(?i)pretend (you are|to be) an? (unfiltered|uncensored)"),
    re.compile(r"(?i)forget (all|any|the)? ?(previous\s+)?(instructions|prompts|assignments?)"),
    re.compile(r"(?i)(tell|explain|reveal|expose|disclose)\s+(me\s+)?what\s+(coding|architecture|system\s+prompt|model|implementation|prompt\s+or\s+assignment)\b"),
    re.compile(r"(?i)(reveal|what|tell|disclose|show)\s+(are\s+)?(the\s+)?(technologies|tech|tools|frameworks|libraries)\s+(used|built|powering)\b"),
]

HARMFUL_KEYWORDS = [
    "build a bomb",
    "synthesize a nerve agent",
    "how to make a weapon",
    "child sexual abuse material",
]

HARMFUL_REGEX_PATTERNS = [
    re.compile(r"\b(?:make|build|create|synthesize|develop|construct|generate|manufacture|prepare|recipes?\s+for)\b[^.!?]*?\b(?:bomb|explosive|weapon|poison|toxin|nerve\s+agent|chemical\s+weapon|malware|virus|ransomware|spyware|trojan)\b", re.IGNORECASE),
    re.compile(r"\b(?:how\s+to|steps\s+to)\s+[^.!?]*?\b(?:harm|kill|hurt|attack|assassinate|murder|poison|inject|hack|compromise|exploit)\b", re.IGNORECASE),
]


def _matches_any(patterns: List[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def check_api_keys(text: str) -> bool:
    return _matches_any(API_KEY_PATTERNS, text)


def check_passwords(text: str) -> bool:
    return _matches_any(PASSWORD_PATTERNS, text)


def check_credit_card(text: str) -> bool:
    for match in CREDIT_CARD_PATTERN.finditer(text):
        digits = re.sub(r"[ -]", "", match.group())
        if _luhn_valid(digits):
            return True
    return False


def _luhn_valid(number: str) -> bool:
    if not number.isdigit() or not (13 <= len(number) <= 19):
        return False
    total = 0
    reverse_digits = number[::-1]
    for i, ch in enumerate(reverse_digits):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def check_prompt_injection(text: str) -> bool:
    return _matches_any(PROMPT_INJECTION_PATTERNS, text)


def check_harmful_keywords(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in HARMFUL_KEYWORDS)


def check_harmful_patterns(text: str) -> bool:
    return any(p.search(text) for p in HARMFUL_REGEX_PATTERNS)


def run_regex_checks(text: str) -> Tuple[bool, List[str]]:
    """Returns (is_blocked, reasons)."""
    reasons = []
    if check_api_keys(text):
        reasons.append("API key detected")
    if check_passwords(text):
        reasons.append("Password detected")
    if check_credit_card(text):
        reasons.append("Credit card number detected")
    if check_prompt_injection(text):
        reasons.append("Prompt injection pattern detected")
    if check_harmful_keywords(text) or check_harmful_patterns(text):
        reasons.append("Harmful content detected")
    return (len(reasons) > 0, reasons)
