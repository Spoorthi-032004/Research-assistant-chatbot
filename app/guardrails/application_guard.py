"""
Application Guardrails Layer.
Enforces business rules, reduces unnecessary API/LLM calls, validates research
workflows, and keeps the assistant strictly focused on research-related tasks.
"""
import re
from typing import Optional, Dict, Any

# Define regex patterns for prompt injection attempts
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|any\s+|the\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"show\s+hidden\s+prompts", re.IGNORECASE),
    re.compile(r"reveal\s+internal\s+configuration", re.IGNORECASE),
    re.compile(r"show\s+api\s+keys", re.IGNORECASE),
    re.compile(r"print\s+environment\s+variables", re.IGNORECASE),
    re.compile(r"generate\s+(your\s+)?own\s+source\s+code", re.IGNORECASE),
    re.compile(r"explain\s+(your\s+)?backend\s+implementation", re.IGNORECASE),
    re.compile(r"ignore\s+(the\s+)?guardrails", re.IGNORECASE),
    re.compile(r"delete\s+or\s+modify\s+(your\s+)?instructions", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+|any\s+|the\s+)?(previous\s+)?(instructions|prompts|assignments?)", re.IGNORECASE),
    re.compile(r"(tell|explain|reveal|expose|disclose)\s+(me\s+)?what\s+(coding|architecture|system\s+prompt|model|implementation|prompt\s+or\s+assignment)\b", re.IGNORECASE),
    re.compile(r"(reveal|what|tell|disclose|show)\s+(are\s+)?(the\s+)?(technologies|tech|tools|frameworks|libraries)\s+(used|built|powering)\b", re.IGNORECASE),
]

# Define scope violation patterns (non-research queries)
SCOPE_PATTERNS = [
    # Cooking / Recipes / Baking
    re.compile(r"\b(?:how\s+to\s+)?(?:cook|bake|prepare|make)\b[^.!?]*?\b(cake|pizza|pasta|cookie|bread|dinner|lunch|breakfast|salad|soup|sauce|recipe|curry|rice|chicken|dish|food)\b", re.IGNORECASE),
    re.compile(r"recipe\s+for", re.IGNORECASE),
    
    # Creative writing / Entertainment (unless in a research context)
    re.compile(r"\b(?:write|tell|generate|recite)\s+(?:a\s+)?(?:joke|poem|song|story|lyrics|limerick|riddle)\b", re.IGNORECASE),
    
    # Pop culture / Media suggestions
    re.compile(r"\b(?:recommend|suggest|best)\b[^.!?]*?\b(?:movie|film|tv\s+show|series|anime|book|novel|song|music|band|artist)\b", re.IGNORECASE),
    
    # Sports scores
    re.compile(r"\b(?:cricket|football|soccer|baseball|basketball|sports?)\s+(?:scores?|match|game|updates?|standing|schedule)\b", re.IGNORECASE),
    
    # Weather
    re.compile(r"\bweather\b[^.!?]*?\b(?:forecast|updates?|report|today|tomorrow|temperature)\b", re.IGNORECASE),
    
    # Personal advice
    re.compile(r"\b(?:life|relationship|career|dating|financial)\s+advice\b", re.IGNORECASE),
    
    # General coding unrelated to research
    re.compile(r"general\s+coding\s+unrelated\s+to\s+research", re.IGNORECASE),
]

# Define general out-of-scope keyword lists to catch variations
SCOPE_KEYWORDS = [
    "bake a cake", "tell a joke", "write a poem", "recommend a movie", "cricket score",
    "weather update", "weather forecast", "life advice", "movie recommendation", "how to cook"
]

# Define harmful content patterns
HARMFUL_PATTERNS = [
    # Creation of harmful items
    re.compile(r"\b(?:make|build|create|synthesize|develop|construct|generate|manufacture|prepare|recipes?\s+for)\b[^.!?]*?\b(?:bomb|explosive|weapon|poison|toxin|nerve\s+agent|chemical\s+weapon|malware|virus|ransomware|spyware|trojan)\b", re.IGNORECASE),
    
    # Dangerous actions
    re.compile(r"\b(?:how\s+to|steps\s+to)\s+[^.!?]*?\b(?:harm|kill|hurt|attack|assassinate|murder|poison|inject|hack|compromise|exploit)\b", re.IGNORECASE),
    
    # Explicit harmful terms
    re.compile(r"\b(?:bomb\s+making|weapon\s+construction|poison\s+creation|malware\s+generation|credential\s+theft|hacking\s+accounts?|illegal\s+activities|dangerous\s+instructions)\b", re.IGNORECASE),
]

# Define administrative protection patterns
ADMIN_PATTERNS = [
    re.compile(r"delete\s+all\s+conversations", re.IGNORECASE),
    re.compile(r"delete\s+(the\s+)?database", re.IGNORECASE),
    re.compile(r"remove\s+uploaded\s+papers", re.IGNORECASE),
    re.compile(r"drop\s+mongodb\s+collections?", re.IGNORECASE),
    re.compile(r"shutdown\s+(the\s+)?backend", re.IGNORECASE),
    re.compile(r"reset\s+(the\s+)?application", re.IGNORECASE),
    re.compile(r"delete\s+every\s+user's\s+history", re.IGNORECASE),
]

# Workflow patterns
WORKFLOW_SUMMARIZE_PATTERNS = [
    re.compile(r"\bsummarize\s+(this\s+|the\s+)?paper\b", re.IGNORECASE),
    re.compile(r"\bsummary\s+of\s+(this\s+|the\s+)?paper\b", re.IGNORECASE),
    re.compile(r"\bexplain\s+(this\s+|the\s+)?paper\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+(this\s+|the\s+)?paper\s+about\b", re.IGNORECASE),
]

WORKFLOW_SECTION_PATTERNS = [
    re.compile(r"\bexplain\s+section\s+(\d+|[a-zA-Z]+)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+in\s+section\s+(\d+|[a-zA-Z]+)\b", re.IGNORECASE),
    re.compile(r"\bgo\s+over\s+section\s+(\d+|[a-zA-Z]+)\b", re.IGNORECASE),
    re.compile(r"\bsection\s+(\d+|[a-zA-Z]+)\s+content\b", re.IGNORECASE),
]

WORKFLOW_COMPARE_PATTERNS = [
    re.compile(r"\bcompare\s+(the\s+)?(?:(first|second|third|fourth|fifth|last|numbered)\s+)?recommended\s+papers?\b", re.IGNORECASE),
    re.compile(r"\bcompare\s+recommended\s+papers?\b", re.IGNORECASE),
    re.compile(r"\bcompare\s+(the\s+)?recommendations?\b", re.IGNORECASE),
]

WORKFLOW_EXTEND_PATTERNS = [
    re.compile(r"\brecommend\s+papers\s+extending\s+(this\s+|the\s+)?work\b", re.IGNORECASE),
    re.compile(r"\bfind\s+papers\s+extending\s+(this\s+|the\s+)?work\b", re.IGNORECASE),
    re.compile(r"\brecommend\s+papers\s+related\s+to\s+(this\s+|the\s+)?paper\b", re.IGNORECASE),
    re.compile(r"\bextend\s+(this\s+|the\s+)?work\b", re.IGNORECASE),
]


def run_application_guardrails(question: str, session: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Executes business rules checks and session validations on user input.
    Returns a custom warning response if blocked, or None if the request is allowed.
    """
    clean_q = question.strip()

    # 1. Prompt Injection Protection
    if any(p.search(clean_q) for p in PROMPT_INJECTION_PATTERNS):
        return (
            "I can't disclose or override my internal instructions, implementation, "
            "or system configuration. I'm designed to assist with research paper analysis, "
            "research paper discovery, paper comparison, and literature review tasks."
        )

    # 2. Harmful Content Guardrail
    if any(p.search(clean_q) for p in HARMFUL_PATTERNS):
        return (
            "I can't assist with requests involving harmful or illegal activities. "
            "If your interest is academic or research-related, I can provide safe "
            "explanations of the underlying concepts."
        )

    # 3. Administrative Protection
    if any(p.search(clean_q) for p in ADMIN_PATTERNS):
        return "I can't perform administrative or system-level actions through the chat interface."

    # 4. Research Scope Guardrail
    # Check regexes first
    if any(p.search(clean_q) for p in SCOPE_PATTERNS):
        return (
            "This assistant is designed specifically for research paper analysis, "
            "research paper discovery, paper comparison, and literature review assistance. "
            "Please upload a research paper or ask for research papers on a specific topic."
        )
    # Check general keywords
    lowered_q = clean_q.lower()
    if any(kw in lowered_q for kw in SCOPE_KEYWORDS):
        return (
            "This assistant is designed specifically for research paper analysis, "
            "research paper discovery, paper comparison, and literature review assistance. "
            "Please upload a research paper or ask for research papers on a specific topic."
        )

    # 5. Research Workflow Validation
    # Determine session flags
    has_paper = False
    has_recommendations = False
    if session:
        if session.get("paper_id"):
            has_paper = True
        if session.get("recommended_papers"):
            has_recommendations = True

    # "Summarize this paper"
    if any(p.search(clean_q) for p in WORKFLOW_SUMMARIZE_PATTERNS):
        if not has_paper:
            return "Please upload a research paper before requesting paper-specific analysis."

    # "Explain Section 4"
    if any(p.search(clean_q) for p in WORKFLOW_SECTION_PATTERNS):
        if not has_paper:
            return "Please upload a research paper before asking questions about its contents."

    # "Compare the first recommended paper"
    if any(p.search(clean_q) for p in WORKFLOW_COMPARE_PATTERNS):
        if not has_recommendations:
            return (
                "There are no recommended papers available in the current conversation. "
                "Please search for research papers first."
            )

    # "Recommend papers extending this work"
    if any(p.search(clean_q) for p in WORKFLOW_EXTEND_PATTERNS):
        if not has_paper:
            return (
                "Please upload a research paper first or specify a research topic "
                "to search for related papers."
            )

    return None
