"""
Pydantic request/response models shared across the API.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured paper profile
# ---------------------------------------------------------------------------
class StructuredProfile(BaseModel):
    title: Optional[str] = None
    problem_statement: Optional[str] = None
    methodology: Optional[str] = None
    datasets: List[str] = Field(default_factory=list)
    models_used: List[str] = Field(default_factory=list)
    evaluation_metrics: List[str] = Field(default_factory=list)
    results: Optional[str] = None
    limitations: Optional[str] = None
    future_work: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    paper_id: str
    paper_name: str
    sha256: str
    is_duplicate: bool
    summary: str
    structured_information: StructuredProfile
    research_gap: str
    timestamp: datetime


class MultiUploadResponse(BaseModel):
    """
    Response for POST /upload, which accepts 1-3 PDFs at once.

    relatedness:
      - "single"     -> only one PDF was uploaded, no relatedness check run
      - "related"     -> >=2 PDFs uploaded and avg. pairwise cosine
                          similarity of their summaries exceeded the
                          configured threshold; combined_summary is set
      - "unrelated"   -> >=2 PDFs uploaded but they were not similar enough;
                          combined_summary is null and message explains why
    """
    relatedness: str
    average_similarity: Optional[float] = None
    combined_summary: Optional[str] = None
    papers: List[UploadResponse]
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Existing chat session to continue; omit to start a new one")
    paper_id: Optional[str] = Field(default=None, description="sha256 hash / paper id to chat with")
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class CitationChunk(BaseModel):
    chunk_index: int
    page_number: Optional[int] = None
    snippet: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[CitationChunk]
    input_tokens: int
    output_tokens: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=300)
    limit_per_source: int = Field(default=5, ge=1, le=20)


class SearchResultItem(BaseModel):
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    paper_url: Optional[str] = None
    pdf_url: Optional[str] = None
    source: str


class SearchResponse(BaseModel):
    topic: str
    total_results: int
    results: List[SearchResultItem]


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
class CompareRequest(BaseModel):
    paper_ids: List[str] = Field(..., min_length=2, max_length=10)


class CompareResponse(BaseModel):
    paper_ids: List[str]
    comparison_report: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------
class GapRequest(BaseModel):
    paper_id: str
    include_related_papers: bool = Field(default=True)
    related_paper_limit: int = Field(default=5, ge=1, le=10)


class GapResponse(BaseModel):
    paper_id: str
    research_gap: str
    missing_work: List[str]
    possible_improvements: List[str]
    novel_directions: List[str]
    suggested_future_work: List[str]


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
class AnalyticsEvent(BaseModel):
    endpoint: str
    model_used: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    timestamp: datetime


class AnalyticsSummary(BaseModel):
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_estimated_cost_usd: float
    average_latency_ms: float
    events: List[AnalyticsEvent]


# ---------------------------------------------------------------------------
# Generic error
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Extended Session Schemas
# ---------------------------------------------------------------------------
class SessionListItem(BaseModel):
    session_id: str
    paper_id: Optional[str] = None
    title: str
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    sessions: List[SessionListItem]


class SessionDetailResponse(BaseModel):
    session_id: str
    paper_id: Optional[str] = None
    uploaded_paper_metadata: Optional[Dict[str, Any]] = None
    uploaded_papers: List[Dict[str, Any]] = Field(default_factory=list)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_papers: List[Dict[str, Any]] = Field(default_factory=list)
    comparison_results: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CreateSessionRequest(BaseModel):
    paper_id: Optional[str] = None


class CompareRecommendedRequest(BaseModel):
    session_id: str
    paper_id: str
    recommended_paper: SearchResultItem


class CompareRecommendedResponse(BaseModel):
    session_id: str
    paper_id: str
    recommended_title: str
    comparison_level: str  # "full_text", "abstract", or "none"
    comparison_report: str
    disclaimer: Optional[str] = None
    generated_at: datetime


class MessageItem(BaseModel):
    role: str
    content: str


class AppendMessagesRequest(BaseModel):
    messages: List[MessageItem]


class SaveRecommendationsRequest(BaseModel):
    papers: List[SearchResultItem]
