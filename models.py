"""
PulseAgent — Shared Data Models
Pydantic models that flow through the entire agent graph.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


# ─── Enums ──────────────────────────────────────────────────────────────────

class ReviewCategory(str, Enum):
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    UX_FRICTION = "ux_friction"
    PERFORMANCE = "performance"
    PRICING_COMPLAINT = "pricing_complaint"
    PRAISE = "praise"
    CHURN_SIGNAL = "churn_signal"
    INTEGRATION_ISSUE = "integration_issue"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class ReviewSource(str, Enum):
    REDDIT = "reddit"
    APP_STORE = "app_store"
    GOOGLE_PLAY = "google_play"
    TRUSTPILOT = "trustpilot"
    TWITTER = "twitter"
    HACKERNEWS = "hackernews"
    G2 = "g2"
    FIXTURE = "fixture"


class Priority(str, Enum):
    P0 = "P0"  # Critical — act immediately
    P1 = "P1"  # High — next sprint
    P2 = "P2"  # Medium — backlog
    P3 = "P3"  # Low — nice to have


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


# ─── Core Review ────────────────────────────────────────────────────────────

class Review(BaseModel):
    id: str
    source: ReviewSource
    product: str
    author: Optional[str] = None
    text: str
    rating: Optional[float] = None          # 1–5
    date: datetime
    url: Optional[str] = None
    raw_metadata: dict = Field(default_factory=dict)


class ClassifiedReview(BaseModel):
    review: Review
    category: ReviewCategory
    sentiment: SentimentLabel
    sentiment_score: float = Field(ge=-1.0, le=1.0)   # -1 very negative → +1 very positive
    is_churn_signal: bool = False
    key_phrases: list[str] = Field(default_factory=list)
    classifier_reasoning: str = ""


class ScoredReview(ClassifiedReview):
    urgency_score: float = Field(ge=0.0, le=10.0)
    impact_estimate: float = Field(ge=0.0, le=10.0)
    frequency_weight: float = 1.0           # boosted when same issue appears many times
    urgency_reasoning: str = ""


# ─── Aggregated Cluster ─────────────────────────────────────────────────────

class ReviewCluster(BaseModel):
    cluster_id: str
    category: ReviewCategory
    theme: str                              # e.g. "Slow loading on mobile"
    reviews: list[ScoredReview]
    total_count: int
    avg_urgency: float
    churn_risk_count: int
    top_phrases: list[str] = Field(default_factory=list)
    trend_delta: Optional[float] = None    # % change vs previous window


# ─── RAG Response ───────────────────────────────────────────────────────────

class RAGContext(BaseModel):
    query: str
    retrieved_chunks: list[str]
    source_files: list[str]
    already_resolved: bool = False
    resolution_reference: Optional[str] = None


class DraftResponse(BaseModel):
    review_id: str
    draft: str
    tone: str = "empathetic"
    requires_human_approval: bool = True
    rag_context_used: bool = False


# ─── Roadmap ────────────────────────────────────────────────────────────────

class RoadmapItem(BaseModel):
    item_id: str
    title: str
    description: str
    priority: Priority
    category: ReviewCategory
    affected_users_estimate: int
    churn_risk_score: float
    implementation_effort: str             # "low" | "medium" | "high"
    user_story: str                        # "As a user, I want..."
    acceptance_criteria: list[str]
    source_cluster_ids: list[str]
    competitor_has_it: Optional[bool] = None


# ─── Trend ──────────────────────────────────────────────────────────────────

class TrendAlert(BaseModel):
    category: ReviewCategory
    theme: str
    change_percent: float
    direction: str                         # "rising" | "falling"
    window_days: int
    alert_level: str                       # "info" | "warning" | "critical"
    summary: str


# ─── Pipeline State (LangGraph) ─────────────────────────────────────────────

class PipelineState(BaseModel):
    """Shared mutable state passed between all LangGraph nodes."""
    # Input
    product_name: str = ""
    raw_reviews: list[Review] = Field(default_factory=list)

    # Intermediate
    classified_reviews: list[ClassifiedReview] = Field(default_factory=list)
    scored_reviews: list[ScoredReview] = Field(default_factory=list)
    clusters: list[ReviewCluster] = Field(default_factory=list)

    # Outputs
    draft_responses: list[DraftResponse] = Field(default_factory=list)
    roadmap_items: list[RoadmapItem] = Field(default_factory=list)
    trend_alerts: list[TrendAlert] = Field(default_factory=list)

    # Meta
    run_id: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)