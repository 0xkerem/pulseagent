"""
Shared pytest fixtures for PulseAgent test suite.
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models import (
    ClassifiedReview,
    PipelineState,
    Priority,
    Review,
    ReviewCategory,
    ReviewCluster,
    ReviewSource,
    ScoredReview,
    SentimentLabel,
)


@pytest.fixture
def sample_review() -> Review:
    return Review(
        id="test-001",
        source=ReviewSource.FIXTURE,
        product="TestProduct",
        author="test_user",
        text="The app crashes every time I open a large database. Very frustrating, considering switching to a competitor.",
        rating=1.0,
        date=datetime(2024, 10, 15, 10, 0, 0),
    )


@pytest.fixture
def sample_review_positive() -> Review:
    return Review(
        id="test-002",
        source=ReviewSource.FIXTURE,
        product="TestProduct",
        author="happy_user",
        text="Absolutely love this product! Best tool I've used in years. The collaboration features are fantastic.",
        rating=5.0,
        date=datetime(2024, 10, 16, 10, 0, 0),
    )


@pytest.fixture
def sample_classified_review(sample_review) -> ClassifiedReview:
    return ClassifiedReview(
        review=sample_review,
        category=ReviewCategory.BUG_REPORT,
        sentiment=SentimentLabel.NEGATIVE,
        sentiment_score=-0.8,
        is_churn_signal=True,
        key_phrases=["crashes", "large database", "switching to competitor"],
        classifier_reasoning="User reports a crash bug and expresses intent to churn.",
    )


@pytest.fixture
def sample_scored_review(sample_classified_review) -> ScoredReview:
    return ScoredReview(
        **sample_classified_review.model_dump(),
        urgency_score=8.5,
        impact_estimate=7.0,
        frequency_weight=1.5,
        urgency_reasoning="High urgency: churn signal + negative sentiment + crash",
    )


@pytest.fixture
def sample_cluster(sample_scored_review) -> ReviewCluster:
    return ReviewCluster(
        cluster_id="clust-001",
        category=ReviewCategory.BUG_REPORT,
        theme="App crashes on large databases",
        reviews=[sample_scored_review],
        total_count=1,
        avg_urgency=8.5,
        churn_risk_count=1,
        top_phrases=["crashes", "large database"],
    )


@pytest.fixture
def empty_pipeline_state() -> PipelineState:
    return PipelineState(
        product_name="TestProduct",
        run_id="test-run-001",
        started_at=datetime.now(),
    )


@pytest.fixture
def populated_pipeline_state(
    sample_review,
    sample_review_positive,
    sample_classified_review,
    sample_scored_review,
    sample_cluster,
) -> PipelineState:
    return PipelineState(
        product_name="TestProduct",
        run_id="test-run-002",
        started_at=datetime.now(),
        raw_reviews=[sample_review, sample_review_positive],
        classified_reviews=[sample_classified_review],
        scored_reviews=[sample_scored_review],
        clusters=[sample_cluster],
    )
