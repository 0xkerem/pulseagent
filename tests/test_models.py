"""
Tests for PulseAgent data models.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from models import (
    ClassifiedReview,
    PipelineState,
    Priority,
    Review,
    ReviewCategory,
    ReviewSource,
    RoadmapItem,
    ScoredReview,
    SentimentLabel,
    TrendAlert,
)


class TestReviewModel:
    def test_valid_review_creation(self, sample_review):
        assert sample_review.id == "test-001"
        assert sample_review.source == ReviewSource.FIXTURE
        assert sample_review.rating == 1.0

    def test_review_without_optional_fields(self):
        review = Review(
            id="min-001",
            source=ReviewSource.REDDIT,
            product="TestProd",
            text="Some review text here.",
            date=datetime.now(),
        )
        assert review.author is None
        assert review.rating is None
        assert review.url is None

    def test_review_raw_metadata_defaults_empty(self, sample_review):
        assert isinstance(sample_review.raw_metadata, dict)


class TestClassifiedReviewModel:
    def test_classified_review_inherits_review(self, sample_classified_review):
        assert sample_classified_review.review.id == "test-001"
        assert sample_classified_review.category == ReviewCategory.BUG_REPORT
        assert sample_classified_review.is_churn_signal is True

    def test_sentiment_score_bounds(self):
        with pytest.raises(ValidationError):
            ClassifiedReview(
                review=Review(
                    id="x", source=ReviewSource.FIXTURE,
                    product="P", text="t", date=datetime.now()
                ),
                category=ReviewCategory.BUG_REPORT,
                sentiment=SentimentLabel.NEGATIVE,
                sentiment_score=2.0,   # INVALID: must be <= 1.0
            )

    def test_sentiment_score_lower_bound(self):
        with pytest.raises(ValidationError):
            ClassifiedReview(
                review=Review(
                    id="x", source=ReviewSource.FIXTURE,
                    product="P", text="t", date=datetime.now()
                ),
                category=ReviewCategory.BUG_REPORT,
                sentiment=SentimentLabel.NEGATIVE,
                sentiment_score=-2.0,   # INVALID: must be >= -1.0
            )


class TestScoredReviewModel:
    def test_urgency_score_bounds(self, sample_classified_review):
        with pytest.raises(ValidationError):
            ScoredReview(
                **sample_classified_review.model_dump(),
                urgency_score=11.0,   # INVALID: must be <= 10
                impact_estimate=5.0,
                frequency_weight=1.0,
            )

    def test_valid_scored_review(self, sample_scored_review):
        assert 0.0 <= sample_scored_review.urgency_score <= 10.0
        assert sample_scored_review.frequency_weight >= 1.0


class TestRoadmapItem:
    def test_roadmap_item_creation(self):
        item = RoadmapItem(
            item_id="PULSE-ABC123",
            title="Fix crash on large databases",
            description="Users report crashes when opening databases with 500+ entries.",
            priority=Priority.P0,
            category=ReviewCategory.BUG_REPORT,
            affected_users_estimate=45,
            churn_risk_score=8.0,
            implementation_effort="medium",
            user_story="As a power user, I want databases to load reliably.",
            acceptance_criteria=["No crash on 1000+ entry DB", "Test on all platforms"],
            source_cluster_ids=["clust-001"],
        )
        assert item.priority == Priority.P0
        assert len(item.acceptance_criteria) == 2


class TestPipelineState:
    def test_empty_state_defaults(self, empty_pipeline_state):
        assert empty_pipeline_state.raw_reviews == []
        assert empty_pipeline_state.classified_reviews == []
        assert empty_pipeline_state.errors == []

    def test_populated_state(self, populated_pipeline_state):
        assert len(populated_pipeline_state.raw_reviews) == 2
        assert len(populated_pipeline_state.classified_reviews) == 1
        assert len(populated_pipeline_state.clusters) == 1


class TestTrendAlert:
    def test_trend_alert_directions(self):
        alert = TrendAlert(
            category=ReviewCategory.PERFORMANCE,
            theme="Slow loading times",
            change_percent=75.0,
            direction="rising",
            window_days=14,
            alert_level="critical",
            summary="Performance complaints rose 75% in the last 14 days.",
        )
        assert alert.direction == "rising"
        assert alert.alert_level == "critical"
