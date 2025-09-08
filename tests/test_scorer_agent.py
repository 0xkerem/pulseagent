"""
Tests for UrgencyScorerAgent — urgency formula and priority logic.
These tests mock the LLM to avoid API calls.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.scorer_agent import UrgencyScorerAgent
from models import (
    ClassifiedReview,
    Review,
    ReviewCategory,
    ReviewSource,
    SentimentLabel,
)
from datetime import datetime


def make_classified(
    id: str,
    text: str = "test",
    category: ReviewCategory = ReviewCategory.BUG_REPORT,
    sentiment_score: float = -0.5,
    is_churn_signal: bool = False,
    rating: float | None = None,
) -> ClassifiedReview:
    return ClassifiedReview(
        review=Review(
            id=id,
            source=ReviewSource.FIXTURE,
            product="TestProd",
            text=text,
            date=datetime.now(),
            rating=rating,
        ),
        category=category,
        sentiment=SentimentLabel.NEGATIVE if sentiment_score < 0 else SentimentLabel.POSITIVE,
        sentiment_score=sentiment_score,
        is_churn_signal=is_churn_signal,
        key_phrases=["test phrase"],
    )


class TestUrgencyFormula:
    def setup_method(self):
        with patch("agents.scorer_agent.get_llm"):
            self.agent = UrgencyScorerAgent()

    def test_churn_signal_increases_urgency(self):
        without_churn = self.agent._compute_urgency(
            make_classified("a", is_churn_signal=False, sentiment_score=-0.5)
        )
        with_churn = self.agent._compute_urgency(
            make_classified("b", is_churn_signal=True, sentiment_score=-0.5)
        )
        assert with_churn > without_churn

    def test_negative_sentiment_higher_urgency_than_positive(self):
        negative = self.agent._compute_urgency(
            make_classified("a", sentiment_score=-0.9)
        )
        positive = self.agent._compute_urgency(
            make_classified("b", sentiment_score=0.9)
        )
        assert negative > positive

    def test_urgency_bounded_0_to_10(self):
        extreme = self.agent._compute_urgency(
            make_classified("a", sentiment_score=-1.0, is_churn_signal=True, rating=1.0),
            frequency_weight=3.0,
            impact_estimate=10.0,
        )
        assert 0.0 <= extreme <= 10.0

    def test_low_rating_increases_urgency(self):
        high_rating = self.agent._compute_urgency(
            make_classified("a", rating=5.0, sentiment_score=-0.3)
        )
        low_rating = self.agent._compute_urgency(
            make_classified("b", rating=1.0, sentiment_score=-0.3)
        )
        assert low_rating > high_rating

    def test_frequency_weight_increases_urgency(self):
        low_freq = self.agent._compute_urgency(
            make_classified("a"), frequency_weight=1.0
        )
        high_freq = self.agent._compute_urgency(
            make_classified("b"), frequency_weight=3.0
        )
        assert high_freq > low_freq


class TestScoreragentRun:
    @pytest.mark.asyncio
    async def test_run_produces_scored_reviews(self, populated_pipeline_state):
        with patch("agents.scorer_agent.get_llm") as mock_llm:
            # Mock LLM clustering response
            mock_instance = AsyncMock()
            mock_instance.ainvoke.return_value = MagicMock(
                content='[{"theme": "Test theme", "review_ids": ["test-001"], "top_phrases": ["crash"]}]'
            )
            mock_llm.return_value = mock_instance

            agent = UrgencyScorerAgent()
            result = await agent.run(populated_pipeline_state)

        assert len(result.scored_reviews) == len(populated_pipeline_state.classified_reviews)
        for r in result.scored_reviews:
            assert 0.0 <= r.urgency_score <= 10.0

    @pytest.mark.asyncio
    async def test_run_creates_clusters(self, populated_pipeline_state):
        with patch("agents.scorer_agent.get_llm") as mock_llm:
            mock_instance = AsyncMock()
            mock_instance.ainvoke.return_value = MagicMock(
                content='[{"theme": "Bug cluster", "review_ids": ["test-001"], "top_phrases": []}]'
            )
            mock_llm.return_value = mock_instance

            agent = UrgencyScorerAgent()
            result = await agent.run(populated_pipeline_state)

        assert len(result.clusters) >= 1
        for cluster in result.clusters:
            assert cluster.avg_urgency >= 0.0
