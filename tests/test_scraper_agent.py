"""
Tests for ScraperAgent — focuses on fixture loading and deduplication.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents.scraper_agent import FixtureScraper, ScraperAgent
from models import PipelineState, ReviewSource


class TestFixtureScraper:
    FIXTURE_DIR = str(ROOT / "data" / "fixtures")

    def test_loads_notion_fixture(self):
        scraper = FixtureScraper(self.FIXTURE_DIR)
        reviews = scraper.scrape("notion", limit=100)
        assert len(reviews) > 0

    def test_respects_limit(self):
        scraper = FixtureScraper(self.FIXTURE_DIR)
        reviews = scraper.scrape("notion", limit=5)
        assert len(reviews) <= 5

    def test_falls_back_to_default(self):
        scraper = FixtureScraper(self.FIXTURE_DIR)
        reviews = scraper.scrape("nonexistent_product_xyz", limit=100)
        assert len(reviews) > 0

    def test_review_fields_populated(self):
        scraper = FixtureScraper(self.FIXTURE_DIR)
        reviews = scraper.scrape("notion", limit=1)
        r = reviews[0]
        assert r.id is not None
        assert r.text != ""
        assert r.date is not None
        assert r.product == "notion"

    def test_missing_fixture_dir_returns_empty(self):
        scraper = FixtureScraper("/nonexistent/path")
        reviews = scraper.scrape("notion", limit=10)
        assert reviews == []


class TestScraperAgent:
    FIXTURE_DIR = str(ROOT / "data" / "fixtures")

    @pytest.mark.asyncio
    async def test_scraper_populates_state(self, empty_pipeline_state):
        agent = ScraperAgent(
            use_fixtures=True,
            fixture_dir=self.FIXTURE_DIR,
            limit=10,
        )
        empty_pipeline_state.product_name = "notion"
        result = await agent.run(empty_pipeline_state)
        assert len(result.raw_reviews) > 0
        assert len(result.raw_reviews) <= 10

    @pytest.mark.asyncio
    async def test_scraper_deduplicates_reviews(self, empty_pipeline_state):
        """Reviews with duplicate IDs should be deduplicated."""
        agent = ScraperAgent(
            use_fixtures=True,
            fixture_dir=self.FIXTURE_DIR,
            limit=100,
        )
        empty_pipeline_state.product_name = "notion"
        result = await agent.run(empty_pipeline_state)
        ids = [r.id for r in result.raw_reviews]
        assert len(ids) == len(set(ids)), "Duplicate review IDs found!"

    @pytest.mark.asyncio
    async def test_scraper_sets_product_name(self, empty_pipeline_state):
        agent = ScraperAgent(
            use_fixtures=True,
            fixture_dir=self.FIXTURE_DIR,
            limit=5,
        )
        empty_pipeline_state.product_name = "notion"
        result = await agent.run(empty_pipeline_state)
        for review in result.raw_reviews:
            assert review.product == "notion"
