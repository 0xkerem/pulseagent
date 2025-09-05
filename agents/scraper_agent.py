"""
PulseAgent — Scraper Agent
Collects reviews from multiple sources.
In production: uses Playwright for dynamic scraping + httpx for APIs.
In demo/CI: loads from fixture JSON files.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from models import Review, ReviewSource, PipelineState


# ─── Source Scrapers ────────────────────────────────────────────────────────

class FixtureScraper:
    """Loads pre-saved review fixtures for demo/testing."""

    def __init__(self, fixture_dir: str = "./data/fixtures"):
        self.fixture_dir = Path(fixture_dir)

    def scrape(self, product: str, limit: int = 100) -> list[Review]:
        fixture_file = self.fixture_dir / f"{product.lower().replace(' ', '_')}.json"

        if not fixture_file.exists():
            logger.warning(f"No fixture found for '{product}', using default.")
            fixture_file = self.fixture_dir / "default.json"

        if not fixture_file.exists():
            logger.error("No fixture files found at all.")
            return []

        with open(fixture_file) as f:
            raw: list[dict] = json.load(f)

        reviews = []
        for item in raw[:limit]:
            reviews.append(Review(
                id=item.get("id", str(uuid.uuid4())),
                source=ReviewSource(item.get("source", "fixture")),
                product=product,
                author=item.get("author"),
                text=item["text"],
                rating=item.get("rating"),
                date=datetime.fromisoformat(item.get("date", datetime.now().isoformat())),
                url=item.get("url"),
                raw_metadata=item.get("metadata", {}),
            ))

        logger.info(f"[Scraper] Loaded {len(reviews)} reviews from fixture: {fixture_file.name}")
        return reviews


class RedditScraper:
    """Scrapes Reddit posts/comments via public JSON API (no auth required)."""

    BASE_URL = "https://www.reddit.com"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def scrape_subreddit(
        self,
        subreddit: str,
        query: str,
        limit: int = 25,
    ) -> list[Review]:
        url = f"{self.BASE_URL}/r/{subreddit}/search.json"
        params = {"q": query, "limit": limit, "sort": "new", "restrict_sr": "on"}
        headers = {"User-Agent": "PulseAgent/0.1 (review intelligence bot)"}

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        reviews = []
        for post in data.get("data", {}).get("children", []):
            pd = post["data"]
            text = pd.get("selftext") or pd.get("title", "")
            if len(text.strip()) < 20:
                continue
            reviews.append(Review(
                id=pd["id"],
                source=ReviewSource.REDDIT,
                product=query,
                author=pd.get("author"),
                text=text[:2000],
                rating=None,
                date=datetime.fromtimestamp(pd["created_utc"]),
                url=f"{self.BASE_URL}{pd.get('permalink', '')}",
                raw_metadata={"score": pd.get("score", 0), "subreddit": subreddit},
            ))

        logger.info(f"[Scraper] Reddit r/{subreddit}: {len(reviews)} posts fetched")
        return reviews


class AppStoreScraper:
    """Scrapes App Store reviews via public RSS feed."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def scrape(self, app_id: str, product: str, country: str = "us", limit: int = 50) -> list[Review]:
        url = (
            f"https://itunes.apple.com/{country}/rss/customerreviews/"
            f"page=1/id={app_id}/sortby=mostrecent/json"
        )
        headers = {"User-Agent": "PulseAgent/0.1"}

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        reviews = []
        entries = data.get("feed", {}).get("entry", [])
        for entry in entries[1:limit+1]:  # first entry is app metadata
            try:
                reviews.append(Review(
                    id=entry["id"]["label"],
                    source=ReviewSource.APP_STORE,
                    product=product,
                    author=entry["author"]["name"]["label"],
                    text=entry["content"]["label"][:2000],
                    rating=float(entry["im:rating"]["label"]),
                    date=datetime.fromisoformat(
                        entry["updated"]["label"].replace("Z", "+00:00")
                    ),
                    url=entry.get("link", {}).get("attributes", {}).get("href"),
                ))
            except (KeyError, ValueError) as e:
                logger.debug(f"Skipping malformed entry: {e}")

        logger.info(f"[Scraper] App Store app_id={app_id}: {len(reviews)} reviews fetched")
        return reviews


# ─── Agent Node ─────────────────────────────────────────────────────────────

class ScraperAgent:
    """
    LangGraph node: collects reviews and writes them to PipelineState.
    Uses fixture scraper by default; switches to live scrapers when configured.
    """

    def __init__(
        self,
        use_fixtures: bool = True,
        fixture_dir: str = "./data/fixtures",
        reddit_subreddits: Optional[list[str]] = None,
        app_store_app_id: Optional[str] = None,
        limit: int = 100,
    ):
        self.use_fixtures = use_fixtures
        self.fixture_scraper = FixtureScraper(fixture_dir)
        self.reddit_scraper = RedditScraper()
        self.app_store_scraper = AppStoreScraper()
        self.reddit_subreddits = reddit_subreddits or []
        self.app_store_app_id = app_store_app_id
        self.limit = limit

    async def run(self, state: PipelineState) -> PipelineState:
        logger.info(f"[ScraperAgent] Starting for product: '{state.product_name}'")
        all_reviews: list[Review] = []

        if self.use_fixtures:
            all_reviews = self.fixture_scraper.scrape(state.product_name, self.limit)
        else:
            # Reddit live scrape
            for sub in self.reddit_subreddits:
                try:
                    reviews = await self.reddit_scraper.scrape_subreddit(
                        subreddit=sub,
                        query=state.product_name,
                        limit=min(25, self.limit),
                    )
                    all_reviews.extend(reviews)
                except Exception as e:
                    err = f"Reddit scraper failed for r/{sub}: {e}"
                    logger.error(err)
                    state.errors.append(err)

            # App Store live scrape
            if self.app_store_app_id:
                try:
                    reviews = await self.app_store_scraper.scrape(
                        app_id=self.app_store_app_id,
                        product=state.product_name,
                        limit=min(50, self.limit),
                    )
                    all_reviews.extend(reviews)
                except Exception as e:
                    err = f"App Store scraper failed: {e}"
                    logger.error(err)
                    state.errors.append(err)

        # Deduplicate by review ID
        seen: set[str] = set()
        unique: list[Review] = []
        for r in all_reviews:
            if r.id not in seen:
                seen.add(r.id)
                unique.append(r)

        state.raw_reviews = unique[: self.limit]
        logger.success(
            f"[ScraperAgent] Collected {len(state.raw_reviews)} unique reviews."
        )
        return state