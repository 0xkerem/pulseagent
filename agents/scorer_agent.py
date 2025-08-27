"""
PulseAgent — Urgency Scorer Agent
Computes an urgency score (0-10) for each classified review.

Formula:
  urgency = (sentiment_weight × 3)
           + (churn_multiplier if churn_signal)
           + (frequency_weight)
           + (impact_estimate × 2)

Also clusters reviews by category+theme using LLM to detect frequency.
"""
from __future__ import annotations

import json
import uuid
from collections import defaultdict
from textwrap import dedent

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import get_settings
from llm_factory import get_llm
from models import (
    ClassifiedReview,
    PipelineState,
    ReviewCategory,
    ReviewCluster,
    ScoredReview,
)

CLUSTER_SYSTEM = dedent("""
You are a product analyst. Group these classified reviews into thematic clusters.
Each cluster should represent a specific recurring issue or request.

For each cluster return:
- theme: short descriptive title (max 8 words)
- review_ids: list of review IDs belonging to this cluster
- top_phrases: up to 5 common phrases across reviews

Return ONLY valid JSON array:
[
  {
    "theme": "<theme>",
    "review_ids": ["<id1>", "<id2>"],
    "top_phrases": ["phrase1", "phrase2"]
  }
]
""").strip()


class UrgencyScorerAgent:
    """
    LangGraph node: scores each review for urgency and groups them into clusters.
    """

    def __init__(self):
        self.llm = get_llm("scorer")
        self.settings = get_settings()

    def _compute_urgency(
        self,
        review: ClassifiedReview,
        frequency_weight: float = 1.0,
        impact_estimate: float = 5.0,
    ) -> float:
        # Sentiment contribution: negative = higher urgency
        sentiment_weight = (1.0 - review.sentiment_score) / 2.0  # 0→1
        base = sentiment_weight * 3.0

        # Churn multiplier
        churn_bonus = 0.0
        if review.is_churn_signal:
            churn_bonus = self.settings.churn_urgency_multiplier

        # Rating penalty (low rating = more urgent)
        rating_penalty = 0.0
        if review.review.rating is not None:
            rating_penalty = max(0, (3.0 - review.review.rating) * 0.5)

        score = base + churn_bonus + rating_penalty + (frequency_weight * 0.5) + (impact_estimate * 0.2)
        return round(min(10.0, max(0.0, score)), 2)

    async def _cluster_category(
        self,
        category: ReviewCategory,
        reviews: list[ClassifiedReview],
    ) -> list[ReviewCluster]:
        """Ask LLM to group reviews of the same category into themes."""
        if len(reviews) == 1:
            # Single review → single cluster
            r = reviews[0]
            return [ReviewCluster(
                cluster_id=str(uuid.uuid4())[:8],
                category=category,
                theme=r.key_phrases[0] if r.key_phrases else category.value,
                reviews=[],      # will be filled after scoring
                total_count=1,
                avg_urgency=0.0,
                churn_risk_count=1 if r.is_churn_signal else 0,
                top_phrases=r.key_phrases,
            )]

        # Prepare compact review list for LLM
        compact = [
            {"id": r.review.id, "text": r.review.text[:300], "phrases": r.key_phrases}
            for r in reviews
        ]
        user_msg = (
            f"Category: {category.value}\n"
            f"Reviews:\n{json.dumps(compact, indent=2)}"
        )

        messages = [
            SystemMessage(content=CLUSTER_SYSTEM),
            HumanMessage(content=user_msg),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            cluster_data: list[dict] = json.loads(raw)
        except Exception as e:
            logger.warning(f"Clustering failed for {category}: {e}. Falling back to single cluster.")
            cluster_data = [{
                "theme": category.value.replace("_", " ").title(),
                "review_ids": [r.review.id for r in reviews],
                "top_phrases": [],
            }]

        # Build a review_id → review map
        id_map = {r.review.id: r for r in reviews}
        clusters: list[ReviewCluster] = []

        for cd in cluster_data:
            cluster_reviews_classified = [
                id_map[rid] for rid in cd.get("review_ids", []) if rid in id_map
            ]
            clusters.append(ReviewCluster(
                cluster_id=str(uuid.uuid4())[:8],
                category=category,
                theme=cd.get("theme", category.value),
                reviews=[],       # scored reviews added later
                total_count=len(cluster_reviews_classified),
                avg_urgency=0.0,
                churn_risk_count=sum(1 for r in cluster_reviews_classified if r.is_churn_signal),
                top_phrases=cd.get("top_phrases", []),
            ))
            # Attach the classified reviews temporarily using a custom attribute
            clusters[-1].__dict__["_classified"] = cluster_reviews_classified

        return clusters

    async def run(self, state: PipelineState) -> PipelineState:
        logger.info(
            f"[UrgencyScorerAgent] Scoring {len(state.classified_reviews)} reviews..."
        )

        # Group by category
        by_category: dict[ReviewCategory, list[ClassifiedReview]] = defaultdict(list)
        for r in state.classified_reviews:
            by_category[r.category].append(r)

        # Cluster within each category
        all_clusters: list[ReviewCluster] = []
        for category, cat_reviews in by_category.items():
            clusters = await self._cluster_category(category, cat_reviews)
            all_clusters.extend(clusters)

        # Compute frequency weights: reviews in large clusters get higher weight
        # Build review_id → cluster size
        rev_to_cluster_size: dict[str, int] = {}
        for cluster in all_clusters:
            classified = cluster.__dict__.get("_classified", [])
            for r in classified:
                rev_to_cluster_size[r.review.id] = cluster.total_count

        # Score every review
        scored: list[ScoredReview] = []
        for cr in state.classified_reviews:
            cluster_size = rev_to_cluster_size.get(cr.review.id, 1)
            freq_weight = min(3.0, 1.0 + (cluster_size - 1) * 0.2)
            impact = 5.0 + min(3.0, cluster_size * 0.3)   # larger cluster = higher impact
            urgency = self._compute_urgency(cr, freq_weight, impact)

            scored.append(ScoredReview(
                **cr.model_dump(),
                urgency_score=urgency,
                impact_estimate=round(impact, 2),
                frequency_weight=round(freq_weight, 2),
                urgency_reasoning=(
                    f"Cluster size={cluster_size}, "
                    f"churn={'yes' if cr.is_churn_signal else 'no'}, "
                    f"sentiment={cr.sentiment_score:.2f}"
                ),
            ))

        state.scored_reviews = scored

        # Finalize clusters: attach ScoredReview objects + compute avg urgency
        scored_map = {s.review.id: s for s in scored}
        for cluster in all_clusters:
            classified = cluster.__dict__.pop("_classified", [])
            cluster.reviews = [
                scored_map[r.review.id]
                for r in classified
                if r.review.id in scored_map
            ]
            if cluster.reviews:
                cluster.avg_urgency = round(
                    sum(r.urgency_score for r in cluster.reviews) / len(cluster.reviews), 2
                )

        state.clusters = sorted(all_clusters, key=lambda c: c.avg_urgency, reverse=True)

        logger.success(
            f"[UrgencyScorerAgent] {len(scored)} reviews scored, "
            f"{len(all_clusters)} clusters formed."
        )
        return state
