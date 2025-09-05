"""
PulseAgent — Roadmap Agent
Transforms review clusters into actionable, prioritized product roadmap items.
Outputs Jira-ready user stories with acceptance criteria.

Priority algorithm:
  P0: churn_risk > 5 OR avg_urgency > 8
  P1: avg_urgency > 6 OR cluster_size > 15
  P2: avg_urgency > 4
  P3: everything else
"""
from __future__ import annotations

import asyncio
import json
import uuid
from textwrap import dedent

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_factory import get_llm
from models import (
    PipelineState,
    Priority,
    ReviewCategory,
    ReviewCluster,
    RoadmapItem,
)

ROADMAP_SYSTEM = dedent("""
You are a senior product manager translating user feedback into actionable backlog items.

Given a review cluster summary, generate a roadmap item in JSON:
{
  "title": "<short action title, max 10 words>",
  "description": "<what the problem is and why it matters, 2-3 sentences>",
  "implementation_effort": "low|medium|high",
  "user_story": "As a <type of user>, I want <goal> so that <benefit>.",
  "acceptance_criteria": [
    "<measurable criterion 1>",
    "<measurable criterion 2>",
    "<measurable criterion 3>"
  ],
  "competitor_has_it": <true|false|null>
}

Be specific and actionable. Acceptance criteria must be testable.
Return ONLY the JSON object, no extra text.
""").strip()


def _safe_value(field) -> str:
    """Return .value if it's an enum, else return as string."""
    return field.value if hasattr(field, "value") else str(field)


def _safe_category(field) -> ReviewCategory:
    """Coerce string or enum to ReviewCategory."""
    if isinstance(field, ReviewCategory):
        return field
    try:
        return ReviewCategory(str(field))
    except ValueError:
        return ReviewCategory.OTHER


def _assign_priority(cluster: ReviewCluster) -> Priority:
    if cluster.churn_risk_count >= 5 or cluster.avg_urgency >= 8.0:
        return Priority.P0
    if cluster.avg_urgency >= 6.0 or cluster.total_count >= 15:
        return Priority.P1
    if cluster.avg_urgency >= 4.0:
        return Priority.P2
    return Priority.P3


def _extract_content(response) -> str:
    """Safely extract string content from LLM response."""
    content = response.content
    if isinstance(content, list):
        parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
        return " ".join(parts).strip()
    return str(content).strip()


def _fallback_item(cluster: ReviewCluster, priority: Priority) -> RoadmapItem:
    """Generate a basic roadmap item without LLM when API fails."""
    cat_str = _safe_value(cluster.category)
    return RoadmapItem(
        item_id=f"PULSE-{str(uuid.uuid4())[:6].upper()}",
        title=cluster.theme[:60] if cluster.theme else f"Improve {cat_str}",
        description=(
            f"Users have reported issues related to: {cluster.theme}. "
            f"This affects approximately {cluster.total_count} users "
            f"with {cluster.churn_risk_count} churn signals detected."
        ),
        priority=priority,
        category=_safe_category(cluster.category),
        affected_users_estimate=cluster.total_count,
        churn_risk_score=float(cluster.churn_risk_count),
        implementation_effort="medium",
        user_story=f"As a user, I want {cluster.theme} so that my experience improves.",
        acceptance_criteria=[
            f"Issue '{cluster.theme}' is resolved",
            "Affected users confirm the fix",
            "No regression in related features",
        ],
        source_cluster_ids=[cluster.cluster_id],
        competitor_has_it=None,
    )


class RoadmapAgent:
    """
    LangGraph node: generates a prioritized product roadmap from review clusters.
    Skips PRAISE clusters (positive feedback — no action needed).
    """

    # Delay between API calls to avoid rate limiting (seconds)
    INTER_REQUEST_DELAY = 2.0

    def __init__(self):
        self.llm = get_llm("roadmap")

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=False,
    )
    async def _generate_item_with_retry(self, cluster: ReviewCluster, priority: Priority) -> RoadmapItem | None:
        """Attempt LLM generation — returns None on unrecoverable failure."""
        cat_str = _safe_value(cluster.category)
        sample_texts = [r.review.text[:200] for r in cluster.reviews[:5]]
        summary = {
            "category": cat_str,
            "theme": cluster.theme,
            "total_reviews": cluster.total_count,
            "avg_urgency": cluster.avg_urgency,
            "churn_signals": cluster.churn_risk_count,
            "top_phrases": cluster.top_phrases,
            "sample_reviews": sample_texts,
        }

        messages = [
            SystemMessage(content=ROADMAP_SYSTEM),
            HumanMessage(content=f"Cluster summary:\n{json.dumps(summary, indent=2)}"),
        ]

        response = await self.llm.ainvoke(messages)
        raw = _extract_content(response)

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)   # raises JSONDecodeError → triggers retry

        return RoadmapItem(
            item_id=f"PULSE-{str(uuid.uuid4())[:6].upper()}",
            title=parsed["title"],
            description=parsed["description"],
            priority=priority,
            category=_safe_category(cluster.category),
            affected_users_estimate=cluster.total_count,
            churn_risk_score=float(cluster.churn_risk_count),
            implementation_effort=parsed.get("implementation_effort", "medium"),
            user_story=parsed["user_story"],
            acceptance_criteria=parsed.get("acceptance_criteria", []),
            source_cluster_ids=[cluster.cluster_id],
            competitor_has_it=parsed.get("competitor_has_it"),
        )

    async def run(self, state: PipelineState) -> PipelineState:
        actionable = [
            c for c in state.clusters
            if _safe_value(c.category) != ReviewCategory.PRAISE.value
            and c.total_count > 0
        ]

        logger.info(
            f"[RoadmapAgent] Generating roadmap for {len(actionable)} actionable clusters..."
        )

        items: list[RoadmapItem] = []

        for i, cluster in enumerate(actionable):
            priority = _assign_priority(cluster)

            # Small delay between requests to respect Gemini rate limits
            if i > 0:
                await asyncio.sleep(self.INTER_REQUEST_DELAY)

            try:
                item = await self._generate_item_with_retry(cluster, priority)
                if item is None:
                    raise ValueError("Retry returned None")
            except Exception as e:
                logger.warning(
                    f"[RoadmapAgent] LLM failed for cluster '{cluster.theme}' "
                    f"({e.__class__.__name__}) — using fallback item."
                )
                item = _fallback_item(cluster, priority)

            items.append(item)
            logger.debug(f"[RoadmapAgent] {item.item_id} [{priority.value}] — {item.title}")

        priority_order = {Priority.P0: 0, Priority.P1: 1, Priority.P2: 2, Priority.P3: 3}
        items.sort(key=lambda i: (priority_order[i.priority], -i.churn_risk_score))

        state.roadmap_items = items

        from collections import Counter
        dist = Counter(_safe_value(i.priority) for i in items)
        logger.success(
            f"[RoadmapAgent] {len(items)} roadmap items generated. "
            f"Priority distribution: {dict(dist)}"
        )
        return state