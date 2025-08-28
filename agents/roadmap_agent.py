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

import json
import uuid
from textwrap import dedent

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

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


def _assign_priority(cluster: ReviewCluster) -> Priority:
    if cluster.churn_risk_count >= 5 or cluster.avg_urgency >= 8.0:
        return Priority.P0
    if cluster.avg_urgency >= 6.0 or cluster.total_count >= 15:
        return Priority.P1
    if cluster.avg_urgency >= 4.0:
        return Priority.P2
    return Priority.P3


class RoadmapAgent:
    """
    LangGraph node: generates a prioritized product roadmap from review clusters.
    Skips PRAISE clusters (positive feedback → no action needed).
    """

    def __init__(self):
        self.llm = get_llm("roadmap")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def _generate_item(self, cluster: ReviewCluster, priority: Priority) -> RoadmapItem:
        # Build concise cluster summary for LLM
        sample_texts = [r.review.text[:200] for r in cluster.reviews[:5]]
        summary = {
            "category": cluster.category.value,
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
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Roadmap JSON parse failed: {e}")
            parsed = {
                "title": cluster.theme,
                "description": f"Recurring issue: {cluster.theme}",
                "implementation_effort": "medium",
                "user_story": f"As a user, I want {cluster.theme} to be improved.",
                "acceptance_criteria": ["Issue is resolved", "Users confirm fix"],
                "competitor_has_it": None,
            }

        return RoadmapItem(
            item_id=f"PULSE-{str(uuid.uuid4())[:6].upper()}",
            title=parsed["title"],
            description=parsed["description"],
            priority=priority,
            category=cluster.category,
            affected_users_estimate=cluster.total_count,
            churn_risk_score=float(cluster.churn_risk_count),
            implementation_effort=parsed.get("implementation_effort", "medium"),
            user_story=parsed["user_story"],
            acceptance_criteria=parsed.get("acceptance_criteria", []),
            source_cluster_ids=[cluster.cluster_id],
            competitor_has_it=parsed.get("competitor_has_it"),
        )

    async def run(self, state: PipelineState) -> PipelineState:
        # Filter out pure praise clusters — no action needed
        actionable = [
            c for c in state.clusters
            if c.category != ReviewCategory.PRAISE and c.total_count > 0
        ]

        logger.info(
            f"[RoadmapAgent] Generating roadmap for {len(actionable)} actionable clusters..."
        )

        items: list[RoadmapItem] = []
        for cluster in actionable:
            priority = _assign_priority(cluster)
            try:
                item = await self._generate_item(cluster, priority)
                items.append(item)
                logger.debug(
                    f"[RoadmapAgent] {item.item_id} [{priority.value}] — {item.title}"
                )
            except Exception as e:
                err = f"Roadmap generation failed for cluster {cluster.cluster_id}: {e}"
                logger.error(err)
                state.errors.append(err)

        # Sort by priority then churn risk
        priority_order = {Priority.P0: 0, Priority.P1: 1, Priority.P2: 2, Priority.P3: 3}
        items.sort(key=lambda i: (priority_order[i.priority], -i.churn_risk_score))

        state.roadmap_items = items

        # Log summary
        from collections import Counter
        dist = Counter(i.priority.value for i in items)
        logger.success(
            f"[RoadmapAgent] {len(items)} roadmap items generated. "
            f"Priority distribution: {dict(dist)}"
        )
        return state
