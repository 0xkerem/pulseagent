"""
PulseAgent — Trend Agent
Detects rising and falling issue trends by comparing current window
vs. previous window. Generates natural language alerts.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from textwrap import dedent

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from config import get_settings
from llm_factory import get_llm
from models import PipelineState, ReviewCategory, ScoredReview, TrendAlert

TREND_SYSTEM = dedent("""
You are a product analyst writing a concise trend alert.
Given a category trend, write a 1-2 sentence alert summary.
Focus on business impact. Be specific about the change.
Return ONLY the summary text, no JSON.
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


def _extract_content(response) -> str:
    """Safely extract string content from LLM response."""
    content = response.content
    if isinstance(content, list):
        parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
        return " ".join(parts).strip()
    return str(content).strip()


def _window_counts(
    reviews: list[ScoredReview],
    window_days: int,
    offset_days: int = 0,
) -> dict[str, int]:
    """Count reviews per category (as string) within a time window."""
    now = datetime.now()
    start = now - timedelta(days=window_days + offset_days)
    end = now - timedelta(days=offset_days)
    counts: dict[str, int] = defaultdict(int)
    for r in reviews:
        review_date = r.review.date
        # Handle both datetime objects and ISO strings
        if isinstance(review_date, str):
            review_date = datetime.fromisoformat(review_date)
        review_date = review_date.replace(tzinfo=None)
        if start <= review_date <= end:
            counts[_safe_value(r.category)] += 1
    return counts


class TrendAgent:
    """
    LangGraph node: compares current vs. previous window and
    generates TrendAlert objects for significant changes.
    """

    CHANGE_THRESHOLD = 0.30
    MIN_VOLUME = 3

    def __init__(self):
        self.llm = get_llm("trend")
        self.settings = get_settings()

    def _alert_level(self, change_pct: float, churn_involved: bool) -> str:
        abs_change = abs(change_pct)
        if abs_change >= 1.0 or churn_involved:
            return "critical"
        if abs_change >= 0.5:
            return "warning"
        return "info"

    async def _generate_summary(
        self,
        category_str: str,
        direction: str,
        change_pct: float,
        theme: str,
    ) -> str:
        msg = (
            f"Category: {category_str}\n"
            f"Direction: {direction}\n"
            f"Change: {change_pct*100:.0f}%\n"
            f"Theme: {theme}"
        )
        messages = [
            SystemMessage(content=TREND_SYSTEM),
            HumanMessage(content=msg),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            return _extract_content(response)
        except Exception:
            direction_word = "increased" if direction == "rising" else "decreased"
            return (
                f"{category_str.replace('_', ' ').title()} complaints "
                f"have {direction_word} by {change_pct*100:.0f}% "
                f"in the past {self.settings.trend_window_days} days."
            )

    async def run(self, state: PipelineState) -> PipelineState:
        window = self.settings.trend_window_days
        logger.info(f"[TrendAgent] Analyzing trends over {window}-day window...")

        current = _window_counts(state.scored_reviews, window_days=window, offset_days=0)
        previous = _window_counts(state.scored_reviews, window_days=window, offset_days=window)

        alerts: list[TrendAlert] = []

        all_cat_strs = set(current.keys()) | set(previous.keys())
        for cat_str in all_cat_strs:
            curr_count = current.get(cat_str, 0)
            prev_count = previous.get(cat_str, 0)

            if curr_count < self.MIN_VOLUME and prev_count < self.MIN_VOLUME:
                continue

            if prev_count == 0:
                change_pct = 1.0 if curr_count > 0 else 0.0
            else:
                change_pct = (curr_count - prev_count) / prev_count

            if abs(change_pct) < self.CHANGE_THRESHOLD:
                continue

            direction = "rising" if change_pct > 0 else "falling"
            category = _safe_category(cat_str)

            cat_reviews = [
                r for r in state.scored_reviews
                if _safe_value(r.category) == cat_str
            ]
            churn_in_cat = any(r.is_churn_signal for r in cat_reviews)

            theme = cat_str.replace("_", " ").title()
            for cluster in state.clusters:
                if _safe_value(cluster.category) == cat_str:
                    theme = cluster.theme
                    break

            summary = await self._generate_summary(cat_str, direction, change_pct, theme)
            level = self._alert_level(change_pct, churn_in_cat)

            alerts.append(TrendAlert(
                category=category,
                theme=theme,
                change_percent=round(change_pct * 100, 1),
                direction=direction,
                window_days=window,
                alert_level=level,
                summary=summary,
            ))

            logger.info(
                f"[TrendAgent] [{level.upper()}] {cat_str}: "
                f"{direction} {change_pct*100:.0f}%"
            )

        alerts.sort(key=lambda a: (
            {"critical": 0, "warning": 1, "info": 2}[a.alert_level],
            -abs(a.change_percent),
        ))

        state.trend_alerts = alerts
        logger.success(f"[TrendAgent] {len(alerts)} trend alerts generated.")
        return state