"""
PulseAgent — Classifier Agent
Classifies each review into a category and extracts sentiment using Groq (fast).
Processes reviews in batches to stay within rate limits.
"""
from __future__ import annotations

import json
from textwrap import dedent

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from llm_factory import get_llm
from models import (
    ClassifiedReview,
    PipelineState,
    Review,
    ReviewCategory,
    SentimentLabel,
)



def _safe_value(field) -> str:
    """Return .value if it's an enum, else return as string."""
    return field.value if hasattr(field, "value") else str(field)

SYSTEM_PROMPT = dedent("""
You are a product review analyst. Analyze the given user review and return a JSON object.

Categories:
- bug_report: crashes, errors, broken features
- feature_request: asking for new functionality
- ux_friction: confusing UI, hard to use, bad workflow
- performance: slow, laggy, high battery/memory usage
- pricing_complaint: too expensive, bad value, billing issues
- praise: positive feedback, compliments
- churn_signal: user threatening to leave, cancel, switch to competitor
- integration_issue: problems with third-party integrations, APIs
- documentation: missing docs, unclear instructions
- other: doesn't fit above

Sentiment: positive | neutral | negative | mixed
Sentiment score: float from -1.0 (very negative) to 1.0 (very positive)
is_churn_signal: true if user explicitly mentions leaving, cancelling, or switching

Return ONLY valid JSON, no markdown, no explanation:
{
  "category": "<category>",
  "sentiment": "<sentiment>",
  "sentiment_score": <float>,
  "is_churn_signal": <bool>,
  "key_phrases": ["<phrase1>", "<phrase2>"],
  "reasoning": "<one sentence explanation>"
}
""").strip()


class ClassifierAgent:
    """
    LangGraph node: classifies all raw reviews.
    Uses Groq for speed — processes in small batches.
    """

    BATCH_SIZE = 10

    def __init__(self):
        self.llm = get_llm("classifier")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def _classify_one(self, review: Review) -> ClassifiedReview:
        # Use .value to ensure enum serializes to plain string for LLM
        user_msg = (
            f"Product: {review.product}\n"
            f"Source: {_safe_value(review.source)}\n"
            f"Rating: {review.rating if review.rating is not None else 'N/A'}\n"
            f"Review text: {review.text[:1500]}"
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        response = await self.llm.ainvoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for review {review.id}: {e}\nRaw: {raw[:200]}")
            # Fallback: safe defaults
            parsed = {
                "category": "other",
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "is_churn_signal": False,
                "key_phrases": [],
                "reasoning": "Parse error — defaulted.",
            }

        # Validate enums safely
        try:
            category = ReviewCategory(parsed["category"])
        except ValueError:
            category = ReviewCategory.OTHER

        try:
            sentiment = SentimentLabel(parsed["sentiment"])
        except ValueError:
            sentiment = SentimentLabel.NEUTRAL

        return ClassifiedReview(
            review=review,
            category=category,
            sentiment=sentiment,
            sentiment_score=float(parsed.get("sentiment_score", 0.0)),
            is_churn_signal=bool(parsed.get("is_churn_signal", False)),
            key_phrases=parsed.get("key_phrases", []),
            classifier_reasoning=parsed.get("reasoning", ""),
        )

    async def run(self, state: PipelineState) -> PipelineState:
        logger.info(
            f"[ClassifierAgent] Classifying {len(state.raw_reviews)} reviews..."
        )
        classified: list[ClassifiedReview] = []
        errors = 0

        for i, review in enumerate(state.raw_reviews):
            try:
                result = await self._classify_one(review)
                classified.append(result)
                if (i + 1) % 10 == 0:
                    logger.debug(f"[ClassifierAgent] Progress: {i+1}/{len(state.raw_reviews)}")
            except Exception as e:
                # Log the full error detail so we can debug API issues
                err_detail = str(e)
                if hasattr(e, 'response'):
                    try:
                        err_detail += f" | Response: {e.response.text[:300]}"
                    except Exception:
                        pass
                err = f"Classification failed for review {review.id}: {err_detail}"
                logger.error(err)
                state.errors.append(err)
                errors += 1

        state.classified_reviews = classified
        logger.success(
            f"[ClassifierAgent] Done. {len(classified)} classified, {errors} failed."
        )

        # Log category distribution
        from collections import Counter
        dist = Counter(_safe_value(r.category) for r in classified)
        logger.info(f"[ClassifierAgent] Category distribution: {dict(dist)}")

        return state