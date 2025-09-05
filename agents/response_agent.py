"""
PulseAgent — Response Agent
Generates empathetic, helpful draft responses for top-urgency reviews.
Integrates RAG context to reference actual documentation or changelogs.
All drafts require human approval before sending (human-in-the-loop).
"""
from __future__ import annotations

from textwrap import dedent

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from llm_factory import get_llm
from models import DraftResponse, PipelineState, ScoredReview
from agents.rag_agent import RAGAgent

RESPONSE_SYSTEM = dedent("""
You are a senior customer success manager writing a public response to a product review.

Guidelines:
- Be empathetic and acknowledge the user's experience in the first sentence
- Be specific — reference the exact issue they mentioned
- If the issue is resolved, mention the specific fix (use the provided resolution reference)
- If it's a feature request, thank them and explain next steps
- If it's a churn signal, be especially warm and offer direct support contact
- Never be defensive or dismissive
- Keep it under 150 words
- Sign off as the product team

Tone options: empathetic, professional, friendly — pick based on review sentiment.

Return ONLY the response text, no JSON, no metadata.
""").strip()


def _safe_value(field) -> str:
    """Return .value if it's an enum, else return as string.
    After LangGraph dict round-trip, enum fields may be plain strings."""
    return field.value if hasattr(field, "value") else str(field)


def _extract_content(response) -> str:
    """Safely extract text content from LLM response.
    Gemini sometimes returns a list of content parts instead of a plain string."""
    content = response.content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return " ".join(parts).strip()
    return str(content).strip()


class ResponseAgent:
    """
    LangGraph node: generates draft responses for high-urgency reviews.
    Processes top N reviews by urgency score to stay within rate limits.
    """

    TOP_N = 20
    URGENCY_THRESHOLD = 4.0

    def __init__(self, rag_agent: RAGAgent):
        self.llm = get_llm("response")
        self.rag = rag_agent

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def _generate(self, review: ScoredReview, rag_context_text: str) -> str:
        user_msg = dedent(f"""
        Product: {review.review.product}
        Review (source: {_safe_value(review.review.source)}):
        "{review.review.text[:800]}"

        Category: {_safe_value(review.category)}
        Sentiment: {_safe_value(review.sentiment)} (score: {review.sentiment_score:.2f})
        Is churn signal: {review.is_churn_signal}
        Rating: {review.review.rating if review.review.rating is not None else 'N/A'}

        {f'Documentation context: {rag_context_text}' if rag_context_text else ''}
        """).strip()

        messages = [
            SystemMessage(content=RESPONSE_SYSTEM),
            HumanMessage(content=user_msg),
        ]
        response = await self.llm.ainvoke(messages)
        return _extract_content(response)

    async def run(self, state: PipelineState) -> PipelineState:
        candidates = [
            r for r in state.scored_reviews
            if r.urgency_score >= self.URGENCY_THRESHOLD
        ]
        candidates.sort(key=lambda r: r.urgency_score, reverse=True)
        targets = candidates[: self.TOP_N]

        logger.info(
            f"[ResponseAgent] Generating drafts for {len(targets)} reviews "
            f"(threshold={self.URGENCY_THRESHOLD}, total candidates={len(candidates)})"
        )

        drafts: list[DraftResponse] = []

        for review in targets:
            rag_ctx = await self.rag.query(review.review.text[:500])
            rag_text = ""
            if rag_ctx.retrieved_chunks:
                rag_text = rag_ctx.retrieved_chunks[0][:400]
                if rag_ctx.already_resolved and rag_ctx.resolution_reference:
                    rag_text = f"RESOLVED: {rag_ctx.resolution_reference}"

            try:
                draft_text = await self._generate(review, rag_text)
                drafts.append(DraftResponse(
                    review_id=review.review.id,
                    draft=draft_text,
                    tone="empathetic",
                    requires_human_approval=True,
                    rag_context_used=bool(rag_ctx.retrieved_chunks),
                ))
            except Exception as e:
                err = f"Response generation failed for {review.review.id}: {e}"
                logger.error(err)
                state.errors.append(err)

        state.draft_responses = drafts
        logger.success(f"[ResponseAgent] {len(drafts)} draft responses generated.")
        return state