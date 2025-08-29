"""
PulseAgent — LangGraph Pipeline
Wires all agents into a stateful directed graph.

Graph flow:
  scraper → classifier → scorer → [rag + response] → roadmap → trend → END
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from agents.classifier_agent import ClassifierAgent
from agents.rag_agent import RAGAgent
from agents.response_agent import ResponseAgent
from agents.roadmap_agent import RoadmapAgent
from agents.scorer_agent import UrgencyScorerAgent
from agents.scraper_agent import ScraperAgent
from agents.trend_agent import TrendAgent
from models import PipelineState


# LangGraph requires a TypedDict or dict as the state schema.
# We use plain dict and serialize/deserialize PipelineState manually.
class GraphState(TypedDict, total=False):
    product_name: str
    run_id: str
    raw_reviews: list
    classified_reviews: list
    scored_reviews: list
    clusters: list
    draft_responses: list
    roadmap_items: list
    trend_alerts: list
    started_at: Optional[str]
    completed_at: Optional[str]
    errors: list

def _state_to_dict(state: PipelineState) -> dict:
    # mode='json' converts datetime → ISO string, enum → value (LangGraph-safe)
    return state.model_dump(mode="json")


def _dict_to_state(d: dict) -> PipelineState:
    return PipelineState.model_validate(d)


def make_pipeline(
    use_fixtures: bool = True,
    fixture_dir: str = "./data/fixtures",
    docs_dir: Optional[str] = "./data/docs",
    reddit_subreddits: Optional[list[str]] = None,
    app_store_app_id: Optional[str] = None,
    review_limit: int = 100,
) -> "CompiledGraph":
    """
    Build and compile the PulseAgent LangGraph pipeline.

    Args:
        use_fixtures: Use local fixture data instead of live scraping.
        fixture_dir: Path to fixture JSON files.
        docs_dir: Path to product documentation for RAG ingestion.
        reddit_subreddits: List of subreddits for live scraping.
        app_store_app_id: Apple App Store app ID for live scraping.
        review_limit: Max reviews per run.

    Returns:
        Compiled LangGraph graph ready for async invocation.
    """

    # ── Instantiate agents ──────────────────────────────────────────────────
    scraper = ScraperAgent(
        use_fixtures=use_fixtures,
        fixture_dir=fixture_dir,
        reddit_subreddits=reddit_subreddits or [],
        app_store_app_id=app_store_app_id,
        limit=review_limit,
    )
    classifier = ClassifierAgent()
    scorer = UrgencyScorerAgent()
    rag = RAGAgent(docs_dir=docs_dir)
    responder = ResponseAgent(rag_agent=rag)
    roadmap = RoadmapAgent()
    trend = TrendAgent()

    # ── Define async node functions ─────────────────────────────────────────
    async def scraper_node(state: dict) -> dict:
        result = await scraper.run(_dict_to_state(state))
        return _state_to_dict(result)

    async def classifier_node(state: dict) -> dict:
        result = await classifier.run(_dict_to_state(state))
        return _state_to_dict(result)

    async def scorer_node(state: dict) -> dict:
        result = await scorer.run(_dict_to_state(state))
        return _state_to_dict(result)

    async def response_node(state: dict) -> dict:
        result = await responder.run(_dict_to_state(state))
        return _state_to_dict(result)

    async def roadmap_node(state: dict) -> dict:
        result = await roadmap.run(_dict_to_state(state))
        return _state_to_dict(result)

    async def trend_node(state: dict) -> dict:
        result = await trend.run(_dict_to_state(state))
        return _state_to_dict(result)

    # ── Build graph ─────────────────────────────────────────────────────────
    graph = StateGraph(GraphState)

    graph.add_node("scraper", scraper_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("scorer", scorer_node)
    graph.add_node("responder", response_node)
    graph.add_node("roadmap", roadmap_node)
    graph.add_node("trend", trend_node)

    graph.set_entry_point("scraper")
    graph.add_edge("scraper", "classifier")
    graph.add_edge("classifier", "scorer")
    graph.add_edge("scorer", "responder")
    graph.add_edge("responder", "roadmap")
    graph.add_edge("roadmap", "trend")
    graph.add_edge("trend", END)

    return graph.compile()


async def run_pipeline(
    product_name: str,
    use_fixtures: bool = True,
    fixture_dir: str = "./data/fixtures",
    docs_dir: Optional[str] = "./data/docs",
    **kwargs: Any,
) -> PipelineState:
    """
    High-level helper: build pipeline, run it, return final PipelineState.
    """
    graph = make_pipeline(
        use_fixtures=use_fixtures,
        fixture_dir=fixture_dir,
        docs_dir=docs_dir,
        **kwargs,
    )

    initial = PipelineState(
        product_name=product_name,
        run_id=str(uuid.uuid4())[:8],
        started_at=datetime.now(),
    )

    logger.info(f"[Pipeline] Starting run {initial.run_id} for '{product_name}'")

    result_dict = await graph.ainvoke(_state_to_dict(initial))
    final = _dict_to_state(result_dict)
    final.completed_at = datetime.now()

    elapsed = (final.completed_at - final.started_at).total_seconds()
    logger.success(
        f"[Pipeline] Run {final.run_id} completed in {elapsed:.1f}s. "
        f"Reviews={len(final.raw_reviews)}, "
        f"Roadmap items={len(final.roadmap_items)}, "
        f"Draft responses={len(final.draft_responses)}, "
        f"Trend alerts={len(final.trend_alerts)}, "
        f"Errors={len(final.errors)}"
    )

    return final
