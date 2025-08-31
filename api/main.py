"""
PulseAgent — FastAPI REST API
Provides async endpoints to trigger pipeline runs and query results.
Supports WebSocket streaming for real-time progress updates.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from config import get_settings
from graph.pipeline import run_pipeline
from models import PipelineState


# ─── In-memory run store (replace with Redis/DB in production) ───────────────
_runs: dict[str, PipelineState] = {}
_run_status: dict[str, str] = {}   # run_id → "pending" | "running" | "done" | "error"


# ─── Request / Response schemas ─────────────────────────────────────────────

class RunRequest(BaseModel):
    product_name: str
    use_fixtures: bool = True
    fixture_dir: str = "./data/fixtures"
    docs_dir: Optional[str] = "./data/docs"
    review_limit: int = 50
    reddit_subreddits: list[str] = []
    app_store_app_id: Optional[str] = None


class RunStatus(BaseModel):
    run_id: str
    status: str
    product_name: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    review_count: int = 0
    roadmap_count: int = 0
    draft_count: int = 0
    alert_count: int = 0
    error_count: int = 0


# ─── App lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PulseAgent API starting up...")
    yield
    logger.info("PulseAgent API shutting down.")


app = FastAPI(
    title="PulseAgent API",
    description="Multi-agent product review intelligence system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0", "timestamp": datetime.now().isoformat()}


@app.post("/runs", response_model=RunStatus, status_code=202)
async def start_run(req: RunRequest):
    """Kick off an async pipeline run. Returns run_id immediately."""
    run_id = str(uuid.uuid4())[:8]
    _run_status[run_id] = "pending"

    async def _execute():
        _run_status[run_id] = "running"
        try:
            state = await run_pipeline(
                product_name=req.product_name,
                use_fixtures=req.use_fixtures,
                fixture_dir=req.fixture_dir,
                docs_dir=req.docs_dir,
                review_limit=req.review_limit,
                reddit_subreddits=req.reddit_subreddits,
                app_store_app_id=req.app_store_app_id,
            )
            state.run_id = run_id
            _runs[run_id] = state
            _run_status[run_id] = "done"
        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}")
            _run_status[run_id] = "error"

    asyncio.create_task(_execute())

    return RunStatus(
        run_id=run_id,
        status="pending",
        product_name=req.product_name,
        started_at=datetime.now(),
    )


@app.get("/runs/{run_id}", response_model=RunStatus)
async def get_run_status(run_id: str):
    """Poll run status."""
    if run_id not in _run_status:
        raise HTTPException(status_code=404, detail="Run not found")

    status = _run_status[run_id]
    state = _runs.get(run_id)

    return RunStatus(
        run_id=run_id,
        status=status,
        product_name=state.product_name if state else None,
        started_at=state.started_at if state else None,
        completed_at=state.completed_at if state else None,
        review_count=len(state.raw_reviews) if state else 0,
        roadmap_count=len(state.roadmap_items) if state else 0,
        draft_count=len(state.draft_responses) if state else 0,
        alert_count=len(state.trend_alerts) if state else 0,
        error_count=len(state.errors) if state else 0,
    )


@app.get("/runs/{run_id}/roadmap")
async def get_roadmap(run_id: str):
    """Get the generated roadmap items for a completed run."""
    state = _get_completed_run(run_id)
    return {
        "run_id": run_id,
        "product": state.product_name,
        "items": [item.model_dump() for item in state.roadmap_items],
    }


@app.get("/runs/{run_id}/responses")
async def get_responses(run_id: str):
    """Get draft responses for high-urgency reviews."""
    state = _get_completed_run(run_id)
    # Enrich with original review text
    id_map = {r.review.id: r for r in state.scored_reviews}
    enriched = []
    for draft in state.draft_responses:
        scored = id_map.get(draft.review_id)
        enriched.append({
            **draft.model_dump(),
            "original_review": scored.review.text[:300] if scored else None,
            "category": scored.category.value if scored else None,
            "urgency_score": scored.urgency_score if scored else None,
        })
    return {"run_id": run_id, "drafts": enriched}


@app.get("/runs/{run_id}/clusters")
async def get_clusters(run_id: str):
    """Get review clusters with urgency scores."""
    state = _get_completed_run(run_id)
    return {
        "run_id": run_id,
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "category": c.category.value,
                "theme": c.theme,
                "total_count": c.total_count,
                "avg_urgency": c.avg_urgency,
                "churn_risk_count": c.churn_risk_count,
                "top_phrases": c.top_phrases,
            }
            for c in state.clusters
        ],
    }


@app.get("/runs/{run_id}/trends")
async def get_trends(run_id: str):
    """Get trend alerts."""
    state = _get_completed_run(run_id)
    return {
        "run_id": run_id,
        "alerts": [alert.model_dump() for alert in state.trend_alerts],
    }


@app.get("/runs/{run_id}/summary")
async def get_summary(run_id: str):
    """Get full pipeline summary."""
    state = _get_completed_run(run_id)
    from collections import Counter

    cat_dist = Counter(r.category.value for r in state.classified_reviews)
    sentiment_dist = Counter(r.sentiment.value for r in state.classified_reviews)
    churn_count = sum(1 for r in state.classified_reviews if r.is_churn_signal)

    return {
        "run_id": run_id,
        "product": state.product_name,
        "started_at": state.started_at,
        "completed_at": state.completed_at,
        "total_reviews": len(state.raw_reviews),
        "category_distribution": dict(cat_dist),
        "sentiment_distribution": dict(sentiment_dist),
        "churn_signals": churn_count,
        "clusters_formed": len(state.clusters),
        "roadmap_items": len(state.roadmap_items),
        "draft_responses": len(state.draft_responses),
        "trend_alerts": len(state.trend_alerts),
        "errors": state.errors,
    }


# ─── WebSocket for live progress ─────────────────────────────────────────────

@app.websocket("/ws/runs/{run_id}")
async def websocket_run_progress(websocket: WebSocket, run_id: str):
    """Stream run status updates via WebSocket."""
    await websocket.accept()
    try:
        while True:
            status = _run_status.get(run_id, "not_found")
            state = _runs.get(run_id)
            await websocket.send_json({
                "run_id": run_id,
                "status": status,
                "review_count": len(state.raw_reviews) if state else 0,
                "roadmap_count": len(state.roadmap_items) if state else 0,
            })
            if status in ("done", "error", "not_found"):
                break
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        pass


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_completed_run(run_id: str) -> PipelineState:
    if run_id not in _run_status:
        raise HTTPException(status_code=404, detail="Run not found")
    status = _run_status[run_id]
    if status != "done":
        raise HTTPException(
            status_code=202,
            detail=f"Run is not complete yet. Current status: {status}",
        )
    return _runs[run_id]
