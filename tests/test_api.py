"""
Tests for PulseAgent FastAPI endpoints.
Uses httpx TestClient — mocks the pipeline to avoid LLM calls.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.main import app, _run_status, _runs
from models import PipelineState


@pytest.fixture(autouse=True)
def clear_run_store():
    """Reset in-memory run store between tests to avoid state bleed."""
    _run_status.clear()
    _runs.clear()
    yield
    _run_status.clear()
    _runs.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def completed_run_state():
    return PipelineState(
        product_name="TestProduct",
        run_id="done-run-001",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        raw_reviews=[],
        classified_reviews=[],
        scored_reviews=[],
        clusters=[],
        roadmap_items=[],
        draft_responses=[],
        trend_alerts=[],
        errors=[],
    )


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data


class TestRunEndpoints:
    def test_start_run_returns_202(self, client):
        with patch("api.main.run_pipeline", new_callable=AsyncMock) as mock:
            mock.return_value = PipelineState(
                product_name="TestProduct",
                run_id="mocked-run",
                started_at=datetime.now(),
                completed_at=datetime.now(),
            )
            response = client.post("/runs", json={
                "product_name": "TestProduct",
                "use_fixtures": True,
                "review_limit": 10,
            })
        assert response.status_code == 202
        data = response.json()
        assert "run_id" in data
        assert data["status"] in ("pending", "running", "done")

    def test_get_nonexistent_run_returns_404(self, client):
        response = client.get("/runs/nonexistent-run-id")
        assert response.status_code == 404

    def test_roadmap_for_incomplete_run_returns_202_or_404(self, client):
        """
        A run that was just created but hasn't finished yet
        should return 202 (pending/running) or 404 if not yet registered.
        We inject the run directly into the store as 'pending' to test this path.
        """
        run_id = "pending-test-run"
        _run_status[run_id] = "pending"
        # Do NOT add to _runs — state not ready yet

        roadmap_response = client.get(f"/runs/{run_id}/roadmap")
        assert roadmap_response.status_code == 202

    def test_roadmap_returns_data_for_completed_run(self, client, completed_run_state):
        """A completed run should return roadmap data (empty list is fine)."""
        run_id = "done-run-001"
        _run_status[run_id] = "done"
        _runs[run_id] = completed_run_state

        response = client.get(f"/runs/{run_id}/roadmap")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_summary_for_completed_run(self, client, completed_run_state):
        """Summary endpoint should work for a completed run."""
        run_id = "done-run-001"
        _run_status[run_id] = "done"
        _runs[run_id] = completed_run_state

        response = client.get(f"/runs/{run_id}/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["product"] == "TestProduct"
        assert "total_reviews" in data


class TestAPIValidation:
    def test_run_request_requires_product_name(self, client):
        response = client.post("/runs", json={})
        assert response.status_code == 422  # Unprocessable Entity

    def test_run_request_default_values(self, client):
        with patch("api.main.run_pipeline", new_callable=AsyncMock) as mock:
            mock.return_value = PipelineState(
                product_name="TestProd",
                run_id="x",
                started_at=datetime.now(),
            )
            response = client.post("/runs", json={"product_name": "TestProd"})
        assert response.status_code == 202
        data = response.json()
        assert data["product_name"] == "TestProd"