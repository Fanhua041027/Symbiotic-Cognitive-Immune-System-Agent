"""Integration tests for the FastAPI REST API endpoints."""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# Skip all tests if FastAPI not installed
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_metrics():
    """Mock core.metrics.metrics.get_summary."""
    with patch("core.metrics.metrics.get_summary") as mock:
        mock.return_value = {
            "status": "ok",
            "records": 5,
            "success_rate": 80.0,
            "anomaly_rate": 20.0,
            "immune_activation_rate": 15.0,
            "escalation_rate": 0.0,
            "avg_antibodies_per_query": 0.6,
            "anomaly_breakdown": {"monitor": 1},
            "latency": {"avg_seconds": 2.5, "p95_seconds": 5.0, "max_seconds": 8.0},
            "session_duration_seconds": 120.0,
            "total_llm_time_seconds": 12.5,
        }
        yield mock


@pytest.fixture
def mock_memory():
    """Mock core.memory.memory_db methods."""
    with patch("core.memory.memory_db") as mock:
        mock.count.return_value = 3
        mock._backend = "memory"
        mock.list_antibodies.return_value = [
            {"id": "0", "error_pattern": "loop risk", "code": "fix", "context": "ctx"},
        ]
        mock.delete_antibody.return_value = True
        mock.clear_all.return_value = 3
        yield mock


@pytest.fixture
def client(mock_metrics, mock_memory):
    """Create a TestClient with mocked dependencies."""
    # Import app after patching so it uses the mocked dependencies
    from api import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["version"] == "1.1.0"
        assert "config" in data

    def test_root_redirects_to_health(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------
class TestQuery:
    def test_query_returns_result(self, client):
        """POST /query returns the result from run_single_query."""
        fake_result = {
            "final_output": "Hello world",
            "anomalies": [],
            "antibodies": [{"code": "fix", "explanation": "fixed"}],
            "is_immune_active": True,
            "validation_status": "passed",
            "error": None,
            "escalation_report": None,
            "duration": 1.5,
        }
        with patch("immune_agent.run_single_query", return_value=fake_result):
            resp = client.post("/query", json={"query": "test query"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["final_output"] == "Hello world"
        assert data["is_immune_active"] is True
        assert len(data["antibodies"]) == 1

    def test_query_rejects_empty(self, client):
        """POST /query with empty query returns 422."""
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    def test_query_handles_timeout(self, client):
        """POST /query forwards timeout parameter."""
        fake_result = {"final_output": "ok", "anomalies": [], "antibodies": [],
                       "is_immune_active": False, "validation_status": None,
                       "error": None, "escalation_report": None, "duration": 0.5}
        with patch("immune_agent.run_single_query", return_value=fake_result) as mock:
            resp = client.post("/query", json={"query": "hello", "timeout": 30})
        assert resp.status_code == 200
        mock.assert_called_once_with("hello", timeout=30.0)

    def test_query_reports_error(self, client):
        """POST /query surfaces run_single_query errors."""
        with patch(
            "immune_agent.run_single_query",
            side_effect=RuntimeError("API failure"),
        ):
            resp = client.post("/query", json={"query": "hello"})
        assert resp.status_code == 500
        assert "API failure" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------
class TestStats:
    def test_stats_returns_metrics_and_memory(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        assert data["metrics"]["records"] == 5
        assert "immune_memory" in data


# ---------------------------------------------------------------------------
# Config endpoint
# ---------------------------------------------------------------------------
class TestConfig:
    def test_patch_config_returns_updated_keys(self, client):
        with patch("core.config.save_config") as mock_save:
            mock_save.return_value = []
            resp = client.patch("/config", json={"updates": {"MAX_ITERATIONS": "10"}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "MAX_ITERATIONS" in data["updated"]

    def test_patch_config_returns_warnings(self, client):
        with patch("core.config.save_config") as mock_save:
            mock_save.return_value = ["ERROR saving config: permission denied"]
            resp = client.patch("/config", json={"updates": {"SOME_KEY": "val"}})
        assert resp.status_code == 200
        assert resp.json()["status"] == "partial"

    def test_patch_config_empty_body(self, client):
        resp = client.patch("/config", json={"updates": {}})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Memory endpoints
# ---------------------------------------------------------------------------
class TestMemory:
    def test_list_memory_empty(self, client, mock_memory):
        mock_memory.list_antibodies.return_value = []
        resp = client.get("/memory")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_list_memory_with_data(self, client):
        resp = client.get("/memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert len(data["antibodies"]) == 1

    def test_list_memory_limit(self, client, mock_memory):
        client.get("/memory?limit=10")
        mock_memory.list_antibodies.assert_called_once_with(limit=10)

    def test_delete_antibody_found(self, client, mock_memory):
        mock_memory.delete_antibody.return_value = True
        resp = client.delete("/memory/0")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_antibody_not_found(self, client, mock_memory):
        mock_memory.delete_antibody.return_value = False
        resp = client.delete("/memory/999")
        assert resp.status_code == 404

    def test_clear_memory(self, client, mock_memory):
        resp = client.delete("/memory")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"


# ---------------------------------------------------------------------------
# Demo endpoints
# ---------------------------------------------------------------------------
class TestDemo:
    def test_list_demos(self, client):
        resp = client.get("/demo")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert "demos" in data

    def test_run_demo_found(self, client):
        fake_result = {
            "final_output": "demo result", "anomalies": [], "antibodies": [],
            "is_immune_active": False, "validation_status": None,
            "error": None, "escalation_report": None, "duration": 0.5,
        }
        with patch("immune_agent.run_single_query", return_value=fake_result):
            resp = client.post("/demo/case_1")
        assert resp.status_code == 200, resp.json()
        assert resp.json()["final_output"] == "demo result"

    def test_run_demo_not_found(self, client):
        resp = client.post("/demo/nonexistent_demo")
        assert resp.status_code == 404
