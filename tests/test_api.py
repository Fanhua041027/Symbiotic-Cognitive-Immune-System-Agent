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

    def test_cors_headers_present(self, client):
        """CORS middleware adds Access-Control-Allow-Origin header."""
        resp = client.get("/health", headers={"Origin": "http://localhost:8501"})
        assert resp.headers.get("access-control-allow-origin") == "*"


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
        assert "circuit_breaker" in data

    def test_stats_contains_circuit_breaker_status(self, client):
        resp = client.get("/stats")
        data = resp.json()
        cb = data["circuit_breaker"]
        assert isinstance(cb, dict)


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


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------
class TestSessions:
    def test_list_sessions(self, client):
        """GET /sessions returns session list."""
        resp = client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "count" in data

    def test_get_session_not_found(self, client):
        """GET /sessions/{id} returns 404 for unknown session."""
        resp = client.get("/sessions/nonexistent")
        assert resp.status_code == 404

    def test_get_session_found(self, client):
        """GET /sessions/{id} returns session details."""
        from unittest.mock import patch

        from core.agent_session import AgentSession
        fake_session = AgentSession(session_id="test-001")
        fake_session.record_turn({
            "final_output": "ok", "user_query": "hello", "anomalies": [],
            "antibodies": [], "is_immune_active": False,
            "validation_status": None, "escalation_report": None,
            "duration": 0.5, "error": None,
        })
        with patch("core.agent_session.AgentSession.load", return_value=fake_session):
            resp = client.get("/sessions/test-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "test-001"
        assert "summary" in data
        assert "recent_turns" in data

    def test_reset_session(self, client):
        """POST /sessions/reset creates a new session."""
        from unittest.mock import patch

        from core.agent_session import AgentSession
        new_sess = AgentSession(session_id="new-sess")
        with patch("core.agent_session.reset_session", return_value=new_sess):
            resp = client.post("/sessions/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reset"
        assert data["new_session_id"] == "new-sess"


# ---------------------------------------------------------------------------
# Extended query validation
# ---------------------------------------------------------------------------
class TestQueryExtended:
    """Boundary and default-value tests for the query endpoint."""

    def test_query_default_timeout(self, client):
        """POST /query without timeout uses default 60.0."""
        fake_result = {
            "final_output": "ok", "anomalies": [], "antibodies": [],
            "is_immune_active": False, "validation_status": None,
            "error": None, "escalation_report": None, "duration": 0.5,
        }
        with patch("immune_agent.run_single_query", return_value=fake_result) as mock:
            resp = client.post("/query", json={"query": "hello"})
        assert resp.status_code == 200
        mock.assert_called_once_with("hello", timeout=60.0)

    def test_query_timeout_too_high(self, client):
        """POST /query with timeout > 300 returns 422."""
        resp = client.post("/query", json={"query": "hello", "timeout": 301})
        assert resp.status_code == 422

    def test_query_timeout_too_low(self, client):
        """POST /query with timeout < 1 returns 422."""
        resp = client.post("/query", json={"query": "hello", "timeout": 0})
        assert resp.status_code == 422

    def test_query_missing_query_field(self, client):
        """POST /query without query field returns 422."""
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_query_at_max_length(self, client):
        """POST /query with 5000 char query works."""
        fake_result = {
            "final_output": "ok", "anomalies": [], "antibodies": [],
            "is_immune_active": False, "validation_status": None,
            "error": None, "escalation_report": None, "duration": 0.5,
        }
        long_query = "a" * 5000
        with patch("immune_agent.run_single_query", return_value=fake_result) as mock:
            resp = client.post("/query", json={"query": long_query})
        assert resp.status_code == 200
        mock.assert_called_once_with(long_query, timeout=60.0)

    def test_query_exceeds_max_length(self, client):
        """POST /query with > 5000 char query returns 422."""
        resp = client.post("/query", json={"query": "a" * 5001})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Error handling: stats
# ---------------------------------------------------------------------------
class TestStatsErrors:
    """Error propagation from stats endpoint."""

    def test_stats_on_metrics_error(self, client, mock_metrics):
        """GET /stats returns 500 when metrics.get_summary fails."""
        mock_metrics.side_effect = RuntimeError("metrics crash")
        resp = client.get("/stats")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Error handling: config
# ---------------------------------------------------------------------------
class TestConfigErrors:
    """Error propagation from config endpoint."""

    def test_patch_config_server_error(self, client):
        """PATCH /config returns 500 when save_config raises."""
        with patch("core.config.save_config", side_effect=RuntimeError("write failed")):
            resp = client.patch("/config", json={"updates": {"MAX_ITERATIONS": "10"}})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Error handling: memory
# ---------------------------------------------------------------------------
class TestMemoryErrors:
    """Error propagation from memory endpoints."""

    def test_list_memory_error(self, client, mock_memory):
        """GET /memory returns 500 when list_antibodies fails."""
        mock_memory.list_antibodies.side_effect = RuntimeError("DB error")
        resp = client.get("/memory")
        assert resp.status_code == 500

    def test_delete_antibody_error(self, client, mock_memory):
        """DELETE /memory/{id} returns 500 when delete_antibody fails."""
        mock_memory.delete_antibody.side_effect = RuntimeError("DB error")
        resp = client.delete("/memory/0")
        assert resp.status_code == 500

    def test_clear_memory_error(self, client, mock_memory):
        """DELETE /memory returns 500 when clear_all fails."""
        mock_memory.clear_all.side_effect = RuntimeError("DB error")
        resp = client.delete("/memory")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Error handling: sessions
# ---------------------------------------------------------------------------
class TestSessionErrors:
    """Error propagation from session endpoints."""

    def test_list_sessions_error(self, client):
        """GET /sessions returns 500 when list_sessions fails."""
        with patch(
            "core.agent_session.AgentSession.list_sessions",
            side_effect=RuntimeError("fail"),
        ):
            resp = client.get("/sessions")
        assert resp.status_code == 500

    def test_get_session_error(self, client):
        """GET /sessions/{id} returns 500 when load fails."""
        with patch(
            "core.agent_session.AgentSession.load",
            side_effect=RuntimeError("fail"),
        ):
            resp = client.get("/sessions/test-001")
        assert resp.status_code == 500

    def test_reset_session_error(self, client):
        """POST /sessions/reset returns 500 when reset fails."""
        with patch(
            "core.agent_session.reset_session",
            side_effect=RuntimeError("fail"),
        ):
            resp = client.post("/sessions/reset")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Error handling: demo
# ---------------------------------------------------------------------------
class TestDemoErrors:
    """Error propagation from demo endpoint."""

    def test_run_demo_error(self, client):
        """POST /demo/{name} returns 500 when run_single_query raises."""
        with patch(
            "immune_agent.run_single_query",
            side_effect=RuntimeError("demo crash"),
        ):
            resp = client.post("/demo/case_1")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# CORS preflight
# ---------------------------------------------------------------------------
class TestCORSExtended:
    """CORS preflight request handling."""

    def test_cors_preflight(self, client):
        """OPTIONS request returns CORS headers."""
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "*"
        assert resp.headers.get("access-control-allow-methods") is not None


# ---------------------------------------------------------------------------
# Health degraded
# ---------------------------------------------------------------------------
class TestHealthDegraded:
    """Health endpoint behavior without API keys."""

    def test_health_degraded_without_key(self, client, monkeypatch):
        """GET /health returns degraded status when no API key is set."""
        import core.config as cfg_mod
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
        cfg_mod._validated = False
        cfg_mod._values.clear()
        resp = client.get("/health")
        data = resp.json()
        assert "degraded" in data["status"].lower()
        assert data["config"]["has_api_key"] is False
