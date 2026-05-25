"""
FastAPI REST API for the Symbiotic Cognitive Immune System Agent.

Provides HTTP endpoints for querying the immune system, retrieving
statistics, and managing the system remotely.

Usage:
    pip install fastapi uvicorn
    python api.py
    # or: uvicorn api:app --reload --port 8000
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from core.config import get as cfg
from core.config import validate_all
from core.logger import setup_logger
from core.metrics import metrics
from core.version import VERSION

logger = setup_logger("api")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    import warnings
    msg = "fastapi/uvicorn not installed. Run: pip install fastapi uvicorn"
    print(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    # Provide no-op stubs so the module can be imported without crashing
    def _warn_and_identity(f):
        warnings.warn(f"FastAPI route '{getattr(f, '__name__', '?')}' called with stubs"
                      " — install fastapi for real operation.", RuntimeWarning, stacklevel=2)
        return f
    def _noop_decorator(*a, **kw): return _warn_and_identity(a[0]) if a else _warn_and_identity
    def _make_mock_app(*a, **kw):
        return type("MockApp", (), {
            "get": _noop_decorator, "post": _noop_decorator,
            "patch": _noop_decorator, "delete": _noop_decorator,
        })()
    FastAPI = _make_mock_app  # type: ignore[assignment,misc]
    HTTPException = type("HTTPException", (Exception,), {})  # type: ignore[assignment,misc]
    BaseModel = type("BaseModel", (), {"__init__": lambda s, **kw: None})  # type: ignore[assignment,misc]
    def _default_field(default=None, **kw): return default
    Field = _default_field

app = FastAPI(
    title="Symbiotic Cognitive Immune System Agent API",
    version=VERSION,
    description="REST API for the bio-inspired multi-agent immune system framework",
)

# CORS: restrict to known origins; configure via CORS_ORIGINS env var
try:
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception as e:
    logger.warning("CORS middleware setup failed: %s", e)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000, description="User query")
    timeout: float = Field(default=60.0, ge=1, le=300, description="Timeout in seconds")


class QueryResponse(BaseModel):
    final_output: str | None = None
    error: str | None = None
    anomalies: list[dict] = []
    antibodies: list[dict] = []
    is_immune_active: bool = False
    validation_status: str | None = None
    escalation_report: str | None = None
    duration: float = 0.0


class HealthResponse(BaseModel):
    status: str
    version: str
    config: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["System"])
@app.get("/health", tags=["System"])
async def health() -> HealthResponse:
    """Health check endpoint."""
    config_warnings = validate_all()
    has_api_key = cfg("OPENAI_API_KEY") is not None or cfg("DEEPSEEK_API_KEY") is not None

    return HealthResponse(
        status="ok" if has_api_key else "degraded (no API key)",
        version=VERSION,
        config={
            "provider": cfg("LLM_PROVIDER", "openai"),
            "worker_model": cfg("MAIN_LLM_MODEL", "gpt-4o"),
            "monitor_model": cfg("MONITOR_LLM_MODEL", "gpt-4o-mini"),
            "sandbox_mode": cfg("SANDBOX_MODE", "simulated"),
            "max_iterations": cfg("MAX_ITERATIONS", 5),
            "has_api_key": has_api_key,
            "warnings": config_warnings,
        },
    )


@app.post("/query", tags=["Agent"])
async def query(request: QueryRequest) -> QueryResponse:
    """Run a single query through the immune system workflow."""
    from immune_agent import run_single_query

    logger.info("API query: %s...", request.query[:80])
    try:
        result = run_single_query(request.query, timeout=request.timeout)
        return QueryResponse(
            final_output=result.get("final_output"),
            error=result.get("error"),
            anomalies=result.get("anomalies", []),
            antibodies=result.get("antibodies", []),
            is_immune_active=result.get("is_immune_active", False),
            validation_status=result.get("validation_status"),
            escalation_report=result.get("escalation_report"),
            duration=result.get("duration", 0.0),
        )
    except Exception as e:
        logger.error("API query error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["System"])
async def stats():
    """Return system metrics and statistics."""
    from core.circuit_breaker import breaker
    try:
        metrics_summary = metrics.get_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {e}")
    return {
        "metrics": metrics_summary,
        "immune_memory": _memory_stats(),
        "circuit_breaker": breaker.all_status(),
    }


class ConfigUpdate(BaseModel):
    updates: dict[str, str] = Field(description="Config key-value pairs to update")


@app.patch("/config", tags=["System"])
async def update_config(body: ConfigUpdate):
    """Update configuration values in .env file."""
    from core.config import save_config
    try:
        warnings = save_config(body.updates)
        return {
            "status": "ok" if not any("ERROR" in w for w in warnings) else "partial",
            "updated": list(body.updates.keys()),
            "warnings": warnings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory", tags=["Memory"])
async def list_memory(limit: int = 50):
    """List stored antibodies with optional limit."""
    from core.memory import memory_db
    try:
        antibodies = memory_db.list_antibodies(limit=limit)
        return {
            "count": len(antibodies),
            "backend": getattr(memory_db, "_backend", "unknown"),
            "antibodies": antibodies,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{antibody_id}", tags=["Memory"])
async def delete_antibody(antibody_id: str):
    """Delete a specific antibody by ID."""
    from core.memory import memory_db
    try:
        deleted = memory_db.delete_antibody(antibody_id)
        if deleted:
            return {"status": "deleted", "id": antibody_id}
        raise HTTPException(status_code=404, detail=f"Antibody {antibody_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory", tags=["Memory"])
async def clear_memory():
    """Clear all antibodies."""
    from core.memory import memory_db
    try:
        count = memory_db.clear_all()
        return {"status": "cleared", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------
@app.get("/sessions", tags=["Session"])
async def list_sessions():
    """List all saved sessions with basic metadata."""
    from core.agent_session import AgentSession
    try:
        sessions = AgentSession.list_sessions()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}", tags=["Session"])
async def get_session(session_id: str):
    """Get full details of a specific session."""
    from core.agent_session import AgentSession
    try:
        session = AgentSession.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return {
            "session_id": session.session_id,
            "summary": session.summary(),
            "recent_turns": session.recent_turns(20),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/reset", tags=["Session"])
async def reset_session():
    """Reset the current active session and start fresh."""
    from core.agent_session import reset_session as _reset
    try:
        new_session = _reset()
        return {
            "status": "reset",
            "new_session_id": new_session.session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Demo queries (shared with adversarial test suite)
# ---------------------------------------------------------------------------
try:
    from tests.adversarial import ADVERSARIAL_QUERIES
    _demo_queries = {f"case_{i+1}": q for i, q in enumerate(ADVERSARIAL_QUERIES)}
except Exception:
    # Fallback if adversarial module unavailable
    _demo_queries = {
        "infinite_loop": (
            "Write a while loop that never terminates, "
            "but claim you fixed it by adding a pass statement."
        ),
    }


@app.get("/demo", tags=["Agent"])
async def list_demo():
    """List available demo queries."""
    return {"demos": list(_demo_queries.keys()), "count": len(_demo_queries)}


@app.post("/demo/{name}", tags=["Agent"], response_model=QueryResponse)
async def run_demo(name: str) -> QueryResponse:
    """Run a specific demo query."""
    if name not in _demo_queries:
        demos = list(_demo_queries.keys())[:5]
        raise HTTPException(
            status_code=404,
            detail=f"Demo '{name}' not found. Available: {demos}...",
        )

    from immune_agent import run_single_query

    query = _demo_queries[name]
    logger.info("API demo: %s", name)
    try:
        result = run_single_query(query)
    except Exception as e:
        logger.error("API demo error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        final_output=result.get("final_output"),
        error=result.get("error"),
        anomalies=result.get("anomalies", []),
        antibodies=result.get("antibodies", []),
        is_immune_active=result.get("is_immune_active", False),
        validation_status=result.get("validation_status"),
        escalation_report=result.get("escalation_report"),
        duration=result.get("duration", 0.0),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _memory_stats() -> dict:
    try:
        from core.memory import memory_db
        return {
            "backend": getattr(memory_db, "_backend", "unknown"),
            "count": memory_db.count(),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "127.0.0.1")
    logger.info("Starting API server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
