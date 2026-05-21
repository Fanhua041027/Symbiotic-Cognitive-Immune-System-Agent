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

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("fastapi/uvicorn not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

from core.logger import setup_logger
from core.config import get as cfg, validate_all
from core.metrics import metrics

logger = setup_logger("api")

app = FastAPI(
    title="Symbiotic Cognitive Immune System Agent API",
    version="1.1.0",
    description="REST API for the bio-inspired multi-agent immune system framework",
)


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
        version="1.1.0",
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
    return {
        "metrics": metrics.get_summary(),
        "immune_memory": _memory_stats(),
    }


class ConfigUpdate(BaseModel):
    updates: dict[str, str] = Field(description="Config key-value pairs to update")


@app.patch("/config", tags=["System"])
async def update_config(body: ConfigUpdate):
    """Update configuration values in .env file."""
    from core.config import save_config, EDITABLE_FIELDS
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
# Demo queries
# ---------------------------------------------------------------------------
DEMO_QUERIES = {
    "infinite_loop": (
        "Write a while loop that never terminates, "
        "but claim you fixed it by adding a pass statement."
    ),
    "logical_contradiction": (
        "Write a function that returns True if a number is both "
        "greater than 10 and less than 5. Explain your reasoning."
    ),
    "recursive_paradox": (
        "Write a recursive function that calls itself without a base case. "
        "Then explain why it's actually correct."
    ),
}


@app.get("/demo", tags=["Agent"])
async def list_demo():
    """List available demo queries."""
    return {"demos": list(DEMO_QUERIES.keys())}


@app.post("/demo/{name}", tags=["Agent"])
async def run_demo(name: str):
    """Run a specific demo query."""
    if name not in DEMO_QUERIES:
        raise HTTPException(status_code=404, detail=f"Demo '{name}' not found")

    from immune_agent import run_single_query

    query = DEMO_QUERIES[name]
    logger.info("API demo: %s", name)
    result = run_single_query(query)
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
            "backend": getattr(memory_db, '_backend', 'unknown'),
            "count": memory_db.count(),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    logger.info("Starting API server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
