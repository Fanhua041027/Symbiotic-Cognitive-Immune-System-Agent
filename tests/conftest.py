"""pytest configuration and shared fixtures."""
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


@pytest.fixture
def immune_state() -> dict[str, Any]:
    """Create a minimal ImmunologyState fixture."""
    return {
        "user_query": "test",
        "task_steps": [],
        "anomalies": [],
        "antibodies": [],
        "final_output": None,
        "is_immune_active": False,
        "validation_status": None,
        "iteration_count": 0,
        "escalation_report": None,
        "request_id": None,
        "workflow_trace": [],
    }


@pytest.fixture
def mock_cfg(monkeypatch):
    """Fixture to mock core.config.get with a controllable dict."""
    import core.config as cfg_mod
    _values: dict[str, Any] = {}

    def _mock_get(key: str, default: Any = None) -> Any:
        return _values.get(key, default)

    monkeypatch.setattr(cfg_mod, "get", _mock_get)
    return _values
