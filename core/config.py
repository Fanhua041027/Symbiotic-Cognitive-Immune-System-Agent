"""Configuration validation and management.

Validates environment variables at startup and provides
typed access to all configuration values.
"""

import os
from typing import Any, Optional

from dotenv import load_dotenv

from core.logger import setup_logger

load_dotenv()

logger = setup_logger("config")

# Validation rules: (key, default, type, required, description)
_CONFIG_DEFS: list[tuple[str, Any, type, bool, str]] = [
    ("OPENAI_API_KEY", None, str, True, "OpenAI API key"),
    ("MAIN_LLM_MODEL", "gpt-4o", str, False, "Worker LLM model"),
    ("MONITOR_LLM_MODEL", "gpt-4o-mini", str, False, "Monitor LLM model"),
    ("ANTIBODY_LLM_MODEL", "gpt-4o", str, False, "Antibody generator LLM model"),
    ("LLM_TEMPERATURE", 0.7, float, False, "LLM temperature"),
    ("MAX_ITERATIONS", 5, int, False, "Max immune iterations"),
    ("SANDBOX_MODE", "simulated", str, False, "Sandbox mode: simulated/ast/docker"),
    ("LOG_LEVEL", "INFO", str, False, "Log level: DEBUG/INFO/WARNING/ERROR"),
    ("ESCALATION_THRESHOLD", 3, int, False, "Escalation failure threshold"),
]

# Cache
_values: dict[str, Any] = {}
_validated = False

VALID_SANDBOX_MODES = {"simulated", "ast", "docker"}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}


def validate_all() -> list[str]:
    """Validate all config values. Returns list of warning messages."""
    warnings: list[str] = []

    for key, default, cast_type, required, desc in _CONFIG_DEFS:
        raw = os.getenv(key)

        if raw is None:
            if required:
                warnings.append(f"MISSING: {key} ({desc}) — set in .env or environment")
            continue

        try:
            if cast_type == bool:
                _values[key] = raw.lower() in ("1", "true", "yes")
            elif cast_type == int:
                _values[key] = int(raw)
            elif cast_type == float:
                _values[key] = float(raw)
            else:
                _values[key] = raw
        except ValueError:
            warnings.append(
                f"INVALID: {key}={raw!r} — expected {cast_type.__name__}, using default"
            )
            _values[key] = default

    # Special validations
    sandbox = _values.get("SANDBOX_MODE", "simulated")
    if sandbox not in VALID_SANDBOX_MODES:
        warnings.append(
            f"INVALID: SANDBOX_MODE={sandbox!r} — must be one of {VALID_SANDBOX_MODES}"
        )
        _values["SANDBOX_MODE"] = "simulated"

    log_level = _values.get("LOG_LEVEL", "INFO")
    if log_level not in VALID_LOG_LEVELS:
        warnings.append(
            f"INVALID: LOG_LEVEL={log_level!r} — must be one of {VALID_LOG_LEVELS}"
        )
        _values["LOG_LEVEL"] = "INFO"

    global _validated
    _validated = True

    return warnings


def get(key: str, default: Any = None) -> Any:
    """Get a typed config value after validation."""
    if not _validated:
        validate_all()
    return _values.get(key, default)


def show_summary() -> None:
    """Log a summary of current configuration."""
    warnings = validate_all()

    logger.info("Configuration loaded:")
    for key, default, cast_type, required, desc in _CONFIG_DEFS:
        if key == "OPENAI_API_KEY":
            val = "****" if _values.get(key) else "NOT SET"
        else:
            val = _values.get(key, default)
        logger.info("  %s = %s", key, val)

    if warnings:
        for w in warnings:
            logger.warning("  ⚠ %s", w)
