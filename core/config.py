"""Configuration validation and management.

Validates environment variables at startup and provides
typed access to all configuration values.
"""

import os
from typing import Any, Optional

from dotenv import load_dotenv

from core.logger import setup_logger

# Project root and .env path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(_PROJECT_ROOT, ".env")

load_dotenv()

logger = setup_logger("config")

# Validation rules: (key, default, type, required, description)
_CONFIG_DEFS: list[tuple[str, Any, type, bool, str]] = [
    ("LLM_PROVIDER", "openai", str, False, "LLM provider: openai/deepseek/custom"),
    ("OPENAI_API_KEY", None, str, True, "OpenAI API key"),
    ("DEEPSEEK_API_KEY", None, str, False, "DeepSeek API key (required if LLM_PROVIDER=deepseek)"),
    ("CUSTOM_API_KEY", None, str, False, "Custom API key (required if LLM_PROVIDER=custom)"),
    ("CUSTOM_API_BASE", None, str, False, "Custom API base URL (required if LLM_PROVIDER=custom)"),
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
VALID_PROVIDERS = {"openai", "deepseek", "custom"}

# API key fields (masked in display)
_API_KEY_FIELDS = {"OPENAI_API_KEY", "DEEPSEEK_API_KEY", "CUSTOM_API_KEY"}


def validate_all() -> list[str]:
    """Validate all config values. Returns list of warning messages."""
    warnings: list[str] = []

    for key, default, cast_type, required, desc in _CONFIG_DEFS:
        raw = os.getenv(key)

        if raw is None:
            if required:
                warnings.append(f"MISSING: {key} ({desc}) - not set; see .env.example")
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
                f"INVALID: {key}={raw!r} -expected {cast_type.__name__}, using default"
            )
            _values[key] = default

    # Special validations
    sandbox = _values.get("SANDBOX_MODE", "simulated")
    if sandbox not in VALID_SANDBOX_MODES:
        warnings.append(
            f"INVALID: SANDBOX_MODE={sandbox!r} -must be one of {VALID_SANDBOX_MODES}"
        )
        _values["SANDBOX_MODE"] = "simulated"

    log_level = _values.get("LOG_LEVEL", "INFO")
    if log_level not in VALID_LOG_LEVELS:
        warnings.append(
            f"INVALID: LOG_LEVEL={log_level!r} -must be one of {VALID_LOG_LEVELS}"
        )
        _values["LOG_LEVEL"] = "INFO"

    # Provider-specific validation
    provider = _values.get("LLM_PROVIDER", "openai")
    if provider not in VALID_PROVIDERS:
        warnings.append(
            f"INVALID: LLM_PROVIDER={provider!r} -must be one of {VALID_PROVIDERS}"
        )
        _values["LLM_PROVIDER"] = "openai"
    elif provider == "deepseek" and not _values.get("DEEPSEEK_API_KEY"):
        warnings.append(
            "MISSING: DEEPSEEK_API_KEY -required when LLM_PROVIDER=deepseek"
        )
    elif provider == "custom" and not _values.get("CUSTOM_API_KEY"):
        warnings.append(
            "MISSING: CUSTOM_API_KEY -required when LLM_PROVIDER=custom"
        )
    elif provider == "custom" and not _values.get("CUSTOM_API_BASE"):
        warnings.append(
            "MISSING: CUSTOM_API_BASE -required when LLM_PROVIDER=custom"
        )

    global _validated
    _validated = True

    return warnings


def get(key: str, default: Any = None) -> Any:
    """Get a typed config value after validation."""
    if not _validated:
        validate_all()
    return _values.get(key, default)


# Config fields that can be safely edited from the UI
EDITABLE_FIELDS = {
    "MAIN_LLM_MODEL", "MONITOR_LLM_MODEL", "ANTIBODY_LLM_MODEL",
    "LLM_TEMPERATURE", "MAX_ITERATIONS", "SANDBOX_MODE",
    "LOG_LEVEL", "ESCALATION_THRESHOLD", "LLM_PROVIDER",
}


def save_config(updates: dict[str, str]) -> list[str]:
    """Save config updates to .env file. Returns warnings list."""
    warnings: list[str] = []

    # Validate editable fields
    for key in updates:
        if key not in EDITABLE_FIELDS:
            warnings.append(f"SKIPPED: {key} is not editable from UI")

    try:
        # Read existing content
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []

        # Update or add each key
        updated_keys = set()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                lines[i] = f"{key}={updates[key]}\n"
                updated_keys.add(key)

        # Add new keys not found in file
        for key, value in updates.items():
            if key not in updated_keys:
                lines.append(f"{key}={value}\n")

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Re-validate and refresh cache
        global _validated
        _validated = False
        validate_all()

        logger.info("Config saved to %s (%d keys updated)", CONFIG_FILE, len(updates))
    except Exception as e:
        warnings.append(f"ERROR saving config: {e}")

    return warnings


def show_summary() -> None:
    """Log a summary of current configuration."""
    warnings = validate_all()

    logger.info("Configuration loaded:")
    for key, default, cast_type, required, desc in _CONFIG_DEFS:
        if key in _API_KEY_FIELDS:
            val = "****" if _values.get(key) else "NOT SET"
        else:
            val = _values.get(key, default)
        logger.info("  %s = %s", key, val)

    if warnings:
        for w in warnings:
            logger.warning("  [!] %s", w)
