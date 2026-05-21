"""Unit tests for core immune system components."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.escalation import EscalationTracker
from core.sandbox import (
    ASTValidator,
    validate_simulated,
    validate_ast,
)
from core.state import ImmunologyState


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class TestState:
    def test_state_has_required_fields(self):
        """Verify ImmunologyState has all expected fields."""
        state: ImmunologyState = {
            "user_query": "test",
            "task_steps": [],
            "anomalies": [],
            "antibodies": [],
            "final_output": None,
            "is_immune_active": False,
            "validation_status": None,
            "iteration_count": 0,
            "escalation_report": None,
        }
        assert state["user_query"] == "test"
        assert state["iteration_count"] == 0
        assert state["escalation_report"] is None

    def test_state_iteration_count_increments(self):
        """State correctly handles iteration count updates."""
        state: ImmunologyState = {
            "user_query": "test", "task_steps": [], "anomalies": [],
            "antibodies": [], "final_output": None,
            "is_immune_active": False, "validation_status": None,
            "iteration_count": 0, "escalation_report": None,
        }
        assert state["iteration_count"] == 0
        state["iteration_count"] = 1
        assert state["iteration_count"] == 1


# ---------------------------------------------------------------------------
# Sandbox validators
# ---------------------------------------------------------------------------
class TestSandboxSimulated:
    def test_valid_code_passes(self):
        assert validate_simulated("def fix_loop():\n    max_iter = 100\n    return")

    def test_short_code_fails(self):
        assert not validate_simulated("x = 1")

    def test_code_with_fix_keyword_passes(self):
        assert validate_simulated("Add a guard clause to prevent infinite recursion")


class TestASTValidator:
    def test_valid_python_code(self):
        valid, reason = ASTValidator.validate("x = 1\ny = x + 2\nprint(y)")
        assert valid
        assert reason == ""

    def test_syntax_error(self):
        valid, reason = ASTValidator.validate("def foo(:")
        assert not valid
        assert "Syntax error" in reason

    def test_os_module_detected(self):
        valid, reason = ASTValidator.validate("import os\nos.system('rm -rf /')")
        assert not valid
        assert "os" in reason.lower() or "dangerous" in reason.lower()

    def test_subprocess_detected(self):
        valid, reason = ASTValidator.validate("import subprocess\nsubprocess.run(['ls'])")
        assert not valid
        assert "subprocess" in reason.lower() or "dangerous" in reason.lower()

    def test_exec_detected(self):
        valid, reason = ASTValidator.validate('exec("print(1)")')
        assert not valid
        assert "exec" in reason.lower() or "dangerous" in reason.lower()

    def test_eval_detected(self):
        valid, reason = ASTValidator.validate('eval("1+1")')
        assert not valid
        assert "eval" in reason.lower() or "dangerous" in reason.lower()

    def test_empty_code(self):
        valid, reason = ASTValidator.validate("")
        assert not valid
        assert "Empty" in reason

    def test_whitespace_only(self):
        valid, reason = ASTValidator.validate("   \n  ")
        assert not valid

    def test_antibody_guard_code_valid(self):
        code = """
def safe_function():
    max_iterations = 100
    counter = 0
    while counter < max_iterations:
        counter += 1
    return counter
"""
        valid, reason = ASTValidator.validate(code)
        assert valid
        assert reason == ""


class TestValidateAntibody:
    @patch("core.sandbox.SANDBOX_MODE", "simulated")
    def test_simulated_mode(self):
        from core.sandbox import validate_antibody
        valid, _ = validate_antibody("def fix():\n    return True")
        assert valid

    @patch("core.sandbox.SANDBOX_MODE", "ast")
    def test_ast_mode_valid(self):
        from core.sandbox import validate_antibody
        valid, _ = validate_antibody("x = 1")
        assert valid

    @patch("core.sandbox.SANDBOX_MODE", "ast")
    def test_ast_mode_invalid(self):
        from core.sandbox import validate_antibody
        valid, reason = validate_antibody("import os\nos.exit(1)")
        assert not valid


# ---------------------------------------------------------------------------
# Escalation tracker
# ---------------------------------------------------------------------------
class TestEscalation:
    def test_single_failure_no_escalation(self):
        tracker = EscalationTracker()
        result = tracker.record_failure("test query", "loop detected", 1)
        assert result is None

    def test_success_resets_counter(self):
        tracker = EscalationTracker()
        tracker.record_failure("q1", "err", 1)
        tracker.record_failure("q2", "err", 1)
        tracker.record_success()
        result = tracker.record_failure("q3", "err", 1)
        assert result is None  # Only 1 consecutive after reset

    def test_escalation_after_threshold(self, tmp_path):
        tracker = EscalationTracker()
        # Override the global ESCALATION_DIR for test
        import core.escalation as esc
        original_dir = esc.ESCALATION_DIR
        esc.ESCALATION_DIR = str(tmp_path)
        esc.MAX_FAILURES = 2  # Lower threshold for test

        tracker.record_failure("q1", "err1", 1)
        result = tracker.record_failure("q2", "err2", 1)

        assert result is not None
        assert "escalation" in result
        assert os.path.exists(result)

        # Clean up
        esc.ESCALATION_DIR = original_dir

    def test_consecutive_failures_property(self):
        tracker = EscalationTracker()
        assert tracker.consecutive_failures == 0
        tracker.record_failure("q", "e", 1)
        assert tracker.consecutive_failures == 1
        tracker.record_success()
        assert tracker.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class TestASTValidatorExtended:
    """Extended AST validator tests covering edge cases."""

    def test_nested_dangerous_attribute_call(self):
        """Detect deeply nested attribute calls like os.path.join(...)"""
        valid, reason = ASTValidator.validate(
            "import os\nos.path.join('a', 'b')"
        )
        assert not valid
        assert "dangerous" in reason.lower()
        assert "os" in reason.lower()

    def test_dunder_method_call(self):
        """Detect dunder method calls via AST."""
        valid, reason = ASTValidator.validate('obj.__init__()')
        assert not valid
        assert "dunder" in reason.lower()

    def test_safe_getattr_with_dunder_string(self):
        """getattr with a dunder string arg is NOT a dunder call — should pass."""
        valid, reason = ASTValidator.validate('getattr(obj, "__init__")')
        assert valid

    def test_safe_function_with_module_name(self):
        """A function named 'os' that is actually not os module should be OK."""
        valid, reason = ASTValidator.validate(
            'class OS:\n    @staticmethod\n    def system(cmd): pass\nOS.system("ls")'
        )
        assert valid

    def test_multiline_complex_code(self):
        """Valid complex multi-line code should pass."""
        code = """
def process(data, max_iter=100):
    result = []
    for i in range(max_iter):
        if data[i] is not None:
            result.append(data[i] * 2)
        if len(result) >= 10:
            break
    return sorted(result)
"""
        valid, reason = ASTValidator.validate(code)
        assert valid, f"Expected valid, got: {reason}"


class TestValidateAntibodyExtended:
    """Extended antibody validation tests."""

    @patch("core.sandbox.SANDBOX_MODE", "simulated")
    def test_simulated_empty(self):
        from core.sandbox import validate_antibody
        valid, _ = validate_antibody("")
        assert not valid

    @patch("core.sandbox.SANDBOX_MODE", "simulated")
    def test_simulated_code_with_guard(self):
        from core.sandbox import validate_antibody
        code = "if counter > max_iterations: break"
        valid, _ = validate_antibody(code)
        assert valid

    @patch("core.sandbox.SANDBOX_MODE", "ast")
    def test_ast_mode_nested_danger(self):
        from core.sandbox import validate_antibody
        valid, reason = validate_antibody("import subprocess\nsubprocess.call(['ls'])")
        assert not valid


class TestConfig:
    """Tests for configuration validation."""

    def test_config_basic_get(self):
        from core.config import get
        val = get("MAX_ITERATIONS", 5)
        assert isinstance(val, int)

    def test_config_sandbox_modes_valid(self):
        from core.config import VALID_SANDBOX_MODES
        assert "simulated" in VALID_SANDBOX_MODES
        assert "ast" in VALID_SANDBOX_MODES
        assert "docker" in VALID_SANDBOX_MODES

    def test_config_providers_valid(self):
        from core.config import VALID_PROVIDERS
        assert "openai" in VALID_PROVIDERS
        assert "deepseek" in VALID_PROVIDERS
        assert "custom" in VALID_PROVIDERS


class TestMetrics:
    """Tests for the metrics tracking module."""

    def test_empty_metrics_summary(self):
        from core.metrics import MetricsTracker
        mt = MetricsTracker(window_size=10)
        summary = mt.get_summary()
        assert summary["status"] == "no_data"
        assert summary["records"] == 0

    def test_record_query_tracks_anomaly(self):
        from core.metrics import MetricsTracker
        mt = MetricsTracker(window_size=10)
        result = {
            "user_query": "test query",
            "final_output": "result",
            "anomalies": [{"source": "monitor", "reason": "loop detected"}],
            "antibodies": [{"code": "fix"}],
            "is_immune_active": True,
            "validation_status": "passed",
            "escalation_report": None,
        }
        mt.record_query(result)
        summary = mt.get_summary()
        assert summary["records"] == 1
        assert summary["anomaly_rate"] == 100.0
        assert "monitor" in summary["anomaly_breakdown"]

    def test_record_failed_query(self):
        from core.metrics import MetricsTracker
        mt = MetricsTracker(window_size=10)
        result = {
            "user_query": "bad query",
            "final_output": None,
            "anomalies": [],
            "antibodies": [],
            "is_immune_active": False,
            "validation_status": None,
            "escalation_report": None,
        }
        mt.record_query(result)
        summary = mt.get_summary()
        assert summary["success_rate"] == 0.0

    def test_metrics_window_limit(self):
        from core.metrics import MetricsTracker
        mt = MetricsTracker(window_size=3)
        for i in range(5):
            mt.record_query({
                "user_query": f"q{i}",
                "final_output": "ok",
                "anomalies": [],
                "antibodies": [],
                "is_immune_active": False,
                "validation_status": None,
                "escalation_report": None,
            })
        assert mt.get_summary()["records"] == 3


class TestLogger:
    def test_logger_creation(self):
        from core.logger import setup_logger
        logger = setup_logger("test_logger")
        assert logger.name == "test_logger"
        assert logger.level > 0  # Has a valid level

    def test_logger_handlers(self):
        from core.logger import setup_logger
        logger = setup_logger("test_handlers")
        assert len(logger.handlers) >= 2  # console + file
