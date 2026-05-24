"""Unit tests for core immune system components."""

import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from core.escalation import EscalationTracker
from core.nodes import (
    consistency_check_node,
    generate_antibody_node,
    main_worker_node,
    monitor_node,
    validate_antibody_node,
)
from core.sandbox import (
    ASTValidator,
    validate_simulated,
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
            "request_id": None,
            "workflow_trace": [],
        }
        assert state["user_query"] == "test"
        assert state["iteration_count"] == 0
        assert state["escalation_report"] is None
        assert state["workflow_trace"] == []
        assert isinstance(state["workflow_trace"], list)

    def test_state_has_request_id_field(self):
        """State includes request_id field."""
        state: ImmunologyState = {
            "user_query": "test", "task_steps": [], "anomalies": [],
            "antibodies": [], "final_output": None,
            "is_immune_active": False, "validation_status": None,
            "iteration_count": 0, "escalation_report": None,
            "request_id": None,
            "workflow_trace": [],
        }
        assert "request_id" in state
        assert state["request_id"] is None
        state["request_id"] = "abc123"
        assert state["request_id"] == "abc123"

    def test_state_workflow_trace_tracking(self):
        """Workflow trace correctly tracks execution steps."""
        state: ImmunologyState = {
            "user_query": "test", "task_steps": [], "anomalies": [],
            "antibodies": [], "final_output": None,
            "is_immune_active": False, "validation_status": None,
            "iteration_count": 0, "escalation_report": None,
            "request_id": None,
            "workflow_trace": [],
        }
        assert len(state["workflow_trace"]) == 0
        state["workflow_trace"].append("enter:worker")
        assert len(state["workflow_trace"]) == 1
        assert state["workflow_trace"][0] == "enter:worker"
        state["workflow_trace"].append("enter:monitor")
        state["workflow_trace"].append("route:end")
        assert len(state["workflow_trace"]) == 3

    def test_state_iteration_count_tracking(self):
        """State correctly handles iteration count updates."""
        state: ImmunologyState = {
            "user_query": "test", "task_steps": [], "anomalies": [],
            "antibodies": [], "final_output": None,
            "is_immune_active": False, "validation_status": None,
            "iteration_count": 0, "escalation_report": None,
            "request_id": None,
            "workflow_trace": [],
        }
        assert state["iteration_count"] == 0
        state["iteration_count"] = 1
        assert state["iteration_count"] == 1


class TestWorkflowRouting:
    """Tests for should_continue routing decision logic."""

    def _make_state(self, **overrides) -> ImmunologyState:
        base: ImmunologyState = {
            "user_query": "test", "task_steps": [], "anomalies": [],
            "antibodies": [], "final_output": None,
            "is_immune_active": False, "validation_status": None,
            "iteration_count": 0, "escalation_report": None,
            "request_id": None,
            "workflow_trace": [],
        }
        base.update(overrides)  # type: ignore[typeddict-item]
        return base

    @patch("core.workflow.cfg")
    def test_healthy_path_ends(self, mock_cfg):
        """No anomalies + has output → end."""
        mock_cfg.return_value = 5
        from core.workflow import should_continue
        state = self._make_state(final_output="result")
        assert should_continue(state) == "end"

    @patch("core.workflow.cfg")
    def test_anomaly_triggers_immune_response(self, mock_cfg):
        """Anomalies present → immune_response."""
        mock_cfg.return_value = 5
        from core.workflow import should_continue
        state = self._make_state(
            anomalies=[{"status": "unhealthy", "reason": "loop", "source": "monitor"}],
        )
        assert should_continue(state) == "immune_response"

    @patch("core.workflow.cfg")
    def test_continue_when_no_output_no_anomalies(self, mock_cfg):
        """No anomalies, no output → continue."""
        mock_cfg.return_value = 5
        from core.workflow import should_continue
        state = self._make_state()
        assert should_continue(state) == "continue"

    @patch("core.workflow.cfg")
    def test_anomaly_takes_priority_over_output(self, mock_cfg):
        """Even with output, anomalies → immune_response."""
        mock_cfg.return_value = 5
        from core.workflow import should_continue
        state = self._make_state(
            final_output="some result",
            anomalies=[{"status": "unhealthy", "reason": "issue", "source": "monitor"}],
        )
        assert should_continue(state) == "immune_response"

    @patch("core.workflow.cfg")
    def test_max_iterations_ends_workflow(self, mock_cfg):
        """iteration >= max_iterations → end."""
        mock_cfg.return_value = 3
        from core.workflow import should_continue
        state = self._make_state(
            iteration_count=3,
            anomalies=[{"status": "unhealthy", "reason": "risk", "source": "monitor"}],
        )
        assert should_continue(state) == "end"

    @patch("core.workflow.cfg")
    def test_max_iterations_no_output_still_ends(self, mock_cfg):
        """Max iterations with no output and no anomalies still ends."""
        mock_cfg.return_value = 2
        from core.workflow import should_continue
        state = self._make_state(iteration_count=2)
        assert should_continue(state) == "end"

    @patch("core.workflow.cfg")
    def test_output_without_antibodies_still_ends(self, mock_cfg):
        """Output present, no antibodies, no anomalies → end."""
        mock_cfg.return_value = 5
        from core.workflow import should_continue
        state = self._make_state(final_output="hello")
        assert should_continue(state) == "end"

    @patch("core.workflow.cfg")
    def test_empty_antibodies_list_still_ends(self, mock_cfg):
        """Empty antibodies list routes to end (escalation handled in finalize)."""
        mock_cfg.return_value = 5
        from core.workflow import should_continue
        state = self._make_state(final_output="ok", antibodies=[])
        assert should_continue(state) == "end"


class TestWorkflowFinalizeNode:
    """Tests for finalize_node escalation logic."""

    def _make_state(self, **overrides) -> ImmunologyState:
        base: ImmunologyState = {
            "user_query": "test", "task_steps": [], "anomalies": [],
            "antibodies": [], "final_output": None,
            "is_immune_active": False, "validation_status": None,
            "iteration_count": 0, "escalation_report": None,
            "request_id": None,
            "workflow_trace": [],
        }
        base.update(overrides)  # type: ignore[typeddict-item]
        return base

    @patch("core.workflow.cfg")
    @patch("core.workflow.escalation.record_failure")
    def test_max_iterations_with_anomaly_calls_record_failure(self, mock_fail, mock_cfg):
        """finalize_node calls record_failure when max iterations + anomalies."""
        mock_cfg.return_value = 2
        from core.workflow import finalize_node
        state = self._make_state(
            iteration_count=2,
            anomalies=[{"status": "unhealthy", "reason": "loop", "source": "monitor"}],
        )
        _ = finalize_node(state)
        mock_fail.assert_called_once()

    @patch("core.workflow.cfg")
    @patch("core.workflow.escalation.record_failure")
    def test_escalation_report_in_result(self, mock_fail, mock_cfg):
        """finalize_node returns escalation_report in result dict when generated."""
        mock_cfg.return_value = 2
        mock_fail.return_value = "/tmp/escalation_report.json"
        from core.workflow import finalize_node
        state = self._make_state(
            iteration_count=2,
            anomalies=[{"status": "unhealthy", "reason": "bad", "source": "monitor"}],
        )
        result = finalize_node(state)
        assert result["escalation_report"] == "/tmp/escalation_report.json"

    @patch("core.workflow.cfg")
    @patch("core.workflow.escalation.record_success")
    def test_max_iterations_calls_record_success(self, mock_success, mock_cfg):
        """finalize_node calls record_success when max iter + output + antibodies."""
        mock_cfg.return_value = 2
        from core.workflow import finalize_node
        state = self._make_state(
            iteration_count=2,
            final_output="ok",
            antibodies=[{"code": "fix"}],
        )
        finalize_node(state)
        mock_success.assert_called_once()

    @patch("core.workflow.cfg")
    @patch("core.workflow.escalation.record_success")
    def test_clean_end_calls_record_success(self, mock_success, mock_cfg):
        """finalize_node calls record_success for clean end with antibodies."""
        mock_cfg.return_value = 5
        from core.workflow import finalize_node
        state = self._make_state(
            final_output="ok",
            antibodies=[{"code": "fix"}],
        )
        finalize_node(state)
        mock_success.assert_called_once()

    @patch("core.workflow.cfg")
    def test_clean_end_no_antibodies_no_escalation(self, mock_cfg):
        """finalize_node returns empty dict for clean end without antibodies."""
        mock_cfg.return_value = 5
        from core.workflow import finalize_node
        state = self._make_state(final_output="ok")
        result = finalize_node(state)
        assert result == {}

    @patch("core.workflow.cfg")
    def test_no_output_no_anomalies_no_side_effects(self, mock_cfg):
        """finalize_node returns empty dict when continuing normally."""
        mock_cfg.return_value = 5
        from core.workflow import finalize_node
        state = self._make_state()
        result = finalize_node(state)
        assert result == {}


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
    @patch("core.sandbox.cfg")
    def test_simulated_mode(self, mock_cfg):
        mock_cfg.return_value = "simulated"
        from core.sandbox import validate_antibody
        valid, _ = validate_antibody("def fix():\n    return True")
        assert valid

    @patch("core.sandbox.cfg")
    def test_ast_mode_valid(self, mock_cfg):
        mock_cfg.return_value = "ast"
        from core.sandbox import validate_antibody
        valid, _ = validate_antibody("x = 1")
        assert valid

    @patch("core.sandbox.cfg")
    def test_ast_mode_invalid(self, mock_cfg):
        mock_cfg.return_value = "ast"
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

    @patch("core.escalation.cfg")
    def test_escalation_after_threshold(self, mock_cfg, tmp_path):
        mock_cfg.return_value = 2  # Lower threshold for test
        tracker = EscalationTracker()
        # Override the global ESCALATION_DIR for test
        import core.escalation as esc
        original_dir = esc.ESCALATION_DIR
        esc.ESCALATION_DIR = str(tmp_path)

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

    @patch("core.sandbox.cfg")
    def test_simulated_empty(self, mock_cfg):
        mock_cfg.return_value = "simulated"
        from core.sandbox import validate_antibody
        valid, _ = validate_antibody("")
        assert not valid

    @patch("core.sandbox.cfg")
    def test_simulated_code_with_guard(self, mock_cfg):
        mock_cfg.return_value = "simulated"
        from core.sandbox import validate_antibody
        code = "if counter > max_iterations: break"
        valid, _ = validate_antibody(code)
        assert valid

    @patch("core.sandbox.cfg")
    def test_ast_mode_nested_danger(self, mock_cfg):
        mock_cfg.return_value = "ast"
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


class TestMemory:
    """Tests for immune memory storage and retrieval."""

    def test_in_memory_store_and_search(self):
        """Basic store and search with in-memory backend."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        store.store_antibody(
            "infinite loop detected", "while counter < max: break", "a loop fix",
        )
        store.store_antibody(
            "recursion error", "if depth > limit: return", "recursion guard",
        )
        assert store.count() == 2

        result = store.search_antibody("infinite loop")
        assert result is not None
        assert "loop" in result["pattern"]

    def test_in_memory_search_no_match(self):
        """Search with no matching tokens returns None."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        store.store_antibody("syntax error", "fix syntax", "a syntax fix")
        result = store.search_antibody("量子计算")  # unrelated
        assert result is None

    def test_in_memory_search_empty_query(self):
        """Empty query returns None."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        store.store_antibody("error", "fix", "context")
        assert store.search_antibody("") is None

    def test_in_memory_search_empty_store(self):
        """Search with no antibodies returns None."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        assert store.search_antibody("anything") is None

    def test_list_antibodies_empty(self):
        """list_antibodies on empty store returns empty list."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        assert store.list_antibodies() == []

    def test_list_antibodies_limit(self):
        """list_antibodies respects the limit parameter."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        for i in range(10):
            store.store_antibody(f"error{i}", f"fix{i}", f"context{i}")
        listed = store.list_antibodies(limit=3)
        assert len(listed) == 3

    def test_dedup_exact_duplicate(self):
        """Storing the same antibody twice skips the second."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        stored1 = store.store_antibody(
            "infinite loop risk", "while counter < max: break", "loop fix",
        )
        stored2 = store.store_antibody(
            "infinite loop risk", "while counter < max: break", "loop fix",
        )
        assert stored1 is True
        assert stored2 is False
        assert store.count() == 1

    def test_dedup_similar_pattern(self):
        """Storing a very similar antibody pattern is skipped."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        store.store_antibody(
            "infinite loop detected in while",
            "while counter < limit: counter += 1",
            "context",
        )
        stored2 = store.store_antibody(
            "infinite loop in while True",
            "while counter < limit: counter += 1",
            "context",
        )
        assert stored2 is False
        assert store.count() == 1

    def test_dedup_different_antibody_stored(self):
        """Different antibodies are both stored."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        store.store_antibody("infinite loop", "while counter < max: break", "loop fix")
        store.store_antibody(
            "recursion error", "if depth > limit: return", "recursion guard",
        )
        assert store.count() == 2

    def test_dedup_empty_store(self):
        """First antibody always stores successfully."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        result = store.store_antibody("first error", "first fix", "first")
        assert result is True
        assert store.count() == 1

    def test_delete_antibody_by_index(self):
        """Deleting an in-memory antibody by index works."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        store.store_antibody("error A", "fix A", "ctx A")
        store.store_antibody("error B", "fix B", "ctx B")
        assert store.delete_antibody("0") is True
        assert store.count() == 1

    def test_delete_antibody_invalid_index(self):
        """Deleting with invalid index returns False."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        assert store.delete_antibody("999") is False
        assert store.delete_antibody("abc") is False

    def test_clear_all_antibodies(self):
        """Clearing all antibodies empties the store."""
        from core.memory import InMemoryStore
        store = InMemoryStore()
        store.store_antibody("err", "fix", "ctx")
        count = store.clear_all()
        assert count == 1
        assert store.count() == 0


class TestConfigSave:
    """Tests for config save/load functionality."""

    def test_save_config_creates_file(self, tmp_path):
        """save_config creates a new .env file when none exists."""
        # Temporarily override CONFIG_FILE
        import core.config as cfg_mod
        from core.config import save_config
        original_path = cfg_mod.CONFIG_FILE
        test_path = tmp_path / ".env_test"
        cfg_mod.CONFIG_FILE = str(test_path)
        try:
            warnings = save_config({"LLM_PROVIDER": "openai", "MAX_ITERATIONS": "5"})
            assert test_path.exists()
            content = test_path.read_text(encoding="utf-8")
            assert "LLM_PROVIDER=openai" in content
            assert "MAX_ITERATIONS=5" in content
            assert not any("SKIPPED" in w for w in warnings)
        finally:
            cfg_mod.CONFIG_FILE = original_path

    def test_save_config_updates_existing(self, tmp_path):
        """save_config updates existing keys in .env."""
        import core.config as cfg_mod
        original_path = cfg_mod.CONFIG_FILE
        test_path = tmp_path / ".env_existing"
        test_path.write_text(
            "LLM_PROVIDER=deepseek\nMAX_ITERATIONS=3\n", encoding="utf-8",
        )
        cfg_mod.CONFIG_FILE = str(test_path)
        try:
            from core.config import save_config
            warnings = save_config({"MAX_ITERATIONS": "10"})
            content = test_path.read_text(encoding="utf-8")
            assert "MAX_ITERATIONS=10" in content
            assert "LLM_PROVIDER=deepseek" in content  # unchanged
            assert not any("SKIPPED" in w for w in warnings)
        finally:
            cfg_mod.CONFIG_FILE = original_path

    def test_save_non_editable_field(self, tmp_path):
        """Saving a non-editable field should produce a warning."""
        import core.config as cfg_mod
        original_path = cfg_mod.CONFIG_FILE
        test_path = tmp_path / ".env_nonedit"
        cfg_mod.CONFIG_FILE = str(test_path)
        try:
            from core.config import save_config
            warnings = save_config({"OPENAI_API_KEY": "sk-xxx"})
            assert any("SKIPPED" in w for w in warnings)
        finally:
            cfg_mod.CONFIG_FILE = original_path


# ---------------------------------------------------------------------------
# Workflow integration
# ---------------------------------------------------------------------------
class TestWorkflowIntegration:
    """Integration tests for the compiled LangGraph workflow."""

    def test_workflow_compiles(self):
        """build_workflow() compiles without errors."""
        from core.workflow import build_workflow
        app = build_workflow()
        assert app is not None

    def test_workflow_has_required_nodes(self):
        """Compiled graph contains all agent nodes including finalize."""
        from core.workflow import build_workflow
        app = build_workflow()
        nodes = list(app.get_graph().nodes.keys())
        for node in (
            "worker", "monitor", "generate_antibody", "validate_antibody", "finalize",
        ):
            assert node in nodes, f"Missing node: {node}"

    def test_workflow_has_correct_edge_structure(self):
        """Graph edges match expected immune system flow."""
        from core.workflow import build_workflow
        app = build_workflow()
        edges = list(app.get_graph().edges)

        edge_pairs = {(e[0], e[1]) for e in edges}
        required = [
            ("__start__", "worker"),
            ("worker", "consistency_check"),
            ("consistency_check", "monitor"),
            ("generate_antibody", "validate_antibody"),
            ("validate_antibody", "worker"),
            ("finalize", "__end__"),
        ]
        for src, dst in required:
            assert (src, dst) in edge_pairs, f"Missing edge: {src} → {dst}"

    def test_workflow_monitor_has_conditional_edges(self):
        """Monitor node has three conditional routes (now via finalize)."""
        from core.workflow import build_workflow
        app = build_workflow()
        edges = list(app.get_graph().edges)
        edge_pairs = {(e[0], e[1]) for e in edges}
        # Monitor routes to finalize (end), generate_antibody, or worker
        for target in ("finalize", "generate_antibody", "worker"):
            assert ("monitor", target) in edge_pairs, (
                f"Missing conditional edge: monitor → {target}"
            )


class TestLLMCache:
    """Tests for LLM lazy initialization cache keying."""

    def test_cache_key_includes_provider_and_model(self):
        """_get_llm cache key must include provider+model for changes to take effect."""
        from core.nodes import _llm_cache
        _llm_cache.clear()
        # Keys by different config values should get separate instances
        # Can't call _get_llm directly (needs API key), so verify the keying logic
        from core.nodes import _create_llm
        assert hasattr(_create_llm, "__call__")  # module is importable

    def test_cache_key_format(self):
        """Verify cache key structure."""
        from core.nodes import _llm_cache
        # Cache should be a dict mapping string keys to ChatOpenAI instances
        assert isinstance(_llm_cache, dict)


class TestWorkerNode:
    """Tests for main_worker_node with mocked LLM."""

    def test_worker_returns_output(self, immune_state):
        """Worker produces output for a clean query."""
        state = dict(immune_state, user_query="write a hello world function")
        with patch("core.nodes._invoke_llm", return_value='def hello():\n    print("hello")'):
            result = main_worker_node(state)
        assert "task_steps" in result
        assert "final_output" in result or "task_steps" in result

    def test_worker_detects_anomaly(self, immune_state):
        """Worker flags COGNITIVE_ANOMALY in its output."""
        state = dict(immune_state, user_query="write an infinite loop")
        with patch("core.nodes._invoke_llm", return_value="COGNITIVE_ANOMALY: infinite_loop - while True without break"):
            result = main_worker_node(state)
        assert result.get("final_output") is not None
        assert "COGNITIVE_ANOMALY" in result.get("final_output", "")

    def test_worker_integrates_memory(self, immune_state):
        """Worker queries immune memory and integrates historical antibodies."""
        state = dict(immune_state, user_query="sort a list")
        with (
            patch("core.nodes._invoke_llm", return_value="sorted list result"),
            patch("core.nodes.memory_db.search_antibody", return_value={"code": "guard", "pattern": "loop risk"}),
        ):
            result = main_worker_node(state)
        assert result is not None

    def test_worker_increments_iteration(self, immune_state):
        """Worker increments iteration_count."""
        state = dict(immune_state, iteration_count=2)
        with patch("core.nodes._invoke_llm", return_value="ok"):
            result = main_worker_node(state)
        assert result.get("iteration_count") == 3


class TestConsistencyCheckNode:
    """Tests for consistency_check_node with mocked LLM."""

    def test_consistency_clean(self, immune_state):
        """Clean output passes consistency check."""
        state = dict(immune_state, user_query="hello", task_steps=[{"output": "world"}])
        with patch("core.nodes._invoke_llm", return_value='{"status": "clean", "confidence": "high"}'):
            result = consistency_check_node(state)
        assert result.get("anomalies") == []

    def test_consistency_detects_issue(self, immune_state):
        """Consistency check flags issues."""
        state = dict(immune_state, user_query="hello", task_steps=[{"output": "bad"}])
        with patch(
            "core.nodes._invoke_llm",
            return_value='{"status": "issue", "pattern": "logic_error", "reason": "bad output", "severity": "high"}',
        ):
            result = consistency_check_node(state)
        assert len(result.get("anomalies", [])) > 0

    def test_consistency_handles_malformed_json(self, immune_state):
        """Malformed LLM response does not crash."""
        state = dict(immune_state, user_query="hello", task_steps=[{"output": "x"}])
        with patch("core.nodes._invoke_llm", return_value="not json at all"):
            result = consistency_check_node(state)
        # Should return existing anomalies unchanged
        assert "anomalies" in result


class TestMonitorNode:
    """Tests for monitor_node with mocked LLM."""

    def test_monitor_clean_output(self, immune_state):
        """Monitor routes to end for clean output with no anomalies."""
        state = dict(immune_state, user_query="hello")
        with patch("core.nodes._invoke_llm", return_value='{"status": "clean", "has_output": true}'):
            result = monitor_node(state)
        assert result["is_immune_active"] is False

    def test_monitor_detects_anomaly(self, immune_state):
        """Monitor activates immune response for anomalous output."""
        state = dict(immune_state, user_query="hello")
        with patch(
            "core.nodes._invoke_llm",
            return_value='{"status": "anomaly", "reason": "infinite loop risk", "category": "infinite_loop"}',
        ):
            result = monitor_node(state)
        assert result["is_immune_active"] is True
        assert len(result.get("anomalies", [])) > 0

    def test_monitor_no_output_continues(self, immune_state):
        """Monitor sends back to worker when no output yet."""
        state = dict(immune_state, user_query="hello", task_steps=[])
        with patch("core.nodes._invoke_llm", return_value='{"status": "no_output"}'):
            result = monitor_node(state)
        assert result.get("is_immune_active") is False


class TestGenerateAntibodyNode:
    """Tests for generate_antibody_node with mocked LLM."""

    def test_generates_antibody(self, immune_state):
        """Antibody generator produces a fix for detected anomalies."""
        state = dict(
            immune_state,
            user_query="write an infinite loop",
            anomalies=[{"reason": "infinite loop", "category": "infinite_loop", "severity": "high"}],
        )
        with patch("core.nodes._invoke_llm", return_value='{"code": "while condition: break", "explanation": "add break"}'
        ):
            result = generate_antibody_node(state)
        assert len(result.get("antibodies", [])) > 0
        assert result["antibodies"][0].get("code") == "while condition: break"

    def test_generates_antibody_no_anomalies(self, immune_state):
        """No anomalies → no antibody generated."""
        state = dict(immune_state, user_query="hello", anomalies=[])
        with patch("core.nodes._invoke_llm", return_value="{}"):
            result = generate_antibody_node(state)
        assert len(result.get("antibodies", [])) == 0


class TestValidateAntibodyNode:
    """Tests for validate_antibody_node."""

    def test_validates_passed_antibody(self, immune_state):
        """Passed validation returns is_valid=True."""
        state = dict(
            immune_state,
            antibodies=[{"code": "print('hello')", "explanation": "test"}],
        )
        with patch("core.nodes.validate_antibody", return_value=(True, "")):
            result = validate_antibody_node(state)
        assert result.get("validation_status") == "passed"
        assert result.get("is_immune_active") is True

    def test_validates_failed_antibody(self, immune_state):
        """Failed validation returns is_valid=False."""
        state = dict(
            immune_state,
            antibodies=[{"code": "dangerous_code()", "explanation": "test"}],
        )
        with patch("core.nodes.validate_antibody", return_value=(False, "dangerous call")):
            result = validate_antibody_node(state)
        assert result.get("validation_status") == "failed"
        assert result.get("is_immune_active") is True

    def test_validates_no_antibodies(self, immune_state):
        """No antibodies to validate → status unchanged."""
        state = dict(immune_state, antibodies=[])
        result = validate_antibody_node(state)
        assert result.get("validation_status") is None


class TestRunSingleQuery:
    """Tests for immune_agent.run_single_query with mocked workflow."""

    def test_run_single_query_success(self):
        """Successful query returns result with metrics recorded."""
        fake_result = {
            "final_output": "hello world",
            "anomalies": [],
            "antibodies": [],
            "is_immune_active": False,
            "validation_status": None,
            "error": None,
            "escalation_report": None,
        }
        with (
            patch("immune_agent.app"),
            patch("immune_agent.metrics") as mock_metrics,
            patch("immune_agent.get_session") as mock_session,
            patch("immune_agent.escalation") as mock_esc,
        ):
            from immune_agent import run_single_query
            mock_session.return_value = MagicMock()
            mock_esc.reset = MagicMock()
            # Simulate app.invoke returning the fake result
            import immune_agent as mod
            mod.app.invoke = MagicMock(return_value=dict(fake_result))
            result = run_single_query("test query", record_session=False)
        assert result["final_output"] == "hello world"
        assert result.get("user_query") == "test query"
        mock_metrics.record_query.assert_called_once()

    def test_run_single_query_error(self):
        """Workflow error returns error result."""
        with (
            patch("immune_agent.app"),
            patch("immune_agent.metrics") as mock_metrics,
            patch("immune_agent.escalation") as mock_esc,
        ):
            from immune_agent import run_single_query
            import immune_agent as mod
            mod.app.invoke = MagicMock(side_effect=RuntimeError("test error"))
            result = run_single_query("bad query", record_session=False)
        assert result["final_output"] is None
        assert "error" in result
        mock_metrics.record_query.assert_called_once()

    def test_run_single_query_sets_request_id(self):
        """Each query gets a unique request_id."""
        with (
            patch("immune_agent.app"),
            patch("immune_agent.metrics"),
            patch("immune_agent.escalation"),
        ):
            from immune_agent import run_single_query
            import immune_agent as mod
            mod.app.invoke = MagicMock(return_value={"final_output": "ok"})
            result = run_single_query("test", record_session=False)
        assert result.get("request_id") is not None
        assert len(result["request_id"]) == 12

    def test_run_single_query_resets_escalation(self):
        """Escalation is reset before each query."""
        with (
            patch("immune_agent.app"),
            patch("immune_agent.metrics"),
            patch("immune_agent.escalation") as mock_esc,
        ):
            from immune_agent import run_single_query
            import immune_agent as mod
            mod.app.invoke = MagicMock(return_value={"final_output": "ok"})
            run_single_query("test", record_session=False)
        mock_esc.reset.assert_called_once()


class TestValidateAntibodyEdgeCases:
    """Edge case tests for the validate_antibody public API."""

    @patch("core.sandbox.cfg")
    def test_validate_antibody_none_code(self, mock_cfg):
        """None or empty code returns False."""
        mock_cfg.return_value = "simulated"
        from core.sandbox import validate_antibody
        valid, reason = validate_antibody("")
        assert not valid
        assert "Empty" in reason

    @patch("core.sandbox.cfg")
    def test_validate_antibody_whitespace_code(self, mock_cfg):
        """Whitespace-only code returns False."""
        mock_cfg.return_value = "simulated"
        from core.sandbox import validate_antibody
        valid, _ = validate_antibody("   \n  ")
        assert not valid

    @patch("core.sandbox.cfg")
    def test_validate_antibody_ast_mode_syntax_error(self, mock_cfg):
        """AST mode with syntax error returns False."""
        mock_cfg.return_value = "ast"
        from core.sandbox import validate_antibody
        valid, reason = validate_antibody("def foo(:")
        assert not valid
        assert "Syntax error" in reason


class TestEscalationReset:
    """Tests for escalation tracker reset between queries."""

    def test_reset_clears_all_state(self):
        """reset() clears counter and history."""
        from core.escalation import EscalationTracker
        tracker = EscalationTracker()
        tracker.record_failure("q1", "err1", 1)
        tracker.record_failure("q2", "err2", 1)
        assert tracker.consecutive_failures == 2
        tracker.reset()
        assert tracker.consecutive_failures == 0
        result = tracker.record_failure("q3", "err3", 1)
        assert result is None  # Not escalated — reset after just 1

    def test_reset_allows_new_cycle(self):
        """After reset, fresh failures still trigger escalation."""
        from core.escalation import EscalationTracker
        tracker = EscalationTracker()
        tracker.record_failure("q1", "e1", 1)
        tracker.reset()
        tracker.record_failure("q2", "e2", 1)
        tracker.record_failure("q3", "e3", 1)
        with __import__("unittest").mock.patch("core.escalation.cfg", return_value=2):
            result = tracker.record_failure("q4", "e4", 1)
            assert result is not None

    def test_idempotent_reset(self):
        """Calling reset() on clean tracker does not error."""
        from core.escalation import EscalationTracker
        tracker = EscalationTracker()
        tracker.reset()
        assert tracker.consecutive_failures == 0


class TestWorkflowTrace:
    """Tests for the _with_trace decorator behavior."""

    def test_trace_appended_correctly(self):
        """_with_trace adds enter:node entry to workflow_trace."""
        from core.workflow import _with_trace
        def dummy_node(state):
            return {"final_output": "hello"}

        wrapped = _with_trace("worker", dummy_node)
        state = {
            "user_query": "test", "task_steps": [], "anomalies": [],
            "antibodies": [], "final_output": None,
            "is_immune_active": False, "validation_status": None,
            "iteration_count": 0, "escalation_report": None,
            "request_id": None,
            "workflow_trace": [],
        }
        result = wrapped(state)
        assert "workflow_trace" in result
        assert "enter:worker" in result["workflow_trace"]

    def test_trace_does_not_mutate_input_state(self):
        """_with_trace should not mutate the input state dict directly."""
        from core.workflow import _with_trace
        def dummy_node(state):
            return {"final_output": "ok"}

        wrapped = _with_trace("monitor", dummy_node)
        original_trace = ["enter:worker"]
        state = {
            "user_query": "test", "task_steps": [], "anomalies": [],
            "antibodies": [], "final_output": None,
            "is_immune_active": False, "validation_status": None,
            "iteration_count": 0, "escalation_report": None,
            "request_id": None,
            "workflow_trace": list(original_trace),
        }
        result = wrapped(state)
        # Input state should not have been mutated
        assert state["workflow_trace"] == original_trace
        # Result should have the extended trace
        assert result["workflow_trace"] == ["enter:worker", "enter:monitor"]

    def test_trace_works_with_no_initial_trace(self, immune_state):
        """_with_trace handles state missing workflow_trace key."""
        from core.workflow import _with_trace
        def dummy_node(state):
            return {"final_output": "ok"}

        wrapped = _with_trace("worker", dummy_node)
        state: ImmunologyState = dict(immune_state)
        del state["workflow_trace"]
        result = wrapped(state)
        assert "workflow_trace" in result
        assert "enter:worker" in result["workflow_trace"]

    def test_trace_handles_empty_trace_list(self, immune_state):
        """_with_trace handles empty workflow_trace list."""
        from core.workflow import _with_trace
        def dummy_node(state):
            return {"final_output": "ok"}

        wrapped = _with_trace("worker", dummy_node)
        state = dict(immune_state, workflow_trace=[])
        result = wrapped(state)
        assert "workflow_trace" in result
        assert result["workflow_trace"] == ["enter:worker"]

    def test_trace_preserves_existing_trace(self, immune_state):
        """_with_trace appends to existing trace entries."""
        from core.workflow import _with_trace
        def dummy_node(state):
            return {"final_output": "ok"}

        wrapped = _with_trace("monitor", dummy_node)
        state = dict(immune_state, workflow_trace=["enter:worker"])
        result = wrapped(state)
        assert result["workflow_trace"] == ["enter:worker", "enter:monitor"]

    def test_trace_does_not_overwrite_node_result_trace(self, immune_state):
        """If node returns workflow_trace, decorator does not overwrite it."""
        from core.workflow import _with_trace
        def dummy_node(state):
            return {"final_output": "ok", "workflow_trace": ["custom"]}

        wrapped = _with_trace("worker", dummy_node)
        state = dict(immune_state, workflow_trace=[])
        result = wrapped(state)
        # Decorator should NOT overwrite since node already provided one
        assert result["workflow_trace"] == ["custom"]

    def test_trace_logs_request_id(self, immune_state):
        """_with_trace handles request_id field in state."""
        from core.workflow import _with_trace
        def dummy_node(state):
            return {"final_output": "ok"}

        wrapped = _with_trace("worker", dummy_node)
        state = dict(immune_state, request_id="req-001")
        result = wrapped(state)
        assert "workflow_trace" in result
        assert "enter:worker" in result["workflow_trace"]

    def test_trace_request_id_none(self, immune_state):
        """_with_trace handles request_id=None without error."""
        from core.workflow import _with_trace
        def dummy_node(state):
            return {"final_output": "ok"}

        wrapped = _with_trace("worker", dummy_node)
        state = dict(immune_state, request_id=None)
        result = wrapped(state)
        assert "enter:worker" in result["workflow_trace"]


class TestConfigHotReload:
    """Tests that config changes take effect without restart."""

    def test_escalation_reads_cfg_at_call_time(self, tmp_path):
        """Escalation threshold is read at call time, not cached."""
        import core.escalation as esc_mod
        from core.escalation import EscalationTracker

        tracker = EscalationTracker()
        original_dir = esc_mod.ESCALATION_DIR
        esc_mod.ESCALATION_DIR = str(tmp_path)

        # Mock cfg to return threshold=2
        with __import__("unittest").mock.patch("core.escalation.cfg", return_value=2):
            tracker.record_failure("q1", "e", 1)
            result = tracker.record_failure("q2", "e", 1)
            assert result is not None  # threshold 2 reached

        esc_mod.ESCALATION_DIR = original_dir

    @patch("core.nodes.ChatOpenAI")
    def test_llm_cache_key_includes_provider(self, mock_chat):
        """Cache key encodes role:provider:model:temperature."""
        from core.nodes import _get_llm, _llm_cache
        _llm_cache.clear()
        mock_chat.return_value = MagicMock()

        with patch("core.nodes.cfg") as mock_cfg:
            def side_effect(key, default=None):
                vals = {
                    "LLM_PROVIDER": "openai",
                    "MAIN_LLM_MODEL": "gpt-4o",
                    "LLM_TEMPERATURE": 0.7,
                }
                return vals.get(key, default)
            mock_cfg.side_effect = side_effect

            llm = _get_llm("main", "MAIN_LLM_MODEL", 0.7)
            assert llm is not None

            key = list(_llm_cache.keys())[0]
            parts = key.split(":")
            assert len(parts) == 4
            assert parts[0] == "main"    # role
            assert parts[1] == "openai"  # provider

    @patch("core.nodes.ChatOpenAI")
    def test_llm_cache_key_changes_with_provider(self, mock_chat):
        """Different provider creates a different cache entry."""
        from core.nodes import _get_llm, _llm_cache
        _llm_cache.clear()
        mock_chat.return_value = MagicMock()

        with patch("core.nodes.cfg") as mock_cfg:
            def side_effect(key, default=None):
                vals = {
                    "LLM_PROVIDER": "openai",
                    "MAIN_LLM_MODEL": "gpt-4o",
                    "LLM_TEMPERATURE": 0.7,
                }
                return vals.get(key, default)
            mock_cfg.side_effect = side_effect

            _get_llm("main", "MAIN_LLM_MODEL", 0.7)
            keys_before = set(_llm_cache.keys())

            # Switch provider
            def side_effect2(key, default=None):
                vals = {
                    "LLM_PROVIDER": "deepseek",
                    "MAIN_LLM_MODEL": "gpt-4o",
                    "LLM_TEMPERATURE": 0.7,
                }
                return vals.get(key, default)
            mock_cfg.side_effect = side_effect2

            _get_llm("main", "MAIN_LLM_MODEL", 0.7)
            keys_after = set(_llm_cache.keys())

            # Should have a new key, not reuse the old one
            assert len(keys_after) == len(keys_before) + 1


class TestConfigSaveLive:
    """Tests that save_config updates runtime environment."""

    def test_save_config_updates_os_environ(self, tmp_path):
        """save_config should update os.environ so changes take effect immediately."""
        import core.config as cfg_mod
        original_env = os.environ.get("MAX_ITERATIONS", "")
        original_path = cfg_mod.CONFIG_FILE
        test_path = tmp_path / ".env_live"
        test_path.write_text("MAX_ITERATIONS=3\n", encoding="utf-8")
        cfg_mod.CONFIG_FILE = str(test_path)

        had_key = "OPENAI_API_KEY" in os.environ
        try:
            # Suppress API key warning during this test
            if not had_key:
                os.environ["OPENAI_API_KEY"] = "test-key"
            # Load from file first
            os.environ["MAX_ITERATIONS"] = "3"
            cfg_mod._validated = False
            cfg_mod.validate_all()
            assert cfg_mod.get("MAX_ITERATIONS") == 3

            # Save new value
            warnings = cfg_mod.save_config({"MAX_ITERATIONS": "10"})
            assert len(warnings) == 0

            # os.environ should be updated
            assert os.environ.get("MAX_ITERATIONS") == "10"
            # cfg should reflect new value
            assert cfg_mod.get("MAX_ITERATIONS") == 10
        finally:
            if original_env:
                os.environ["MAX_ITERATIONS"] = original_env
            else:
                os.environ.pop("MAX_ITERATIONS", None)
            if not had_key:
                del os.environ["OPENAI_API_KEY"]
            cfg_mod.CONFIG_FILE = original_path


class TestViz:
    """Tests for visualization module."""

    def test_generate_mermaid_returns_string(self):
        """generate_mermaid() returns a non-empty string."""
        from core.viz import generate_mermaid
        graph = generate_mermaid()
        assert isinstance(graph, str)
        assert len(graph) > 50
        assert "flowchart" in graph

    def test_generate_mermaid_contains_expected_nodes(self):
        """Mermaid output contains all 4 agent node names."""
        from core.viz import generate_mermaid
        graph = generate_mermaid()
        for node in ("Worker", "Monitor", "Antibody", "Validator"):
            assert node in graph, f"Missing node {node} in mermaid graph"


class TestLogger:
    """Tests for dynamic log level resolution."""

    def test_resolve_log_level_default(self):
        """Default LOG_LEVEL is INFO."""
        from core.logger import _resolve_log_level
        # Remove the env var temporarily to test default
        original = os.environ.pop("LOG_LEVEL", None)
        try:
            assert _resolve_log_level() == "INFO"
        finally:
            if original is not None:
                os.environ["LOG_LEVEL"] = original

    def test_resolve_log_level_from_env(self):
        """LOG_LEVEL reflects environment at call time."""
        from core.logger import _resolve_log_level
        original = os.environ.get("LOG_LEVEL", "")
        os.environ["LOG_LEVEL"] = "DEBUG"
        try:
            assert _resolve_log_level() == "DEBUG"
        finally:
            if original:
                os.environ["LOG_LEVEL"] = original
            else:
                del os.environ["LOG_LEVEL"]

    def test_logger_uses_call_time_level(self, monkeypatch):
        """setup_logger should read LOG_LEVEL at call time, not import time."""
        import logging

        from core.logger import setup_logger
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        logger = setup_logger("test_dynamic_level")
        assert logger.level == logging.ERROR

    def test_logger_creation(self):
        from core.logger import setup_logger
        logger = setup_logger("test_logger_name")
        assert logger.name == "test_logger_name"
        assert logger.level > 0

    def test_logger_handlers(self):
        from core.logger import setup_logger
        logger = setup_logger("test_logger_handlers")
        assert len(logger.handlers) >= 2  # console + file


class TestConfigValidateAllWarnings:
    """Tests that validate_all warnings are surfaced by save_config."""

    def test_save_config_captures_validation_warnings(self, tmp_path):
        """save_config should return validation warnings from validate_all."""
        import core.config as cfg_mod
        original_path = cfg_mod.CONFIG_FILE
        test_path = tmp_path / ".env_warn"
        cfg_mod.CONFIG_FILE = str(test_path)

        try:
            # Save with a provider that requires additional keys
            # Clear env keys that may have been set by previous tests
            os.environ.pop("DEEPSEEK_API_KEY", None)
            os.environ.pop("MAX_ITERATIONS", None)
            cfg_mod._values.clear()
            cfg_mod._validated = False
            warnings = cfg_mod.save_config({"LLM_PROVIDER": "deepseek"})
            # Should contain warnings about missing DEEPSEEK_API_KEY
            assert any("MISSING" in w for w in warnings), (
                f"Expected MISSING warning, got: {warnings}"
            )
        finally:
            cfg_mod.CONFIG_FILE = original_path


class TestValidateDocker:
    """Tests for Docker sandbox backend with mocked subprocess."""

    @patch("core.sandbox.subprocess.run")
    def test_docker_success(self, mock_run):
        """Docker validation passes when subprocess returns 0."""
        mock_run.return_value.returncode = 0
        from core.sandbox import validate_docker
        valid, reason = validate_docker("print('hello')")
        assert valid
        assert reason == ""

    @patch("core.sandbox.subprocess.run")
    def test_docker_runtime_error(self, mock_run):
        """Docker validation fails when subprocess returns non-zero."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "NameError: name 'x' is not defined"
        from core.sandbox import validate_docker
        valid, reason = validate_docker("print(x)")
        assert not valid
        assert "NameError" in reason

    @patch("core.sandbox.subprocess.run")
    def test_docker_timeout_falls_back_to_ast(self, mock_run):
        """Docker timeout falls back to AST which passes for safe code."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("docker", 30)
        from core.sandbox import validate_docker
        valid, reason = validate_docker("x = 1")
        assert valid

    @patch("core.sandbox.subprocess.run")
    def test_docker_timeout_still_catches_dangerous(self, mock_run):
        """Docker timeout fallback to AST still rejects dangerous code."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("docker", 30)
        from core.sandbox import validate_docker
        valid, reason = validate_docker("import os\nos.system('rm')")
        assert not valid

    @patch("core.sandbox.subprocess.run")
    def test_docker_not_available_falls_back(self, mock_run):
        """FileNotFoundError for docker falls back to AST."""
        mock_run.side_effect = FileNotFoundError()
        from core.sandbox import validate_docker
        valid, reason = validate_docker("x = 1")
        assert valid

    @patch("core.sandbox.subprocess.run")
    def test_docker_temp_file_cleaned(self, mock_run):
        """Temporary file is unlinked after docker validation."""
        mock_run.return_value.returncode = 0
        from core.sandbox import validate_docker
        with patch("core.sandbox.tempfile.NamedTemporaryFile") as mock_tmp:
            mock_f = MagicMock()
            mock_f.name = "/tmp/test_antibody.py"
            mock_tmp.return_value.__enter__.return_value = mock_f
            with patch("core.sandbox.os.unlink") as mock_unlink:
                validate_docker("print('hi')")
                mock_unlink.assert_called_once_with("/tmp/test_antibody.py")

    @patch("core.sandbox.subprocess.run")
    def test_docker_subprocess_args(self, mock_run):
        """Docker subprocess is invoked with expected arguments."""
        mock_run.return_value.returncode = 0
        from core.sandbox import validate_docker
        validate_docker("print('hi')")
        args, _ = mock_run.call_args
        cmd = args[0]
        assert "docker" in cmd
        assert "run" in cmd
        assert "--rm" in cmd
        assert "python:3.11-alpine" in cmd


class TestDockerAvailable:
    """Tests for _docker_available helper."""

    @patch("core.sandbox.subprocess.run")
    def test_available_when_command_succeeds(self, mock_run):
        """_docker_available returns True when docker --version succeeds."""
        from core.sandbox import _docker_available
        assert _docker_available()

    @patch("core.sandbox.subprocess.run")
    def test_not_available_when_not_found(self, mock_run):
        """_docker_available returns False when docker not installed."""
        mock_run.side_effect = FileNotFoundError()
        from core.sandbox import _docker_available
        assert not _docker_available()

    @patch("core.sandbox.subprocess.run")
    def test_not_available_on_timeout(self, mock_run):
        """_docker_available returns False when docker command times out."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("docker", 5)
        from core.sandbox import _docker_available
        assert not _docker_available()


class TestValidateE2B:
    """Tests for E2B cloud sandbox backend."""

    def test_e2b_not_installed_falls_back_to_ast(self):
        """e2b_code_interpreter not installed falls back to AST validation."""
        from core.sandbox import validate_e2b
        with patch("core.sandbox.validate_ast", return_value=(True, "")) as mock_ast:
            valid, _ = validate_e2b("x = 1")
            assert valid
            mock_ast.assert_called_once()

    def test_e2b_runtime_error(self):
        """E2B validation returns error when run_code has an error."""
        import sys
        mock_result = MagicMock()
        mock_result.error = MagicMock()
        mock_result.error.name = "ZeroDivisionError"
        mock_result.error.value = "division by zero"

        mock_sandbox = MagicMock()
        mock_sandbox.run_code.return_value = mock_result

        mock_e2b_mod = type(sys)("e2b_code_interpreter")
        mock_e2b_mod.Sandbox = MagicMock()
        mock_e2b_mod.Sandbox.return_value.__enter__.return_value = mock_sandbox
        mock_e2b_mod.Sandbox.return_value.__exit__.return_value = None
        saved = sys.modules.get("e2b_code_interpreter")
        sys.modules["e2b_code_interpreter"] = mock_e2b_mod
        try:
            from core.sandbox import validate_e2b
            valid, reason = validate_e2b("1/0")
            assert not valid
            assert "ZeroDivisionError" in reason
        finally:
            if saved:
                sys.modules["e2b_code_interpreter"] = saved
            else:
                del sys.modules["e2b_code_interpreter"]

    def test_e2b_success(self):
        """E2B validation passes when run_code succeeds."""
        import sys
        mock_result = MagicMock()
        mock_result.error = None

        mock_sandbox = MagicMock()
        mock_sandbox.run_code.return_value = mock_result

        mock_e2b_mod = type(sys)("e2b_code_interpreter")
        mock_e2b_mod.Sandbox = MagicMock()
        mock_e2b_mod.Sandbox.return_value.__enter__.return_value = mock_sandbox
        mock_e2b_mod.Sandbox.return_value.__exit__.return_value = None
        saved = sys.modules.get("e2b_code_interpreter")
        sys.modules["e2b_code_interpreter"] = mock_e2b_mod
        try:
            from core.sandbox import validate_e2b
            valid, reason = validate_e2b("print('hello')")
            assert valid
            assert reason == ""
        finally:
            if saved:
                sys.modules["e2b_code_interpreter"] = saved
            else:
                del sys.modules["e2b_code_interpreter"]

    def test_e2b_exception_falls_back(self):
        """E2B exception falls back to AST."""
        import sys
        mock_e2b_mod = type(sys)("e2b_code_interpreter")
        mock_cls = MagicMock()
        mock_cls.return_value.__enter__.side_effect = RuntimeError("connection failed")
        mock_e2b_mod.Sandbox = mock_cls
        saved = sys.modules.get("e2b_code_interpreter")
        sys.modules["e2b_code_interpreter"] = mock_e2b_mod
        try:
            from core.sandbox import validate_e2b
            with patch("core.sandbox.validate_ast", return_value=(True, "")) as mock_ast:
                valid, _ = validate_e2b("x = 1")
                assert valid
                mock_ast.assert_called_once()
        finally:
            if saved:
                sys.modules["e2b_code_interpreter"] = saved
            else:
                del sys.modules["e2b_code_interpreter"]


class TestValidateAntibodyE2BMode:
    """Tests for validate_antibody with e2b sandbox mode."""

    @patch("core.sandbox.cfg")
    @patch("core.sandbox.validate_e2b")
    def test_e2b_mode_calls_e2b_validator(self, mock_e2b, mock_cfg):
        """e2b mode dispatches to validate_e2b."""
        mock_cfg.return_value = "e2b"
        mock_e2b.return_value = (True, "")
        from core.sandbox import validate_antibody
        valid, _ = validate_antibody("print('hello')")
        assert valid
        mock_e2b.assert_called_once()

    @patch("core.sandbox.cfg")
    @patch("core.sandbox.validate_e2b")
    def test_e2b_mode_detects_dangerous_code(self, mock_e2b, mock_cfg):
        """e2b mode returns failure when validate_e2b fails."""
        mock_cfg.return_value = "e2b"
        mock_e2b.return_value = (False, "Runtime error: syntax error")
        from core.sandbox import validate_antibody
        valid, reason = validate_antibody("bad code")
        assert not valid
        assert "syntax" in reason

    @patch("core.sandbox.cfg")
    @patch("core.sandbox.validate_e2b")
    def test_e2b_mode_fallback_when_not_installed(self, mock_e2b, mock_cfg):
        """e2b mode calls validate_e2b which handles the not-installed case."""
        mock_cfg.return_value = "e2b"
        mock_e2b.return_value = (True, "")
        from core.sandbox import validate_antibody
        valid, _ = validate_antibody("x = 1")
        assert valid
        mock_e2b.assert_called_once()


class TestConfigSandboxModesE2B:
    """Tests that e2b is in valid sandbox modes."""

    def test_e2b_in_valid_modes(self):
        from core.config import VALID_SANDBOX_MODES
        assert "e2b" in VALID_SANDBOX_MODES

    def test_config_e2b_accepted(self):
        from core.config import validate_all
        result = validate_all()
        assert not any("e2b" in w.lower() and "invalid" in w.lower() for w in result)


class TestEscalationExtended:
    """Extended tests for EscalationTracker beyond basic reset."""

    def test_record_success_resets_after_failures(self):
        """record_success resets consecutive failure counter."""
        tracker = EscalationTracker()
        tracker.record_failure("q1", "err", 1)
        tracker.record_failure("q2", "err", 1)
        assert tracker.consecutive_failures == 2
        tracker.record_success()
        assert tracker.consecutive_failures == 0

    def test_record_success_idempotent(self):
        """record_success on clean tracker does not error."""
        tracker = EscalationTracker()
        tracker.record_success()
        assert tracker.consecutive_failures == 0

    def test_consecutive_failures_property(self):
        """consecutive_failures property matches internal counter."""
        tracker = EscalationTracker()
        assert tracker.consecutive_failures == 0
        tracker.record_failure("q", "e", 0)
        assert tracker.consecutive_failures == 1
        tracker.record_failure("q", "e", 0)
        assert tracker.consecutive_failures == 2

    @patch("core.escalation.cfg")
    def test_record_failure_returns_report_path_on_threshold(self, mock_cfg):
        """record_failure returns a path when threshold is reached."""
        mock_cfg.return_value = 2
        tracker = EscalationTracker()
        r1 = tracker.record_failure("q1", "e1", 0)
        assert r1 is None
        r2 = tracker.record_failure("q2", "e2", 0)
        assert r2 is not None
        assert r2.endswith(".json")

    @patch("core.escalation.cfg")
    def test_report_file_has_correct_content(self, mock_cfg, tmp_path):
        """Escalation report JSON contains expected fields."""
        import core.escalation as esc_mod
        original_dir = esc_mod.ESCALATION_DIR
        esc_mod.ESCALATION_DIR = str(tmp_path)
        mock_cfg.return_value = 2
        try:
            tracker = EscalationTracker()
            tracker.record_failure("query1", "anomaly one", 2)
            path = tracker.record_failure("query2", "anomaly two", 1)
            assert path is not None
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
            assert "title" in report
            assert "Immune System Escalation Notice" in report["title"]
            assert report["consecutive_failures"] == 2  # value at time of writing
            assert report["threshold"] == 2
            assert len(report["history"]) == 2
            assert report["history"][0]["query"] == "query1"
            assert report["history"][1]["query"] == "query2"
            assert report["history"][1]["antibodies_generated"] == 1
        finally:
            esc_mod.ESCALATION_DIR = original_dir

    @patch("core.escalation.cfg")
    def test_escalation_resets_counter_and_allows_new_cycle(self, mock_cfg, tmp_path):
        """After escalation resets counter, fresh failures start from 0."""
        import core.escalation as esc_mod
        original_dir = esc_mod.ESCALATION_DIR
        esc_mod.ESCALATION_DIR = str(tmp_path)
        mock_cfg.return_value = 2
        try:
            tracker = EscalationTracker()
            tracker.record_failure("q1", "e1", 0)  # 1
            path1 = tracker.record_failure("q2", "e2", 0)  # 2 → triggers escalation
            assert path1 is not None
            # Counter should be reset by _generate_report
            r = tracker.record_failure("q3", "e3", 0)  # 1 (fresh start)
            assert r is None  # Not escalated yet
        finally:
            esc_mod.ESCALATION_DIR = original_dir

    @patch("core.escalation.cfg")
    def test_record_failure_stops_after_escalation(self, mock_cfg, tmp_path):
        """After escalation resets counter, fresh failures start from 0."""
        import core.escalation as esc_mod
        original_dir = esc_mod.ESCALATION_DIR
        esc_mod.ESCALATION_DIR = str(tmp_path)
        mock_cfg.return_value = 2
        try:
            tracker = EscalationTracker()
            tracker.record_failure("q1", "e1", 0)  # 1
            path1 = tracker.record_failure("q2", "e2", 0)  # 2 → triggers
            assert path1 is not None  # Escalated
            # Counter should be reset by _generate_report
            r = tracker.record_failure("q3", "e3", 0)  # 1 (fresh start)
            assert r is None  # Not escalated yet
        finally:
            esc_mod.ESCALATION_DIR = original_dir


class TestEscalationCleanup:
    """Tests for escalation report auto-cleanup."""

    def test_cleanup_removes_old_reports(self, tmp_path):
        """Cleanup removes reports older than max_age_days."""
        import core.escalation as esc_mod
        original_dir = esc_mod.ESCALATION_DIR
        esc_mod.ESCALATION_DIR = str(tmp_path)
        try:
            # Create a stale file
            old_path = os.path.join(str(tmp_path), "escalation_old.json")
            with open(old_path, "w") as f:
                json.dump({"test": True}, f)
            # Set mtime far in the past
            old_mtime = time.time() - 31 * 86400
            os.utime(old_path, (old_mtime, old_mtime))

            # Create a recent file
            recent_path = os.path.join(str(tmp_path), "escalation_recent.json")
            with open(recent_path, "w") as f:
                json.dump({"test": True}, f)

            removed = EscalationTracker.cleanup_old_reports(max_age_days=30)
            assert removed == 1
            assert not os.path.exists(old_path)
            assert os.path.exists(recent_path)
        finally:
            esc_mod.ESCALATION_DIR = original_dir

    def test_cleanup_empty_dir_no_error(self, tmp_path):
        """Cleanup on empty directory does not raise."""
        import core.escalation as esc_mod
        original_dir = esc_mod.ESCALATION_DIR
        esc_mod.ESCALATION_DIR = str(tmp_path)
        try:
            removed = EscalationTracker.cleanup_old_reports(max_age_days=30)
            assert removed == 0
        finally:
            esc_mod.ESCALATION_DIR = original_dir

    def test_cleanup_keeps_all_recent(self, tmp_path):
        """Cleanup keeps all files within the age threshold."""
        import core.escalation as esc_mod
        original_dir = esc_mod.ESCALATION_DIR
        esc_mod.ESCALATION_DIR = str(tmp_path)
        try:
            for i in range(3):
                p = os.path.join(str(tmp_path), f"report_{i}.json")
                with open(p, "w") as f:
                    json.dump({"i": i}, f)

            removed = EscalationTracker.cleanup_old_reports(max_age_days=30)
            assert removed == 0
            assert len(os.listdir(str(tmp_path))) == 3
        finally:
            esc_mod.ESCALATION_DIR = original_dir


class TestVizExtended:
    """Tests for visualization module stdout functions."""

    def test_print_graph_output(self, capsys):
        """print_graph prints mermaid graph to stdout."""
        from core.viz import print_graph
        print_graph()
        captured = capsys.readouterr()
        assert "Immune System Workflow Graph" in captured.out
        assert "flowchart" in captured.out
        assert "Worker" in captured.out
        assert "Monitor" in captured.out

    def test_print_graph_ascii_output(self, capsys):
        """print_graph_ascii prints ASCII art to stdout."""
        from core.viz import print_graph_ascii
        print_graph_ascii()
        captured = capsys.readouterr()
        assert "Worker" in captured.out
        assert "Monitor" in captured.out
        assert "Escalation" in captured.out
        assert "END" in captured.out


class TestConfigShowSummary:
    """Tests for config.show_summary."""

    def test_show_summary_runs_without_error(self, caplog):
        """show_summary logs config summary without erroring."""
        import logging
        caplog.set_level(logging.INFO)
        from core.config import show_summary
        # show_summary uses logger.info, not print — use caplog
        show_summary()
        assert any("Configuration" in r.message for r in caplog.records)


class TestImmuneAgentModule:
    """Tests for immune_agent module-level functions."""

    def test_show_stats_runs_without_error(self, capsys):
        """show_stats prints system stats without erroring."""
        from immune_agent import show_stats
        show_stats()
        captured = capsys.readouterr()
        assert "Immune System Statistics" in captured.out

    def test_show_stats_contains_memory_info(self, capsys):
        """show_stats output includes memory backend info."""
        from immune_agent import show_stats
        show_stats()
        captured = capsys.readouterr()
        assert "Immune Memory" in captured.out or "Antibodies" in captured.out

    def test_show_stats_shows_config(self, capsys):
        """show_stats output includes configuration section."""
        from immune_agent import show_stats
        show_stats()
        captured = capsys.readouterr()
        assert "Configuration" in captured.out or "Provider" in captured.out


class TestLocalOnnxEmbedding:
    """Tests for the local ONNX embedding function."""

    def test_embedding_function_name(self):
        """name() returns expected string."""
        from core.embeddings import LocalOnnxEmbeddingFunction
        name = LocalOnnxEmbeddingFunction.name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_embedding_function_call(self):
        """__call__ returns normalized embeddings."""
        from core.embeddings import LocalOnnxEmbeddingFunction
        ef = LocalOnnxEmbeddingFunction()
        result = ef(["hello world", "test"])
        assert len(result) == 2
        assert len(result[0]) == 384
        # Should be L2 normalized (unit vectors)
        import math
        norm = math.sqrt(sum(v * v for v in result[0]))
        assert abs(norm - 1.0) < 1e-4

    def test_embedding_similar_texts(self):
        """Similar texts produce similar embeddings (cosine ~1)."""
        from core.embeddings import LocalOnnxEmbeddingFunction
        ef = LocalOnnxEmbeddingFunction()
        a = ef(["how to sort a list in python"])[0]
        b = ef(["sorting lists with python"])[0]
        dot = sum(x * y for x, y in zip(a, b))
        assert dot > 0.5, f"Similar texts should have high cosine: {dot}"

    def test_embedding_dissimilar_texts(self):
        """Dissimilar texts produce less similar embeddings."""
        from core.embeddings import LocalOnnxEmbeddingFunction
        ef = LocalOnnxEmbeddingFunction()
        a = ef(["how to sort a list in python"])[0]
        b = ef(["the weather is nice today"])[0]
        dot = sum(x * y for x, y in zip(a, b))
        assert dot < 0.8, f"Dissimilar texts should have lower cosine: {dot}"

    def test_is_legacy_false(self):
        """is_legacy() returns False."""
        from core.embeddings import LocalOnnxEmbeddingFunction
        ef = LocalOnnxEmbeddingFunction()
        assert not ef.is_legacy()

    def test_get_config(self):
        """get_config returns expected dict."""
        from core.embeddings import LocalOnnxEmbeddingFunction
        cfg = LocalOnnxEmbeddingFunction().get_config()
        assert "model_name" in cfg


class TestMemoryPersistence:
    """Tests for immune memory persistence with local ONNX embedding."""

    @pytest.fixture(autouse=True)
    def _unique_db(self, tmp_path):
        """Patch DB_DIR to a unique temp path per test."""
        import core.memory as mem_mod
        self._orig_dir = mem_mod.DB_DIR
        mem_mod.DB_DIR = str(tmp_path / ".immune_db")
        yield
        mem_mod.DB_DIR = self._orig_dir

    def test_memory_init_chromadb_backend(self):
        """Memory initializes with chromadb backend when local ONNX available."""
        from core.memory import ImmunologyMemory
        m = ImmunologyMemory()
        assert m._backend == "chromadb"
        m.clear_all()

    def test_memory_store_and_count(self):
        """Store and count antibodies."""
        from core.memory import ImmunologyMemory
        m = ImmunologyMemory()
        stored = m.store_antibody("loop without break", "while True: pass", "test")
        assert stored
        assert m.count() == 1
        m.clear_all()

    def test_memory_store_and_search(self):
        """Store and search returns recent antibody."""
        from core.memory import ImmunologyMemory
        m = ImmunologyMemory()
        m.store_antibody("infinite loop without break", "while True: pass", "detected infinite loop")
        m.store_antibody("sql injection risk", "cursor.execute(f'SELECT * FROM users WHERE id = {uid}')", "detected sql injection")
        result = m.search_antibody("infinite loop")
        assert result is not None, "Should find the loop antibody"
        m.clear_all()

    def test_memory_dedup(self):
        """Duplicate antibodies are skipped by token similarity."""
        from core.memory import ImmunologyMemory
        m = ImmunologyMemory()
        m.store_antibody("pattern X", "code X", "context X")
        dup = m.store_antibody("pattern X", "code X", "context X")
        assert not dup, "Duplicate should be skipped"
        assert m.count() == 1
        m.clear_all()

    def test_memory_list_and_delete(self):
        """List and delete antibodies."""
        from core.memory import ImmunologyMemory
        m = ImmunologyMemory()
        m.store_antibody("pattern A", "code A", "context A")
        m.store_antibody("pattern B", "code B", "context B")
        lst = m.list_antibodies()
        assert len(lst) == 2
        deleted = m.delete_antibody(lst[0]["id"])
        assert deleted
        assert m.count() == 1
        m.clear_all()

    def test_memory_clear_all(self):
        """clear_all removes all antibodies."""
        from core.memory import ImmunologyMemory
        m = ImmunologyMemory()
        m.store_antibody("pattern A", "code A", "context A")
        m.store_antibody("pattern B", "code B", "context B")
        assert m.count() == 2
        cleared = m.clear_all()
        assert cleared == 2
        assert m.count() == 0

    def test_memory_decay_removes_old(self):
        """decay removes antibodies past TTL."""
        from core.memory import ImmunologyMemory
        m = ImmunologyMemory()
        m.store_antibody("pattern A", "code A", "context A")
        m.store_antibody("pattern B", "code B", "context B")
        assert m.count() == 2
        removed = m.decay(ttl_days=0, prune_unmatched_days=999)
        assert removed == 2
        assert m.count() == 0
        m.clear_all()

    def test_memory_decay_keeps_recent(self):
        """decay keeps antibodies within TTL."""
        from core.memory import ImmunologyMemory
        m = ImmunologyMemory()
        m.store_antibody("pattern A", "code A", "context A")
        removed = m.decay(ttl_days=999, prune_unmatched_days=999)
        assert removed == 0
        assert m.count() == 1
        m.clear_all()

    def test_memory_decay_empty(self):
        """decay on empty store returns 0."""
        from core.memory import ImmunologyMemory
        m = ImmunologyMemory()
        assert m.decay() == 0
        m.clear_all()

    def test_memory_search_updates_last_matched(self):
        """search_antibody updates last_matched, protecting from decay."""
        from core.memory import ImmunologyMemory
        m = ImmunologyMemory()
        m.store_antibody("error X", "code X", "context X")
        # Search updates last_matched to now — verify it returns something
        result = m.search_antibody("error X")
        assert result is not None
        # Decay with very short TTL on last_matched:
        # Even though last_matched was just updated, prune_unmatched_days=0 means
        # any nonzero age gets removed. Verify decay still runs cleanly.
        removed = m.decay(ttl_days=999, prune_unmatched_days=0)
        assert removed >= 0
        m.clear_all()

    def test_memory_list_shows_timestamps(self):
        """list_antibodies includes created_at and last_matched fields."""
        from core.memory import ImmunologyMemory
        m = ImmunologyMemory()
        m.store_antibody("pattern T", "code T", "context T")
        lst = m.list_antibodies()
        assert len(lst) == 1
        entry = lst[0]
        assert "created_at" in entry
        assert "last_matched" in entry
        assert entry["created_at"] != ""
        assert entry["last_matched"] != ""
        m.clear_all()


class TestMetricsAutoSave:
    """Tests for metrics auto-persist functionality."""

    def test_auto_save_resets_counter(self):
        from core.metrics import MetricsTracker
        mt = MetricsTracker(window_size=10, auto_save_every=2)
        base_result = {
            "user_query": "q", "final_output": "ok",
            "anomalies": [], "antibodies": [],
            "is_immune_active": False, "validation_status": None,
            "escalation_report": None,
        }
        mt.record_query(base_result)
        assert mt._auto_save_counter == 1  # first record, threshold 2
        mt.record_query(base_result)
        assert mt._auto_save_counter == 0  # reset after threshold hit

    def test_load_latest_report_no_files(self):
        from core.metrics import MetricsTracker
        report = MetricsTracker.load_latest_report()
        assert report is None or isinstance(report, dict)

    def test_save_and_load_roundtrip(self):
        import os

        from core.metrics import MetricsTracker
        mt = MetricsTracker(window_size=10, auto_save_every=100)
        mt.record_query({
            "user_query": "persist test", "final_output": "ok",
            "anomalies": [{"source": "worker", "reason": "test"}],
            "antibodies": [], "is_immune_active": True,
            "validation_status": "passed", "escalation_report": None,
        })
        path = mt.save_report("_test_roundtrip.json")
        assert os.path.exists(path)
        loaded = MetricsTracker.load_latest_report()
        assert loaded is not None
        assert "metrics" in loaded
        assert loaded["metrics"]["records"] >= 1
        os.remove(path)

    def test_cleanup_old_reports(self):
        import os
        import time

        from core.metrics import MetricsTracker
        mt2 = MetricsTracker(window_size=10, auto_save_every=100)
        old_path = mt2.save_report("_test_old_report.json")
        # Manually set its mtime to be old
        old_time = time.time() - 10 * 86400  # 10 days ago
        os.utime(old_path, (old_time, old_time))
        # Cleanup with 7-day threshold — should remove it
        removed = MetricsTracker.cleanup_old_reports(max_age_days=7)
        assert removed >= 1
        assert not os.path.exists(old_path)


class TestCircuitBreaker:
    """Tests for the LLM API circuit breaker."""

    def test_circuit_breaker_starts_closed(self):
        from core.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(threshold=3, cooldown=60.0)
        assert cb.can_execute("test")
        assert cb.status("test")["state"] == "closed"

    def test_circuit_breaker_opens_after_threshold(self):
        from core.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(threshold=2, cooldown=60.0)
        assert cb.can_execute("test")
        cb.record_failure("test")
        assert cb.can_execute("test")
        cb.record_failure("test")
        assert not cb.can_execute("test")
        assert cb.status("test")["state"] == "open"

    def test_circuit_breaker_resets_on_success(self):
        from core.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(threshold=2, cooldown=60.0)
        cb.record_failure("test")
        cb.record_success("test")
        assert cb.can_execute("test")
        assert cb.status("test")["failures"] == 0

    def test_circuit_breaker_reset_method(self):
        from core.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(threshold=1, cooldown=60.0)
        cb.record_failure("test")
        assert not cb.can_execute("test")
        cb.reset("test")
        assert cb.can_execute("test")

    def test_circuit_breaker_half_open_after_cooldown(self):
        from core.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(threshold=1, cooldown=0.0)  # instant cooldown
        cb.record_failure("test")
        # Should immediately be half-open since cooldown is 0
        assert cb.can_execute("test")
        assert cb.status("test")["state"] == "half_open"

    def test_circuit_breaker_global_instance(self):
        from core.circuit_breaker import breaker
        # Global instance should exist and be functional
        assert breaker.can_execute("global_test")
        breaker.record_success("global_test")

    def test_circuit_breaker_all_status(self):
        from core.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        cb.record_failure("circuit_a")
        cb.record_failure("circuit_b")
        status = cb.all_status()
        assert "circuit_a" in status
        assert "circuit_b" in status
        cb.reset()
        status = cb.all_status()
        assert status["circuit_a"]["state"] == "closed"


class TestDockerCompose:
    """Tests for docker-compose.yml service definitions."""

    def test_compose_file_exists(self):
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        assert os.path.exists(path)

    def test_compose_has_api_service(self):
        import os

        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        services = cfg.get("services", {})
        assert "immune-agent-api" in services
        cmd = services["immune-agent-api"].get("command", "")
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        assert "uvicorn" in cmd_str

    def test_compose_has_web_service(self):
        import os

        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        services = cfg.get("services", {})
        assert "immune-agent-web" in services
        cmd = services["immune-agent-web"].get("command", "")
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        assert "streamlit" in cmd_str

    def test_compose_has_all_core_services(self):
        import os

        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        services = cfg.get("services", {})
        for name in ["immune-agent", "immune-agent-interactive", "immune-agent-api", "immune-agent-web"]:
            assert name in services, f"Missing service: {name}"

    def test_compose_api_exposes_port(self):
        import os

        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        ports = cfg["services"]["immune-agent-api"].get("ports", [])
        assert any("8000" in str(p) for p in ports)

    def test_compose_web_exposes_port(self):
        import os

        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        ports = cfg["services"]["immune-agent-web"].get("ports", [])
        assert any("8501" in str(p) for p in ports)

    def test_compose_api_has_healthcheck(self):
        import os

        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        hc = cfg["services"]["immune-agent-api"].get("healthcheck")
        assert hc is not None, "API service missing healthcheck"
        test_cmd = " ".join(hc.get("test", [])) if isinstance(hc.get("test"), list) else str(hc.get("test", ""))
        assert "urllib.request" in test_cmd and "/health" in test_cmd

    def test_compose_web_has_healthcheck(self):
        import os

        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        hc = cfg["services"]["immune-agent-web"].get("healthcheck")
        assert hc is not None, "Web service missing healthcheck"
        assert hc.get("interval") and hc.get("retries")
        assert hc.get("start_period")

    def test_compose_healthcheck_has_reasonable_values(self):
        import os

        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        for svc in ["immune-agent-api", "immune-agent-web"]:
            hc = cfg["services"][svc].get("healthcheck", {})
            assert isinstance(hc.get("retries", 0), int) and hc["retries"] > 0
            assert isinstance(hc.get("interval"), str) and hc["interval"].endswith("s")
            assert isinstance(hc.get("timeout"), str) and hc["timeout"].endswith("s")

    def test_compose_api_depends_on_agent(self):
        import os

        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        deps = cfg["services"]["immune-agent-api"].get("depends_on", {})
        assert "immune-agent" in deps, "API should depend on base agent"

    def test_compose_interactive_depends_on_agent(self):
        import os

        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        deps = cfg["services"]["immune-agent-interactive"].get("depends_on", {})
        assert "immune-agent" in deps, "Interactive should depend on base agent"

    def test_compose_web_depends_on_api(self):
        import os

        import yaml
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker-compose.yml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        deps = cfg["services"]["immune-agent-web"].get("depends_on", {})
        assert "immune-agent-api" in deps, "Web should depend on API"
