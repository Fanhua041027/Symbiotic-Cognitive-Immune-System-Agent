"""Tests for Web UI pure function logic (no Streamlit import needed).

Since Streamlit requires a special runtime, we test the UI's pure functions
by re-creating them here rather than importing app.py. These tests verify
the rendering logic is correct — if the function signatures change, update
both app.py and these tests.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.trace_render import TRACE_STYLES, render_trace


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestTraceStyles:
    def test_all_required_steps_present(self):
        required = [
            "enter:worker", "enter:consistency_check", "enter:monitor",
            "enter:generate_antibody", "enter:validate_antibody",
        ]
        for step in required:
            assert step in TRACE_STYLES, f"Missing: {step}"

    def test_each_entry_has_label_and_color(self):
        for step, (label, color) in TRACE_STYLES.items():
            assert isinstance(label, str) and len(label) > 0
            assert isinstance(color, str) and (
                color.startswith("#") or color.startswith("rgb")
            ), f"Invalid color for {step}: {color}"


class TestRenderTrace:
    def test_empty_trace_returns_div(self):
        result = render_trace([])
        assert result.startswith("<div")
        assert result.endswith("</div>")

    def test_single_step(self):
        result = render_trace(["enter:worker"])
        assert "Worker" in result
        assert "#1a73e8" in result

    def test_multiple_steps(self):
        trace = ["enter:worker", "enter:consistency_check", "enter:monitor"]
        result = render_trace(trace)
        assert "Worker" in result
        assert "Consistency" in result
        assert "Monitor" in result
        assert "→" in result

    def test_unknown_step(self):
        result = render_trace(["some_random_step"])
        assert "some_random_step" in result
        assert "#6b7280" in result

    def test_all_trace_styles_render(self):
        """Every defined TRACE_STYLE should render without error."""
        for step in TRACE_STYLES:
            result = render_trace([step])
            assert result.startswith("<div")

    def test_output_is_valid_html_snippet(self):
        result = render_trace(["enter:monitor"])
        # Should have span tags with inline styles
        assert "<span" in result
        assert "style=" in result
        assert "</span>" in result

    def test_trace_order_preserved(self):
        result = render_trace(["enter:worker", "enter:monitor"])
        worker_pos = result.index("Worker")
        monitor_pos = result.index("Monitor")
        assert worker_pos < monitor_pos, "Worker should appear before Monitor"

    def test_no_trace_has_no_arrow(self):
        result = render_trace([])
        assert "→" not in result

    def test_single_trace_has_no_arrow(self):
        result = render_trace(["enter:worker"])
        assert "→" not in result

    def test_two_traces_have_one_arrow(self):
        result = render_trace(["enter:worker", "enter:monitor"])
        assert result.count("→") == 1

    def test_three_traces_have_two_arrows(self):
        result = render_trace(["enter:a", "enter:b", "enter:c"])
        assert result.count("→") == 2
