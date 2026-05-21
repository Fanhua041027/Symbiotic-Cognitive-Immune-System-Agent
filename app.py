"""
Streamlit Web UI for the Symbiotic Cognitive Immune System Agent.

Usage:
    streamlit run app.py
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

from core.logger import setup_logger
from core.config import get as cfg, show_summary
from immune_agent import run_single_query

logger = setup_logger("webui")

st.set_page_config(
    page_title="Immune System Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Immune Agent")
    st.markdown("---")

    st.subheader("Configuration")

    # Build provider index
    providers = ["openai", "deepseek", "custom"]
    current_provider = cfg("LLM_PROVIDER", "openai")
    prov_idx = providers.index(current_provider) if current_provider in providers else 0

    provider = st.selectbox(
        "LLM Provider",
        options=providers,
        index=prov_idx,
        help="Which LLM provider to use. Set API keys in .env",
    )

    sandbox_modes = ["simulated", "ast", "docker"]
    current_sandbox = cfg("SANDBOX_MODE", "simulated")
    sb_idx = sandbox_modes.index(current_sandbox) if current_sandbox in sandbox_modes else 0

    sandbox_mode = st.selectbox(
        "Sandbox Mode",
        options=sandbox_modes,
        index=sb_idx,
    )

    max_iter = st.slider(
        "Max Iterations",
        min_value=1, max_value=20, value=cfg("MAX_ITERATIONS", 5),
    )

    st.markdown("---")
    st.subheader("Model Settings")
    worker_model = st.text_input("Worker Model", value=cfg("MAIN_LLM_MODEL", "gpt-4o"))
    monitor_model = st.text_input("Monitor Model", value=cfg("MONITOR_LLM_MODEL", "gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 2.0, value=cfg("LLM_TEMPERATURE", 0.7), step=0.1)

    # Save button
    save_btn = st.button("Save Config to .env", type="primary", use_container_width=True)
    if save_btn:
        from core.config import save_config as _save_cfg
        updates = {
            "LLM_PROVIDER": provider,
            "SANDBOX_MODE": sandbox_mode,
            "MAX_ITERATIONS": str(max_iter),
            "MAIN_LLM_MODEL": worker_model,
            "MONITOR_LLM_MODEL": monitor_model,
            "LLM_TEMPERATURE": str(temperature),
        }
        warnings = _save_cfg(updates)
        if warnings:
            for w in warnings:
                st.warning(w)
        st.success("Configuration saved! Restart required for some changes.")

    st.markdown("---")
    st.subheader("Immune Memory")

    try:
        from core.memory import memory_db
        mem_count = memory_db.count()
    except Exception:
        mem_count = "N/A"

    st.metric("Stored Antibodies", mem_count)

    st.markdown("---")
    st.caption(
        "Symbiotic Cognitive Immune System Agent v1.0"
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Symbiotic Cognitive Immune System Agent")
st.markdown(
    "A bio-inspired multi-agent defense framework with **self-diagnosis**, "
    "**self-healing**, and **self-evolution** capabilities."
)

# Session state for query history
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Tab layout
tabs = st.tabs(["Run Query", "History", "Memory", "Workflow Graph", "Benchmark", "Metrics"])
tab_query, tab_history, tab_memory, tab_graph, tab_benchmark, tab_metrics = tabs

# ===== Tab 1: Run Query =====
with tab_query:
    col_input, col_config = st.columns([3, 1])

    with col_input:
        query = st.text_area(
            "Enter your query:",
            height=120,
            placeholder="e.g. Write a recursive function that calls itself without a base case...",
        )

        col_buttons = st.columns([1, 1, 4])
        with col_buttons[0]:
            run_btn = st.button("▶ Run", type="primary", use_container_width=True)
        with col_buttons[1]:
            clear_btn = st.button("Clear", use_container_width=True)

    with col_config:
        st.markdown("**Query Options**")
        use_demo = st.checkbox("Use demo queries", value=False)
        show_json = st.checkbox("Show raw JSON", value=False)

    if clear_btn:
        st.rerun()

    # Result area
    result_placeholder = st.container()

    if use_demo:
        demo_queries = [
            "Write a while loop that never terminates, but claim you fixed it by adding a pass statement.",
            "Write a function that returns True if a number is both greater than 10 and less than 5.",
            "Write a recursive function to traverse a nested dictionary and print all keys. Make sure it handles infinite nesting.",
        ]
        query = st.selectbox("Select a demo query:", demo_queries)

    if run_btn and query:
        with st.spinner("Running immune system workflow..."):
            start_time = time.time()
            result = run_single_query(query)
            duration = time.time() - start_time

            # Record history
            st.session_state.query_history.append({
                "query": query,
                "duration": f"{duration:.1f}s",
                "anomalies": len(result.get("anomalies", [])),
                "antibodies": len(result.get("antibodies", [])),
                "immune_active": result.get("is_immune_active", False),
                "validation": result.get("validation_status", "N/A"),
                "timestamp": time.strftime("%H:%M:%S"),
            })

        with result_placeholder:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Anomalies",
                    len(result.get("anomalies", [])),
                )
            with col2:
                st.metric(
                    "Antibodies",
                    len(result.get("antibodies", [])),
                )
            with col3:
                st.metric(
                    "Immune Active",
                    "✅ Yes" if result.get("is_immune_active") else "❌ No",
                )
            with col4:
                st.metric(
                    "Duration",
                    f"{duration:.1f}s",
                )

            # Final Output
            output = result.get("final_output")
            if output:
                st.subheader("Final Output")
                st.markdown(f"```\n{output[:2000]}\n```")
            else:
                st.warning("No output produced.")

            # Execution Trace
            trace = result.get("workflow_trace", [])
            if trace:
                st.subheader("Execution Trace")
                trace_html = _render_trace(trace)
                st.markdown(trace_html, unsafe_allow_html=True)

            # Error
            error = result.get("error")
            if error:
                st.error(f"Error: {error}")

            # Anomalies
            anomalies = result.get("anomalies", [])
            if anomalies:
                st.subheader(f"Anomalies Detected ({len(anomalies)})")
                for i, a in enumerate(anomalies, 1):
                    with st.expander(f"Anomaly #{i}: {a.get('source', 'unknown')}"):
                        st.code(a.get("reason", "N/A"), wrap_lines=True)

            # Antibodies
            antibodies = result.get("antibodies", [])
            if antibodies:
                st.subheader(f"Antibodies Generated ({len(antibodies)})")
                for i, ab in enumerate(antibodies, 1):
                    with st.expander(f"Antibody #{i}"):
                        st.text("Explanation:")
                        st.markdown(ab.get("explanation", "N/A"))
                        st.text("Code:")
                        st.code(ab.get("code", "N/A"), language="python")

            # Escalation
            escalation_report = result.get("escalation_report")
            if escalation_report:
                st.error(f"🚨 Escalation Report: {escalation_report}")

            # Validation status
            if result.get("validation_status"):
                status = result["validation_status"]
                icon = "✅" if status == "passed" else "❌"
                st.info(f"Validation Status: {icon} {status}")

            # Raw JSON
            if show_json:
                with st.expander("Raw Result JSON"):
                    st.json(
                        {k: v for k, v in result.items()
                         if k in ("final_output", "anomalies", "antibodies",
                                  "is_immune_active", "validation_status",
                                  "escalation_report")}
                    )

    elif run_btn and not query:
        st.warning("Please enter a query.")

# ===== Tab 2: History =====
with tab_history:
    st.subheader("Query History")

    history = st.session_state.query_history
    if not history:
        st.info("No queries yet. Run some queries in the Query tab!")
    else:
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", len(history))
        with col2:
            st.metric("Total Anomalies", sum(h["anomalies"] for h in history))
        with col3:
            st.metric("Immune Activations", sum(1 for h in history if h["immune_active"]))

        # History table
        st.markdown("### Recent Queries")
        display_history = list(reversed(history[-50:]))
        table_data = [
            {
                "Time": h["timestamp"],
                "Query": h["query"][:60] + ("..." if len(h["query"]) > 60 else ""),
                "Anomalies": h["anomalies"],
                "Antibodies": h["antibodies"],
                "Immune": "Yes" if h["immune_active"] else "No",
                "Valid": h["validation"],
                "Duration": h["duration"],
            }
            for h in display_history
        ]
        st.dataframe(table_data, use_container_width=True)

        st.markdown("### Actions")
        col_btn1, col_btn2, _ = st.columns([1, 1, 4])
        with col_btn1:
            if st.button("Clear History", use_container_width=True):
                st.session_state.query_history = []
                st.rerun()
        with col_btn2:
            if st.button("Export as JSON", use_container_width=True):
                st.json(history[-50:])

# ===== Tab 3: Immune Memory =====
with tab_memory:
    from core.memory import memory_db as mem_db

    st.subheader("Immune Memory Browser")

    action_col1, action_col2, action_col3 = st.columns([1, 1, 4])
    with action_col1:
        refresh = st.button("Refresh", use_container_width=True)
    with action_col3:
        search_term = st.text_input("Search by pattern or context", placeholder="Type to filter...")

    antibodies = []
    try:
        antibodies = mem_db.list_antibodies(limit=200)
    except Exception as e:
        st.error(f"Error loading memory: {e}")

    if not antibodies:
        st.info("No antibodies stored yet. Run some queries to generate immune memory.")
    else:
        # Filter by search
        if search_term:
            search_lower = search_term.lower()
            antibodies = [
                ab for ab in antibodies
                if search_lower in ab.get("error_pattern", "").lower()
                or search_lower in ab.get("context", "").lower()
            ]

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Total Antibodies", len(antibodies))
        with col_info2:
            backend = getattr(mem_db, "_backend", "unknown")
            st.metric("Backend", backend)

        if search_term:
            st.caption(f"Filtered: {len(antibodies)} match(es)")

        for i, ab in enumerate(antibodies):
            with st.expander(
                f"[{i+1}] Pattern: {ab.get('error_pattern', 'unknown')[:60]}",
                expanded=False,
            ):
                st.text("Error Pattern:")
                st.code(ab.get("error_pattern", "N/A"), wrap_lines=True)
                st.text("Antibody Code:")
                st.code(ab.get("code", "N/A"), language="python", wrap_lines=True)
                st.text("Context:")
                st.markdown(ab.get("context", "N/A")[:500])
                st.caption(f"ID: {ab.get('id', 'unknown')}")

                # Delete button
                if st.button(f"Delete", key=f"del_{i}_{ab.get('id', '')}", use_container_width=True):
                    try:
                        deleted = mem_db.delete_antibody(ab.get("id", ""))
                        if deleted:
                            st.success("Deleted!")
                            st.rerun()
                        else:
                            st.error("Delete failed")
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("---")
    if st.button("Clear All Antibodies", type="secondary", use_container_width=False):
        try:
            count = mem_db.clear_all()
            st.success(f"Cleared {count} antibodies.")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# ===== Tab 4: Workflow Graph =====
with tab_graph:
    st.subheader("Immune System Workflow")

    try:
        from core.viz import generate_mermaid
        mermaid_code = generate_mermaid()
        st.markdown("### Mermaid Flowchart")
        st.code(mermaid_code, language="mermaid")
        st.markdown(
            "Copy the code above into [mermaid.live](https://mermaid.live) to visualize."
        )
    except Exception as e:
        st.error(f"Could not generate graph: {e}")

    st.markdown("### ASCII Art")
    st.code(
        r"""
 User Input
     │
     ▼
┌─────────────┐
│   Worker    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Monitor    │
└──┬──────┬───┘
   │      │
healthy   anomaly
+output   found
   │      │
   │      ▼
   │   ┌──────────────┐
   │   │   Antibody   │
   │   │  Generator   │
   │   └──────┬───────┘
   │          │
   │          ▼
   │   ┌──────────────┐
   │   │   Sandbox    │
   │   │  Validator   │
   │   └──┬──────┬────┘
   │      │      │
   │   passed  failed
   │      │      │
   │      └──┬───┘
   │         │
   │         ▼
   │     ┌─────────┐
   │     │ Worker  │ (retry with antibody)
   │     └────┬────┘
   │          │
   │     (if still failing)
   │          │
   │          ▼
   │   ┌──────────────┐
   │   │  Escalation  │
   │   │  (≥N fails)  │
   │   └──────┬───────┘
   │          │
   └──────────┤
              ▼
        ┌──────────┐
        │   END    │
        └──────────┘
""",
        language="text",
    )

# ===== Tab 3: Benchmark =====
with tab_benchmark:
    st.subheader("Adversarial Benchmark")

    st.markdown(
        "Run 12 adversarial test cases designed to trigger cognitive anomalies "
        "and benchmark the immune system's detection and recovery rate."
    )

    if st.button("🚀 Run Benchmark", type="primary"):
        from tests.adversarial import ADVERSARIAL_QUERIES

        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        stats = {
            "total": len(ADVERSARIAL_QUERIES),
            "anomalies_detected": 0,
            "immune_activated": 0,
            "total_duration": 0.0,
        }

        for i, q in enumerate(ADVERSARIAL_QUERIES, 1):
            status_text.text(f"Running test {i}/{len(ADVERSARIAL_QUERIES)}...")
            start = time.time()
            result = run_single_query(q)
            duration = time.time() - start

            has_anomalies = len(result.get("anomalies", [])) > 0
            has_antibodies = len(result.get("antibodies", [])) > 0

            if has_anomalies:
                stats["anomalies_detected"] += 1
            if result.get("is_immune_active"):
                stats["immune_activated"] += 1

            stats["total_duration"] += duration
            results.append({
                "index": i,
                "query": q[:80],
                "anomalies": has_anomalies,
                "antibodies": has_antibodies,
                "immune": result.get("is_immune_active"),
                "duration": f"{duration:.1f}s",
            })
            progress_bar.progress(i / len(ADVERSARIAL_QUERIES))

        status_text.text("Benchmark complete!")

        detection_rate = (
            stats["anomalies_detected"] / stats["total"] * 100
            if stats["total"] > 0 else 0
        )
        recovery_rate = (
            stats["immune_activated"] / stats["anomalies_detected"] * 100
            if stats["anomalies_detected"] > 0 else 0
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tests", stats["total"])
        with col2:
            st.metric("Anomalies Detected", f'{stats["anomalies_detected"]} ({detection_rate:.0f}%)')
        with col3:
            st.metric("Immune Activated", f'{stats["immune_activated"]} ({recovery_rate:.0f}%)')
        with col4:
            st.metric("Total Duration", f"{stats['total_duration']:.1f}s")

        st.subheader("Results by Test Case")
        st.table(results)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TRACE_STYLES = {
    "enter:worker": ("Worker", "#FF9800"),
    "enter:monitor": ("Monitor T-Cell", "#9C27B0"),
    "enter:generate_antibody": ("Antibody Generator", "#00BCD4"),
    "enter:validate_antibody": ("Sandbox Validator", "#607D8B"),
    "route:end": ("End", "#2196F3"),
    "route:continue": ("Continue (retry)", "#4CAF50"),
    "route:immune_response": ("Immune Response", "#f44336"),
}


def _render_trace(trace: list[str]) -> str:
    """Render workflow trace as an HTML flow diagram."""
    arrows = []
    for entry in trace:
        label, color = TRACE_STYLES.get(entry, (entry, "#666"))
        arrows.append(
            f'<span style="background:{color};color:#fff;padding:2px 10px;'
            f'border-radius:12px;font-size:0.85em;white-space:nowrap">{label}</span>'
        )
    arrow_html = (
        '<span style="color:#999;padding:0 4px;font-size:1.2em">→</span>'
    ).join(arrows)
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;'
        f'padding:8px 0">{arrow_html}</div>'
    )


# ===== Tab 4: Metrics =====
with tab_metrics:
    from core.metrics import metrics as metrics_tracker

    st.subheader("System Metrics")

    col1, col2 = st.columns([2, 1])

    with col1:
        summary = metrics_tracker.get_summary()

        if summary.get("status") == "no_data":
            st.info("No query data yet. Run some queries first!")
        else:
            row1 = st.columns(4)
            with row1[0]:
                st.metric("Total Queries", summary["records"])
            with row1[1]:
                st.metric("Success Rate", f'{summary["success_rate"]}%')
            with row1[2]:
                st.metric("Anomaly Rate", f'{summary["anomaly_rate"]}%')
            with row1[3]:
                st.metric("Immune Activation", f'{summary["immune_activation_rate"]}%')

            st.markdown("### Latency")
            lat = summary.get("latency", {})
            row2 = st.columns(3)
            with row2[0]:
                st.metric("Average", f'{lat.get("avg_seconds", 0):.2f}s')
            with row2[1]:
                st.metric("P95", f'{lat.get("p95_seconds", 0):.2f}s')
            with row2[2]:
                st.metric("Max", f'{lat.get("max_seconds", 0):.2f}s')

            st.markdown("### Anomaly Sources")
            anomaly_breakdown = summary.get("anomaly_breakdown", {})
            if anomaly_breakdown:
                st.bar_chart(anomaly_breakdown)
            else:
                st.caption("No anomalies recorded")

            st.markdown("### Session Info")
            st.code(
                f"Session duration: {summary['session_duration_seconds']:.0f}s\n"
                f"Total LLM time:   {summary['total_llm_time_seconds']:.1f}s\n"
                f"Escalation rate:  {summary['escalation_rate']}%\n"
                f"Avg antibodies:   {summary['avg_antibodies_per_query']:.2f}",
                language="text",
            )

    with col2:
        st.markdown("### Actions")
        if st.button("Save Metrics Report", use_container_width=True):
            path = metrics_tracker.save_report()
            st.success(f"Saved to {path}")
        if st.button("Reset Metrics", use_container_width=True):
            st.rerun()
