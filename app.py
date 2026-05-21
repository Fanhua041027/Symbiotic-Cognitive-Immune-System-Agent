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
    provider = st.selectbox(
        "LLM Provider",
        options=["openai", "deepseek", "custom"],
        index=0,
        help="Which LLM provider to use. Set API keys in .env",
    )

    sandbox_mode = st.selectbox(
        "Sandbox Mode",
        options=["simulated", "ast", "docker"],
        index=0,
    )

    max_iter = st.slider(
        "Max Iterations",
        min_value=1, max_value=20, value=cfg("MAX_ITERATIONS", 5),
    )

    st.markdown("---")
    st.subheader("Immune Memory")
    from core.memory import memory_db

    try:
        mem_count = memory_db.collection.count()
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

# Tab layout
tab_query, tab_graph, tab_benchmark = st.tabs(
    ["Run Query", "Workflow Graph", "Benchmark"]
)

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

# ===== Tab 2: Workflow Graph =====
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
