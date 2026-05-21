"""
Streamlit Web UI for the Symbiotic Cognitive Immune System Agent.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

from core.logger import setup_logger
from core.config import get as cfg
from immune_agent import run_single_query

logger = setup_logger("webui")

st.set_page_config(page_title="Immune System Agent", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Modern Dynamic CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', -apple-system, sans-serif; }
    .main { background: #f0f2f5; }
    .block-container { padding: 1rem 1.5rem !important; max-width: 1400px; }

    h1 { font-weight: 800; font-size: 1.6rem; letter-spacing: -0.03em; color: #0f1419; margin: 0 !important; }
    h2 { font-weight: 700; font-size: 1.2rem; color: #0f1419; margin: 0 0 0.5rem 0 !important; }
    h3 { font-weight: 600; font-size: 1rem; color: #0f1419; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1d23 100%);
        border-right: none;
    }
    section[data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem !important; }
    section[data-testid="stSidebar"] .st-emotion-cache-1mi2ry5 { background: transparent; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #ffffff; }
    section[data-testid="stSidebar"] label { color: rgba(255,255,255,0.6) !important; font-size: 0.75rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.04em; }

    /* Sidebar card sections */
    .sidebar-section {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .sidebar-section label { color: rgba(255,255,255,0.5) !important; font-size: 0.65rem !important; }

    /* Hero Card */
    .hero-card {
        background: linear-gradient(135deg, #0f1419 0%, #1a1d23 100%);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-card h1 { color: white; font-size: 1.4rem; }
    .hero-card p { color: rgba(255,255,255,0.6); font-size: 0.85rem; margin: 0.2rem 0 0 0; }

    /* Stat cards row */
    .stat-row { display: flex; gap: 0.6rem; margin-bottom: 1rem; flex-wrap: wrap; }
    .stat-card {
        flex: 1; min-width: 140px;
        background: white; border-radius: 12px;
        padding: 0.8rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border: 1px solid #e8eaed;
    }
    .stat-card .label { font-size: 0.7rem; font-weight: 600; color: #8e98a3; text-transform: uppercase; letter-spacing: 0.04em; }
    .stat-card .value { font-size: 1.3rem; font-weight: 700; color: #0f1419; letter-spacing: -0.02em; margin-top: 0.15rem; }

    /* Metric overrides */
    [data-testid="stMetric"] { background: transparent; border: none; box-shadow: none; padding: 0 !important; }
    [data-testid="stMetric"] label { font-size: 0.7rem !important; font-weight: 600 !important; color: #8e98a3 !important; text-transform: uppercase; letter-spacing: 0.04em; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 700 !important; color: #0f1419 !important; letter-spacing: -0.02em; }

    .stat-card [data-testid="stMetric"] label { color: #8e98a3 !important; }

    /* Cards */
    .card {
        background: white; border-radius: 12px; padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04); border: 1px solid #e8eaed;
        margin-bottom: 0.8rem;
    }

    /* Tags / Chips */
    .tag {
        display: inline-block; background: #f0f2f5; border-radius: 6px;
        padding: 0.25rem 0.7rem; font-size: 0.75rem; font-weight: 500;
        color: #536471; margin: 0.15rem;
    }
    .tag-active { background: #e8f0fe; color: #1a73e8; }

    /* Buttons */
    .stButton button {
        border-radius: 8px !important; font-weight: 600 !important;
        font-size: 0.85rem !important; border: none !important;
        padding: 0.4rem 1rem !important; transition: all 0.15s ease !important;
    }
    .stButton button[kind="primary"] { background: #1a73e8 !important; color: white !important; }
    .stButton button[kind="primary"]:hover { background: #1557b0 !important; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(26,115,232,0.3) !important; }
    .stButton button[kind="secondary"] { background: #e8eaed !important; color: #0f1419 !important; }
    .stButton button[kind="secondary"]:hover { background: #d2d5d9 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: white; border-radius: 10px; padding: 4px;
        gap: 2px; border: 1px solid #e8eaed;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px; padding: 0.35rem 1rem; font-size: 0.8rem;
        font-weight: 500; color: #536471;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #0f1419; color: white; }

    /* Inputs */
    .stTextInput input, .stTextArea textarea {
        border-radius: 8px !important; border: 1px solid #dadce0 !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #1a73e8 !important; box-shadow: 0 0 0 3px rgba(26,115,232,0.12) !important;
    }
    .stSelectbox > div > div { border-radius: 8px !important; border-color: #dadce0 !important; }

    /* Progress */
    .stProgress > div > div { border-radius: 100px !important; background: #e8eaed !important; }
    .stProgress > div > div > div { background: #1a73e8 !important; border-radius: 100px !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden; border: 1px solid #e8eaed; }

    /* Expander */
    .stExpander { border-radius: 8px !important; border: 1px solid #e8eaed !important; margin-bottom: 0.35rem; }
    .stExpander summary { font-weight: 500; }

    /* Alerts */
    .stAlert { border-radius: 8px !important; border: none !important; }

    /* Divider */
    hr { border-color: #e8eaed !important; margin: 0.8rem 0 !important; }

    /* Caption */
    .stCaption { color: #8e98a3 !important; }

    /* Section badge */
    .section-badge {
        display: inline-block; font-size: 0.65rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.06em;
        color: #8e98a3; margin-bottom: 0.4rem;
    }

    /* Smooth scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #dadce0; border-radius: 3px; }

    /* Sidebar white text for selects */
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] .stSlider label { color: rgba(255,255,255,0.5) !important; }
    section[data-testid="stSidebar"] .st-emotion-cache-1wivap2 { color: white !important; }
    section[data-testid="stSidebar"] .st-emotion-cache-1fhd7pv { color: white !important; }

    /* Sidebar button override */
    section[data-testid="stSidebar"] .stButton button { font-size: 0.8rem !important; padding: 0.35rem 0.8rem !important; }

    /* Code blocks */
    .stCode { border-radius: 8px !important; }
    code { font-size: 0.85em !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:0.5rem 0 1rem 0"><span style="font-size:2rem">🛡️</span><h2 style="color:white;margin:0.3rem 0 0 0">Immune Agent</h2><p style="color:rgba(255,255,255,0.4);font-size:0.75rem;margin:0">Multi-Agent Defense Framework</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<label>Provider</label>', unsafe_allow_html=True)
    providers = ["openai", "deepseek", "custom"]
    current = cfg("LLM_PROVIDER", "openai")
    provider = st.selectbox("provider", options=providers, index=providers.index(current) if current in providers else 0, label_visibility="collapsed")

    sandbox_modes = ["simulated", "ast", "docker"]
    cur_sb = cfg("SANDBOX_MODE", "simulated")
    sandbox_mode = st.selectbox("sandbox", options=sandbox_modes, index=sandbox_modes.index(cur_sb) if cur_sb in sandbox_modes else 0, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<label>Model</label>', unsafe_allow_html=True)
    worker_model = st.text_input("worker", value=cfg("MAIN_LLM_MODEL", "gpt-4o"), label_visibility="collapsed")
    monitor_model = st.text_input("monitor", value=cfg("MONITOR_LLM_MODEL", "gpt-4o-mini"), label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<label>Limits</label>', unsafe_allow_html=True)
    max_iter = st.slider("Max Iterations", 1, 20, value=cfg("MAX_ITERATIONS", 5), label_visibility="collapsed")
    temperature = st.slider("Temperature", 0.0, 2.0, value=cfg("LLM_TEMPERATURE", 0.7), step=0.1, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    save_btn = st.button("Save Config", type="primary", use_container_width=True)
    if save_btn:
        from core.config import save_config as _save_cfg
        ups = {"LLM_PROVIDER": provider, "SANDBOX_MODE": sandbox_mode, "MAX_ITERATIONS": str(max_iter),
               "MAIN_LLM_MODEL": worker_model, "MONITOR_LLM_MODEL": monitor_model, "LLM_TEMPERATURE": str(temperature)}
        w = _save_cfg(ups)
        for x in w: st.warning(x)
        if not w: st.success("Saved!")

    st.markdown("---")
    try:
        from core.memory import memory_db
        mc = memory_db.count()
        mb = getattr(memory_db, "_backend", "unknown")
    except Exception:
        mc, mb = "—", "—"
    st.metric("Antibodies", mc)
    st.caption(f"Backend: {mb}")

# ---------------------------------------------------------------------------
# Main: Hero + Tabs
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div>
        <h1>🛡️ Immune System Agent</h1>
        <p>Self-diagnosing · Self-healing · Self-evolving AI framework</p>
    </div>
</div>
""", unsafe_allow_html=True)

if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Must be defined before tabs that call it
TRACE_STYLES = {
    "enter:worker": ("Worker", "#1a73e8"),
    "enter:monitor": ("Monitor", "#7c3aed"),
    "enter:generate_antibody": ("Antibody Gen", "#0d9488"),
    "enter:validate_antibody": ("Validator", "#6b7280"),
}


def _render_trace(trace: list[str]) -> str:
    tags = []
    for entry in trace:
        label, color = TRACE_STYLES.get(entry, (entry, "#6b7280"))
        tags.append(
            f'<span style="background:{color};color:#fff;padding:2px 10px;'
            f'border-radius:6px;font-size:0.8rem;white-space:nowrap;'
            f'font-weight:500">{label}</span>'
        )
    arrow = '<span style="color:#9ca3af;padding:0 3px;font-size:1rem">→</span>'
    return f'<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;padding:0.4rem 0">{arrow.join(tags)}</div>'


tabs = st.tabs(["Run Query", "History", "Memory", "Workflow", "Benchmark", "Metrics"])
tq, th, tm, tg, tb, tmet = tabs

# ================================================================
# TAB 1: Run Query
# ================================================================
with tq:
    try:
        from core.memory import memory_db
        mbk = getattr(memory_db, "_backend", "unknown")
        mcnt = memory_db.count()
    except:
        mbk, mcnt = "unknown", "N/A"

    # Status row
    st.markdown('<div class="stat-row">', unsafe_allow_html=True)
    items = [
        ("Provider", cfg("LLM_PROVIDER", "openai").capitalize()),
        ("Worker", cfg("MAIN_LLM_MODEL", "gpt-4o")),
        ("Monitor", cfg("MONITOR_LLM_MODEL", "gpt-4o-mini")),
        ("Sandbox", cfg("SANDBOX_MODE", "simulated").capitalize()),
        ("Memory", mbk.capitalize()),
        ("Antibodies", str(mcnt)),
    ]
    for lab, val in items:
        st.markdown(f'<div class="stat-card"><div class="label">{lab}</div><div class="value">{val}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Query area
    st.markdown('<div class="card">', unsafe_allow_html=True)
    query = st.text_area("Query", height=90, placeholder="Write a recursive function that calls itself without a base case...", label_visibility="collapsed")
    c1, c2, c3, _ = st.columns([1, 1.5, 1, 4])
    with c1: run = st.button("▶ Run", type="primary", use_container_width=True)
    with c2: demo = st.checkbox("Use demo", value=False)
    with c3: show_raw = st.checkbox("Raw JSON", value=False)
    if demo:
        dqs = ["Write a while loop that never terminates, but claim you fixed it by adding a pass statement.",
               "Write a function that returns True if a number is both greater than 10 and less than 5.",
               "Write a recursive function to traverse a nested dictionary and print all keys. Make sure it handles infinite nesting."]
        query = st.selectbox("Select demo:", dqs, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Results
    if run and query:
        with st.spinner("Running workflow..."):
            start = time.time()
            result = run_single_query(query)
            dur = time.time() - start
            st.session_state.query_history.append({
                "query": query, "duration": f"{dur:.1f}s",
                "anomalies": len(result.get("anomalies", [])),
                "antibodies": len(result.get("antibodies", [])),
                "immune_active": result.get("is_immune_active", False),
                "validation": result.get("validation_status", "N/A"),
                "timestamp": time.strftime("%H:%M:%S"),
            })

        st.markdown("### Results")
        r1, r2, r3, r4 = st.columns(4)
        with r1: st.metric("Anomalies", len(result.get("anomalies", [])))
        with r2: st.metric("Antibodies", len(result.get("antibodies", [])))
        with r3: st.metric("Immune Active", "Yes" if result.get("is_immune_active") else "No")
        with r4: st.metric("Duration", f"{dur:.1f}s")

        output = result.get("final_output")
        if output:
            st.markdown("**Output**")
            st.code(output[:2000], language="text")
        else:
            st.info("No output produced.")

        trace = result.get("workflow_trace", [])
        if trace:
            st.markdown("**Execution Trace**")
            st.markdown(_render_trace(trace), unsafe_allow_html=True)

        err = result.get("error")
        if err:
            st.error(f"Error: {err}")

        anoms = result.get("anomalies", [])
        if anoms:
            st.markdown(f"**Anomalies ({len(anoms)})**")
            for i, a in enumerate(anoms, 1):
                with st.expander(f"#{i}: {a.get('source', 'unknown')}"):
                    st.code(a.get("reason", "N/A"), wrap_lines=True)

        abs_ = result.get("antibodies", [])
        if abs_:
            st.markdown(f"**Antibodies ({len(abs_)})**")
            for i, ab in enumerate(abs_, 1):
                with st.expander(f"Antibody #{i}"):
                    st.markdown(ab.get("explanation", "N/A"))
                    st.code(ab.get("code", "N/A"), language="python")

        if result.get("escalation_report"):
            st.error(f"Escalation: {result['escalation_report']}")

        vs = result.get("validation_status")
        if vs:
            st.markdown(f"**Validation:** {'✅' if vs == 'passed' else '❌'} {vs}")

        if show_raw:
            with st.expander("Raw JSON"):
                st.json({k: v for k, v in result.items()
                         if k in ("final_output", "anomalies", "antibodies",
                                  "is_immune_active", "validation_status", "escalation_report")})

    elif run and not query:
        st.warning("Enter a query.")

# ================================================================
# TAB 2: History
# ================================================================
with th:
    h = st.session_state.query_history
    st.markdown("### Query History")
    if not h:
        st.info("No queries yet.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total", len(h))
        with c2: st.metric("Anomalies", sum(x["anomalies"] for x in h))
        with c3: st.metric("Immune Activations", sum(1 for x in h if x["immune_active"]))
        display = list(reversed(h[-50:]))
        st.dataframe([{"Time": x["timestamp"], "Query": x["query"][:60] + ("..." if len(x["query"]) > 60 else ""),
                       "Anoms": x["anomalies"], "Abs": x["antibodies"],
                       "Immune": "Yes" if x["immune_active"] else "No", "Dur": x["duration"]} for x in display],
                     use_container_width=True)
        cc1, cc2, _ = st.columns([1, 1, 4])
        with cc1:
            if st.button("Clear", use_container_width=True):
                st.session_state.query_history = []; st.rerun()
        with cc2:
            if st.button("Export JSON", use_container_width=True):
                st.json(h[-50:])

# ================================================================
# TAB 3: Memory
# ================================================================
with tm:
    from core.memory import memory_db as mem
    st.markdown("### Immune Memory")
    cc1, cc2, cc3 = st.columns([1, 1, 3])
    with cc1:
        if st.button("Refresh", use_container_width=True): st.rerun()
    with cc3:
        search = st.text_input("Filter", placeholder="Search by pattern...", label_visibility="collapsed")

    ab_list = []
    try:
        ab_list = mem.list_antibodies(limit=200)
    except Exception as e:
        st.error(f"Error: {e}")

    if not ab_list:
        st.info("No antibodies stored yet.")
    else:
        if search:
            q = search.lower()
            ab_list = [x for x in ab_list if q in x.get("error_pattern", "").lower() or q in x.get("context", "").lower()]
        c1, c2 = st.columns(2)
        with c1: st.metric("Total", len(ab_list))
        with c2: st.metric("Backend", getattr(mem, "_backend", "unknown"))
        for i, ab in enumerate(ab_list):
            with st.expander(f"Pattern: {ab.get('error_pattern', 'unknown')[:60]}", expanded=False):
                st.markdown("**Error Pattern**"); st.code(ab.get("error_pattern", "N/A"), wrap_lines=True)
                st.markdown("**Code**"); st.code(ab.get("code", "N/A"), language="python", wrap_lines=True)
                st.markdown("**Context**"); st.markdown(ab.get("context", "N/A")[:500])
                st.caption(f"ID: {ab.get('id', 'unknown')}")
                if st.button("Delete", key=f"d_{i}_{ab.get('id', '')}", use_container_width=True):
                    if mem.delete_antibody(ab.get("id", "")): st.success("Deleted!"); st.rerun()
    st.markdown("---")
    if st.button("Clear All", use_container_width=False):
        c = mem.clear_all(); st.success(f"Cleared {c} antibodies."); st.rerun()

# ================================================================
# TAB 4: Workflow Graph
# ================================================================
with tg:
    st.markdown("### System Workflow")
    try:
        from core.viz import generate_mermaid
        st.code(generate_mermaid(), language="mermaid")
        st.markdown("Copy into [mermaid.live](https://mermaid.live)")
    except Exception as e:
        st.error(f"Error: {e}")
    st.markdown("**Architecture**")
    st.code(r"""
 User Input → [Worker] → [Monitor T-Cell]
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
           Healthy      Anomaly        Continue
              │             │
              │    [Antibody Generator]
              │             │
              │    [Sandbox Validator]
              │          │       │
              │       Passed   Failed
              │          │       │
              │          └───┬───┘
              │              ▼
              │         [Worker] (retry)
              │              │
              │         [Escalation] (≥N fails)
              └──────────────┤
                             ▼
                          [END]
""", language="text")

# ================================================================
# TAB 5: Benchmark
# ================================================================
with tb:
    st.markdown("### Benchmark")
    st.markdown("Run 12 adversarial test cases to benchmark detection and recovery.")
    if st.button("Run Benchmark", type="primary"):
        from tests.adversarial import ADVERSARIAL_QUERIES
        prog = st.progress(0)
        sts = st.empty()
        results = []
        stats = {"total": len(ADVERSARIAL_QUERIES), "detected": 0, "immune": 0, "dur": 0.0}
        for i, q in enumerate(ADVERSARIAL_QUERIES, 1):
            sts.text(f"Test {i}/{len(ADVERSARIAL_QUERIES)}")
            start = time.time()
            r = run_single_query(q)
            d = time.time() - start
            if len(r.get("anomalies", [])) > 0: stats["detected"] += 1
            if r.get("is_immune_active"): stats["immune"] += 1
            stats["dur"] += d
            results.append({"#": i, "Anomalies": len(r.get("anomalies", [])) > 0, "Immune": r.get("is_immune_active"), "Dur": f"{d:.1f}s"})
            prog.progress(i / len(ADVERSARIAL_QUERIES))
        sts.text("Done!")
        dr = stats["detected"] / stats["total"] * 100
        ir = stats["immune"] / stats["detected"] * 100 if stats["detected"] > 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Tests", stats["total"])
        with c2: st.metric("Detected", f'{stats["detected"]} ({dr:.0f}%)')
        with c3: st.metric("Immune", f'{stats["immune"]} ({ir:.0f}%)')
        with c4: st.metric("Duration", f'{stats["dur"]:.1f}s')
        st.dataframe(results, use_container_width=True)

# ================================================================
# TAB 6: Metrics
# ================================================================
with tmet:
    from core.metrics import metrics as mt
    st.markdown("### System Metrics")
    s = mt.get_summary()
    if s.get("status") == "no_data":
        st.info("No data yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Queries", s["records"])
        with c2: st.metric("Success", f'{s["success_rate"]}%')
        with c3: st.metric("Anomaly", f'{s["anomaly_rate"]}%')
        with c4: st.metric("Immune", f'{s["immune_activation_rate"]}%')
        lat = s.get("latency", {})
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Avg Latency", f'{lat.get("avg_seconds", 0):.2f}s')
        with c2: st.metric("P95", f'{lat.get("p95_seconds", 0):.2f}s')
        with c3: st.metric("Max", f'{lat.get("max_seconds", 0):.2f}s')
        bd = s.get("anomaly_breakdown", {})
        if bd:
            st.markdown("**Anomaly Sources**"); st.bar_chart(bd)
        st.code(f"Session: {s['session_duration_seconds']:.0f}s · LLM: {s['total_llm_time_seconds']:.1f}s · Escalation: {s['escalation_rate']}% · Avg Abs: {s['avg_antibodies_per_query']:.2f}", language="text")

    cx1, cx2 = st.columns([1, 1])
    with cx1:
        if st.button("Save Report", use_container_width=True):
            p = mt.save_report(); st.success(f"Saved: {p}")
    with cx2:
        if st.button("Reset", use_container_width=True): st.rerun()
