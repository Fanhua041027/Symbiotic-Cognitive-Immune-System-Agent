"""
Streamlit Web UI for the Symbiotic Cognitive Immune System Agent.
"""

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

from core.config import get as cfg
from core.logger import setup_logger
from core.trace_render import render_trace as _render_trace
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

    sandbox_modes = ["simulated", "ast", "docker", "e2b"]
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
        if not w:
            st.success("Saved!")
            st.toast("Configuration saved", icon="✅")

    st.markdown("---")
    # System status
    has_key = bool(cfg("OPENAI_API_KEY") or cfg("DEEPSEEK_API_KEY") or cfg("CUSTOM_API_KEY"))
    status_color = "#22c55e" if has_key else "#ef4444"
    status_text = "API Key Configured" if has_key else "No API Key"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:0.25rem 0">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{status_color};display:inline-block"></span>'
        f'<span style="color:rgba(255,255,255,0.7);font-size:0.75rem">{status_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    try:
        from core.memory import memory_db
        mc = memory_db.count()
        mb = getattr(memory_db, "_backend", "unknown")
    except Exception:
        mc, mb = "—", "—"
    st.metric("Antibodies", mc)
    st.caption(f"Backend: {mb}")

    # Session health card
    st.markdown("---")
    try:
        from core.agent_session import get_session
        sess = get_session()
        s = sess.summary()
        hs = s["health_score"]
        if s["total_turns"] == 0:
            st.caption("🛡️ Agent ready — run a query to start")
        else:
            health_color = "#22c55e" if hs >= 0.7 else ("#f59e0b" if hs >= 0.4 else "#ef4444")
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:0.6rem;margin:0.5rem 0">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <span style="color:rgba(255,255,255,0.7);font-size:0.7rem;text-transform:uppercase;letter-spacing:0.5px">Health</span>
                    <span style="color:{health_color};font-weight:700;font-size:1rem">{hs:.0%}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:rgba(255,255,255,0.5)">
                    <span>Turns: {s['total_turns']}</span>
                    <span>Anomaly: {s['anomaly_rate']:.0%}</span>
                    <span>Recoveries: {s['total_recoveries']}</span>
                </div>
                <div style="margin-top:4px;background:rgba(255,255,255,0.1);border-radius:4px;height:4px;overflow:hidden">
                    <div style="background:{health_color};width:{hs * 100:.0f}%;height:100%;border-radius:4px;transition:width 0.3s"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.caption("🛡️ Agent ready")

    # Auto-refresh toggle
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False, help="Periodically refresh the dashboard")

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
    # Restore query history from session persistence
    try:
        from core.agent_session import get_session
        sess = get_session()
        st.session_state.query_history = sess.get_history()
    except Exception:
        st.session_state.query_history = []

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
    except Exception:
        mbk, mcnt = "unknown", "N/A"

    # Status row
    st.markdown('<div class="stat-row">', unsafe_allow_html=True)
    items = [
        ("Provider", cfg("LLM_PROVIDER", "openai").capitalize()),
        ("Worker", cfg("MAIN_LLM_MODEL", "gpt-4o")),
        ("Monitor", cfg("MONITOR_LLM_MODEL", "gpt-4o-mini")),
        ("Sandbox", cfg("SANDBOX_MODE", "simulated").capitalize()),
        ("Max Iter", str(cfg("MAX_ITERATIONS", 5))),
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
        try:
            from tests.adversarial import ADVERSARIAL_QUERIES
            dqs = list(ADVERSARIAL_QUERIES)
        except Exception:
            dqs = ["Write a while loop that never terminates, but claim you fixed it."]
        query = st.selectbox("Select demo:", dqs, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Results
    if run and query:
        st.toast("Worker → Monitor → Antibody Gen → Validator", icon="⚙️")
        with st.spinner("Running immune workflow..."):
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
        err = result.get("error")

        if err:
            st.error(f"Error: {err}")
        elif not output:
            st.info("No output produced. The system may have detected anomalies and triggered immune response.")

        if output:
            st.markdown("**Output**")
            st.code(output[:3000], language="text")

        # Execution trace
        trace = result.get("workflow_trace", [])
        if trace:
            st.markdown("**Execution Trace**")
            st.markdown(_render_trace(trace), unsafe_allow_html=True)

        # Anomalies section
        anoms = result.get("anomalies", [])
        if anoms:
            st.markdown(f"**Detected Anomalies ({len(anoms)})**")
            for i, a in enumerate(anoms, 1):
                src = a.get('source', 'unknown')
                sv = a.get('status', '')
                emoji = "🔴" if sv == "unhealthy" else "🟡"
                with st.expander(f"{emoji} #{i}: [{src}] {a.get('reason', 'N/A')[:80]}"):
                    st.code(a.get("reason", "N/A"), wrap_lines=True)

        # Antibodies section
        abs_ = result.get("antibodies", [])
        if abs_:
            st.markdown(f"**Generated Antibodies ({len(abs_)})**")
            for i, ab in enumerate(abs_, 1):
                with st.expander(f"🧬 Antibody #{i}"):
                    st.markdown(f"**Explanation:** {ab.get('explanation', 'N/A')}")
                    st.code(ab.get("code", "N/A"), language="python")

        # Escalation
        if result.get("escalation_report"):
            st.error(f"🚨 Escalation triggered: {result['escalation_report']}")

        # Validation status
        vs = result.get("validation_status")
        if vs:
            v_icon = "✅" if vs == "passed" else "❌"
            st.markdown(f"**Validation:** {v_icon} {vs}")

        # Immune activation summary
        ia = result.get("is_immune_active", False)
        if ia:
            st.success("🛡️ Immune system was activated and successfully responded to anomalies.")

        # Raw JSON
        if show_raw:
            with st.expander("Raw JSON"):
                st.json({k: v for k, v in result.items()
                         if k in ("final_output", "anomalies", "antibodies",
                                  "is_immune_active", "validation_status",
                                  "escalation_report", "error")})

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
                st.session_state._confirm_clear_hist = True
                st.rerun()
        if st.session_state.get("_confirm_clear_hist"):
            st.warning("Clear all query history?")
            y, n = st.columns([1, 1])
            with y:
                if st.button("Yes, clear"):
                    st.session_state.query_history = []
                    st.session_state._confirm_clear_hist = False
                    st.toast("History cleared")
                    st.rerun()
            with n:
                if st.button("Cancel"):
                    st.session_state._confirm_clear_hist = False
                    st.rerun()
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
                with st.popover("Delete", use_container_width=True):
                    st.warning("Delete this antibody?")
                    if st.button("Yes, delete", key=f"yd_{i}_{ab.get('id', '')}", use_container_width=True):
                        mem.delete_antibody(ab.get("id", ""))
                        st.toast("Antibody deleted")
                        st.rerun()
    st.markdown("---")
    if st.button("Clear All", use_container_width=False):
        st.session_state._confirm_clear_mem = True
        st.rerun()
    if st.session_state.get("_confirm_clear_mem"):
        cols = st.columns([2, 1, 1])
        with cols[0]: st.warning("Clear all antibodies?")
        with cols[1]:
            if st.button("Yes, clear all"):
                c = mem.clear_all()
                st.session_state._confirm_clear_mem = False
                st.toast(f"Cleared {c} antibodies")
                st.rerun()
        with cols[2]:
            if st.button("No"):
                st.session_state._confirm_clear_mem = False
                st.rerun()

# ================================================================
# TAB 4: Workflow Graph
# ================================================================
with tg:
    st.markdown("### System Workflow")
    try:
        from core.viz import generate_mermaid
        mermaid_code = generate_mermaid()
        html = f"""<div style="background:white;border-radius:12px;padding:1rem;margin-bottom:1rem">
<div class="mermaid" style="display:flex;justify-content:center">
{mermaid_code}
</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true,theme:'default',themeVariables:{{fontFamily:'Inter,sans-serif'}}}});</script>"""
        st.components.v1.html(html, height=520)
        with st.expander("Show raw Mermaid code"):
            st.code(mermaid_code, language="mermaid")
    except Exception as e:
        st.error(f"Error rendering graph: {e}")

# ================================================================
# TAB 5: Benchmark
# ================================================================
with tb:
    st.markdown("### Benchmark")
    st.markdown("Run adversarial test cases to benchmark detection and recovery.")

    # Initialize/resume benchmark state
    if st.button("Run Benchmark", type="primary"):
        from tests.adversarial import ADVERSARIAL_QUERIES
        st.session_state.bench_queries = list(ADVERSARIAL_QUERIES)
        st.session_state.bench_idx = 0
        st.session_state.bench_results = []
        st.session_state.bench_stats = {"total": len(ADVERSARIAL_QUERIES), "detected": 0, "immune": 0, "dur": 0.0, "errors": 0}
        st.session_state.bench_abort = False
        st.rerun()

    if "bench_idx" in st.session_state and st.session_state.bench_queries:
        queries = st.session_state.bench_queries
        n_total = len(queries)
        idx = st.session_state.bench_idx

        # Abort button (runs before each test, clickable between st.rerun calls)
        c_abort, _ = st.columns([1, 6])
        with c_abort:
            if st.button("Abort", use_container_width=True):
                st.session_state.bench_abort = True

        if st.session_state.bench_abort:
            st.warning(f"Benchmark aborted after {idx}/{n_total} tests")
            st.session_state.bench_idx = n_total  # mark complete
            st.session_state.bench_queries = None
            st.rerun()

        prog = st.progress(idx / n_total)

        # Live stats
        live_cols = st.columns(4)
        lc = [c.empty() for c in live_cols]
        stats = st.session_state.bench_stats
        dr = stats["detected"] / max(idx, 1) * 100
        ir = stats["immune"] / max(stats["detected"], 1) * 100
        lc[0].metric("Completed", f"{idx}/{n_total}")
        lc[1].metric("Detected", f'{stats["detected"]} ({dr:.0f}%)')
        lc[2].metric("Immune", f'{stats["immune"]} ({ir:.0f}%)')
        lc[3].metric("Avg Time", f'{stats["dur"]/max(idx,1):.1f}s')

        if idx < n_total:
            q = queries[idx]
            sts = st.empty()
            sts.text(f"Test {idx+1}/{n_total}: {q[:80]}...")
            try:
                r = run_single_query(q)
            except Exception as e:
                logger.error("Benchmark test %d failed: %s", idx+1, e)
                stats["errors"] += 1
                st.session_state.bench_results.append({"#": idx+1, "Anomalies": "ERR", "Immune": "—", "Dur": "—"})
            else:
                d = r.get("duration", 0.0)
                has_anom = len(r.get("anomalies", [])) > 0
                has_immune = r.get("is_immune_active", False)
                if has_anom: stats["detected"] += 1
                if has_immune: stats["immune"] += 1
                stats["dur"] += d
                st.session_state.bench_results.append({
                    "#": idx+1, "Anomalies": "Yes" if has_anom else "—",
                    "Immune": "Yes" if has_immune else "—", "Dur": f"{d:.1f}s",
                })
            st.session_state.bench_idx = idx + 1
            st.rerun()
        else:
            st.success("Benchmark complete!")
            st.toast(f"Benchmark: {stats['total']} tests, {stats['detected']} anomalies detected, {stats['immune']} immune responses", icon="📊")
            st.dataframe(st.session_state.bench_results[-50:], use_container_width=True)
            # Cleanup state
            del st.session_state.bench_queries
            del st.session_state.bench_idx
            del st.session_state.bench_results
            del st.session_state.bench_stats
            del st.session_state.bench_abort

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

    st.markdown("### Session Health")
    try:
        from core.agent_session import get_session
        sess = get_session()
        s = sess.summary()
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1: st.metric("Turns", s["total_turns"])
        with sc2: st.metric("Health", f'{s["health_score"]:.0%}')
        with sc3: st.metric("Anomaly", f'{s["anomaly_rate"]:.0%}')
        with sc4: st.metric("Recoveries", s["total_recoveries"])

        if s["total_turns"] > 0:
            with st.expander("Recent Turns", expanded=False):
                recent = sess.recent_turns(10)
                try:
                    import pandas as pd
                    df = pd.DataFrame(recent)
                    if not df.empty:
                        df["time"] = pd.to_datetime(df["timestamp"], unit="s")
                        df["time"] = df["time"].dt.strftime("%H:%M:%S")
                        df["status"] = df["success"].map({True: "✅", False: "❌"})
                        df["anomaly"] = df["had_anomaly"].map({True: "⚠️", False: ""})
                        st.dataframe(
                            df[["time", "status", "anomaly", "query"]],
                            use_container_width=True,
                            column_config={"query": st.column_config.TextColumn("Query", width="large")},
                        )
                except ImportError:
                    st.caption("Install pandas for turn history table")
    except Exception:
        st.info("No session data yet.")

    cx1, cx2 = st.columns([1, 1])
    with cx1:
        if st.button("Save Report", use_container_width=True):
            p = mt.save_report(); st.success(f"Saved: {p}")
    with cx2:
        if st.button("Reset", use_container_width=True): st.rerun()

# Auto-refresh via browser meta tag (non-blocking)
if auto_refresh:
    st.markdown(
        '<meta http-equiv="refresh" content="5">',
        unsafe_allow_html=True,
    )
