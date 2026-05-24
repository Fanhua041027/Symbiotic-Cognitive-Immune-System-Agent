"""Trace rendering for the workflow visualization.

Shared between the Streamlit UI and test code.
"""

TRACE_STYLES = {
    "enter:worker": ("Worker", "#1a73e8"),
    "enter:consistency_check": ("Consistency", "#f59e0b"),
    "enter:monitor": ("Monitor", "#7c3aed"),
    "enter:generate_antibody": ("Antibody Gen", "#0d9488"),
    "enter:validate_antibody": ("Validator", "#6b7280"),
}


def render_trace(trace: list[str]) -> str:
    """Render an execution trace as an HTML span chain."""
    tags = []
    for entry in trace:
        label, color = TRACE_STYLES.get(entry, (entry, "#6b7280"))
        tags.append(
            f'<span style="background:{color};color:#fff;padding:2px 10px;'
            f'border-radius:6px;font-size:0.8rem;white-space:nowrap;'
            f'font-weight:500">{label}</span>',
        )
    arrow = '<span style="color:#9ca3af;padding:0 3px;font-size:1rem">→</span>'
    container = (
        f'<div style="display:flex;flex-wrap:wrap;gap:4px;'
        f'align-items:center;padding:0.4rem 0">{arrow.join(tags)}</div>'
    )
    return container
