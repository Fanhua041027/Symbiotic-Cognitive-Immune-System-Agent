"""Workflow graph visualization utilities."""

from core.logger import setup_logger

logger = setup_logger("viz")


def generate_mermaid() -> str:
    """Generate a Mermaid flowchart of the immune system workflow."""
    return """flowchart TD
    START([User Input]) --> Worker[Worker Agent]

    Worker --> Monitor[Monitor T-Cell]

    Monitor -->|healthy + output| END([Final Output])
    Monitor -->|anomaly detected| Antibody[Antibody Generator]
    Monitor -->|no output yet| Worker

    Antibody --> Validator[Sandbox Validator]
    Validator -->|passed / failed| Worker

    Worker -.->|max iterations| Escalation[Human Escalation]
    Escalation -.-> END

    subgraph Legend
        Direction[ ]
        Agent[Agent Node] -.-> StateChange[State Change]
    end

    style START fill:#4CAF50,color:#fff
    style END fill:#2196F3,color:#fff
    style Escalation fill:#f44336,color:#fff
    style Worker fill:#FF9800,color:#fff
    style Monitor fill:#9C27B0,color:#fff
    style Antibody fill:#00BCD4,color:#fff
    style Validator fill:#607D8B,color:#fff
"""


def print_graph() -> None:
    """Print the Mermaid workflow graph to console."""
    graph = generate_mermaid()
    print("\n" + "=" * 60)
    print("Immune System Workflow Graph (Mermaid JS)")
    print("=" * 60)
    print("Copy the following into https://mermaid.live to visualize:\n")
    print(graph)


def print_graph_ascii() -> None:
    """Print an ASCII-art version of the workflow."""
    ascii_art = r"""
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
"""
    print(ascii_art)
