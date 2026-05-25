# Changelog

## 1.2.0 (2026-05-25)

- **Active adversarial training** (`core/adversarial_trainer.py`):
  - Attacker Agent generates adversarial queries via LLM (10 anomaly categories)
  - Automated training loop with epochs, scoring, and progress tracking
  - Training reports saved to `trainer/` directory
  - CLI: `python immune_agent.py --train --epochs 3 --queries-per-epoch 5`
  - Web UI: Training tab with progress bar and report browser
  - REST API: `POST /train`, `GET /training/reports`

- **Notification system** (`core/notifications.py`):
  - Slack webhook integration for escalation alerts and benchmark reports
  - Generic webhook support for JSON payloads
  - Auto-notify on escalation events (integrated with `core/escalation.py`)
  - Config: `SLACK_WEBHOOK_URL`, `NOTIFICATION_WEBHOOK_URL`

- **Antibody export/import** (`core/memory.py`):
  - Export all antibodies to JSON: `python immune_agent.py --export-antibodies`
  - Import antibodies from JSON: `python immune_agent.py --import-antibodies path/to/export.json`
  - Cross-instance immune memory sharing
  - REST API: `POST /memory/export`, `POST /memory/import`

- **HTML report generator** (`core/reports.py`):
  - Self-contained HTML reports with CSS styling and bar charts
  - Benchmark, training, and metrics report templates
  - CLI: `python immune_agent.py --report`
  - REST API: `POST /reports/generate`

- **API rate limiting enhancement** (`core/ratelimit.py`):
  - Token bucket with remaining() and reset() methods
  - FastAPI middleware integration with 429 responses
  - Config: `RATE_LIMIT_REQUESTS=30`, `RATE_LIMIT_WINDOW=60`

- **Production Docker Compose**:
  - Multi-stage Dockerfile with non-root user and security hardening
  - Named volumes for all data directories
  - 5 service profiles: core, interactive, api, web, daemon
  - Healthchecks on API and Web services
  - Isolated bridge network
  - Profiles for selective service startup

- 283 unit tests passing (100%), 0 warnings
- 22 adversarial test cases across 10 anomaly categories

## 1.1.0

- Thread-safe lazy LLM initialization with cache lock
- E2B cloud sandbox validation backend
- Config hot-reload from Streamlit UI
- Antibody deduplication via Jaccard Token similarity
- Git auto-backup on immune response
- Streamlit UI: status panel, popover, toast, confirmation dialogs
- FastAPI REST endpoints: health, query, stats, memory CRUD, config, demo
- Metrics system with deque-based sliding window and p95 latency
- 148+ unit tests across all core modules
- Fixed CI YAML indentation in `python -c` strings
- Fixed `should_continue` routing function (extracted escalation side effects)

## 1.0.0

- Initial release
- Core LangGraph workflow: Worker → Monitor → Antibody Generator → Sandbox → retry
- Multi-level sandbox: simulated, AST, Docker
- ChromaDB persistent immune memory with automatic in-memory fallback
- Multi-LLM provider support: OpenAI, DeepSeek, custom endpoints
- CLI, Streamlit Web UI, and REST API interfaces
- Adversarial benchmark suite (12 test cases)
- Escalation system with JSON failure reports
- Workflow visualization (Mermaid + ASCII)
