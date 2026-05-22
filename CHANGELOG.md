# Changelog

## 1.1.0 (latest)

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
