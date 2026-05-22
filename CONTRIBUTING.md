# Contributing

Thanks for your interest in the Symbiotic Cognitive Immune System Agent.

## How to Contribute

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/your-feature-name`
3. Make your changes.
4. Run tests: `python -m pytest tests/` (no API key needed for unit tests)
5. Commit with a descriptive message.
6. Push and open a pull request.

## Development Setup

```bash
pip install -r requirements.txt chromadb streamlit fastapi uvicorn pytest pre-commit
pre-commit install
```

## Code Style

- Line length: 90 (enforced by Ruff)
- Follow existing patterns in `core/` modules
- Type hints required for all public functions
- Log via `setup_logger(__name__)` instead of `print()`

## Testing

- Add tests under `tests/` following the existing patterns
- All tests must pass before merge
- New features should include corresponding tests

## Project Structure

- `core/` — framework modules (state, nodes, workflow, sandbox, etc.)
- `tests/` — unit tests and adversarial benchmarks
- `app.py` — Streamlit web UI
- `api.py` — FastAPI REST API
- `immune_agent.py` — CLI entry point

## Questions

Open an issue or reach out via the repository's discussion board.
