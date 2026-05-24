.PHONY: install run demo interactive query test adversarial benchmark stats graph webui \
        lint precommit clean docker-build docker-run docker-interactive

# Install dependencies
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt pytest pre-commit ruff mypy streamlit chromadb

# Run demo
run demo:
	python immune_agent.py

# Interactive mode
interactive:
	python immune_agent.py --interactive

# Custom query (usage: make query Q="your question")
query:
	python immune_agent.py --query "$(Q)"

# System statistics
stats:
	python immune_agent.py --stats

# Workflow graph
graph:
	python immune_agent.py --graph

# Streamlit Web UI
webui:
	streamlit run app.py

# Run tests
test:
	python -m pytest tests/ -v --tb=short

# Run adversarial benchmark
adversarial benchmark:
	python immune_agent.py --benchmark

# Code quality
lint:
	python -m py_compile core/*.py immune_agent.py app.py api.py tests/*.py
	@echo "All files compile OK"

precommit:
	pip install pre-commit
	pre-commit install
	pre-commit run --all-files

# Run API server (requires fastapi + uvicorn)
api:
	python api.py

# Clean caches and logs
clean:
	rm -rf __pycache__ core/__pycache__ tests/__pycache__
	rm -rf logs/ .immune_db/ escalations/ benchmarks/ metrics/ sessions/
	rm -f *.pyc core/*.pyc tests/*.pyc

# Docker
docker-build:
	docker compose build

docker-run:
	docker compose run --rm immune-agent

docker-interactive:
	docker compose run --rm immune-agent-interactive
