.PHONY: install run demo interactive test adversarial clean lint precommit docker-build docker-run docker-interactive

# Install dependencies
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt pytest pre-commit ruff mypy

# Run demo
run demo:
	python immune_agent.py

# Interactive mode
interactive:
	python immune_agent.py --interactive

# Custom query
query:
	python immune_agent.py --query "$(Q)"

# Run tests
test:
	python -m pytest tests/ -v --tb=short

# Run adversarial benchmark
adversarial:
	python tests/adversarial.py

# Code quality
lint:
	python -m py_compile core/*.py immune_agent.py tests/*.py
	@echo "All files compile OK"

precommit:
	pip install pre-commit
	pre-commit install
	pre-commit run --all-files

# Clean caches and logs
clean:
	rm -rf __pycache__ core/__pycache__ tests/__pycache__
	rm -rf logs/ .immune_db/ escalations/ benchmarks/
	rm -f *.pyc core/*.pyc tests/*.pyc

# Docker
docker-build:
	docker compose build

docker-run:
	docker compose run --rm immune-agent

docker-interactive:
	docker compose run --rm immune-agent-interactive
