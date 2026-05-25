# ============================================================================
# Symbiotic Cognitive Immune System Agent — Multi-stage production Dockerfile
# ============================================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

# Install all dependencies (including optional groups)
RUN pip install --no-cache-dir \
    -r requirements.txt \
    chromadb streamlit fastapi uvicorn pydantic \
    && pip install --no-cache-dir --no-deps requests

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

# Security hardening: non-root user
RUN groupadd -r agent && useradd -r -g agent -m -d /home/agent agent

# Copy pre-installed site-packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1

# Copy application code
COPY --chown=agent:agent . .

# Create required directories with correct permissions
RUN mkdir -p /app/logs /app/.immune_db /app/escalations /app/metrics /app/sessions \
    /app/benchmarks /app/reports /app/trainer /app/exports \
    /home/agent/.cache/chroma \
    && chown -R agent:agent /app /home/agent

# Switch to non-root user
USER agent

# Default command
CMD ["python", "immune_agent.py"]
