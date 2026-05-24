FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Install deps globally (not --user) so runtime stage can use them directly
RUN pip install --no-cache-dir -r requirements.txt chromadb

# ------------------------------------------------------------------ # Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy pre-installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Install API and Web UI dependencies
RUN pip install --no-cache-dir fastapi uvicorn streamlit

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application code
COPY . .

# Create non-root user (avoid Docker warning)
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

CMD ["python", "immune_agent.py"]
