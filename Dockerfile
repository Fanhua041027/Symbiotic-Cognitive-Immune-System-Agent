FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Install deps into user site-packages for easy copy to runtime stage
RUN pip install --user --no-cache-dir -r requirements.txt chromadb

# ------------------------------------------------------------------ # Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy pre-installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application code
COPY . .

# Create non-root user (avoid Docker warning)
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

CMD ["python", "immune_agent.py"]
