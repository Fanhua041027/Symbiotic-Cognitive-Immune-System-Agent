FROM python:3.11-slim

WORKDIR /app

# Install system deps for chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install core + optional memory dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt chromadb

# Copy project files
COPY . .

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Default: run demo
CMD ["python", "immune_agent.py"]
