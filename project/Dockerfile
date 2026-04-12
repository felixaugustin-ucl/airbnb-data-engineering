# ── NYC Airbnb Polyglot Analytics Platform — MCP Server + Agent ───────────────
#
# Base: python:3.12-slim
# Reason: PyTorch has no wheel for Python 3.13. All pipeline scripts that use
#         sentence-transformers (build_vector_index.py, mcp_server/server.py)
#         require Python ≤ 3.12. This container unifies the environment so the
#         agent layer runs on the same Python version as the vector pipeline.
#
# What this container runs:
#   mcp_server/server.py  — MCP tool server (stdio transport)
#   agent/agent.py        — LangChain + Ollama ReAct agent
#
# What it does NOT run:
#   Ingestion scripts       — data already fetched, stored in raw/
#   Transformation scripts  — data already transformed, stored in data/ and
#                             processed/. These are mounted read-only.
#   PostgreSQL / MongoDB    — separate services in docker-compose.yml
#   Ollama                  — separate service in docker-compose.yml
#
# ChromaDB is a local persistent client — no container. The pre-built index at
# data_pipeline/processed/chromadb/ is mounted read-only into this container.

FROM python:3.12-slim

# ── System dependencies ────────────────────────────────────────────────────────
# build-essential: required to compile some Python C extensions (e.g. ormsgpack)
# curl:            used in healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Use requirements-agent.txt — lean subset of requirements.txt containing only
# what the MCP server and agent need. Excludes pyspark, shapely, ingestion libs.
COPY requirements-agent.txt .

# Install PyTorch CPU-only first — must precede sentence-transformers so it
# finds torch during install. CPU wheel is ~200MB vs ~800MB for CUDA.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining agent dependencies
RUN pip install --no-cache-dir -r requirements-agent.txt

# ── Pre-download sentence-transformers model ───────────────────────────────────
# Downloads multi-qa-mpnet-base-dot-v1 (~420MB) at build time so the first
# search_reviews / get_neighbourhood_context call is instant at runtime.
# Without this, the first call inside the container would stall for ~5 minutes.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('multi-qa-mpnet-base-dot-v1')"

# ── Application code ───────────────────────────────────────────────────────────
# Copy only the source files — data directories are mounted as volumes
COPY mcp_server/ mcp_server/
COPY agent/      agent/

# ── Environment defaults ───────────────────────────────────────────────────────
# These are overridden by the .env file loaded via docker-compose env_file
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://ollama:11434
ENV CHROMA_PATH=/app/chromadb
# Suppress HuggingFace Hub warnings and progress bars
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV TRANSFORMERS_VERBOSITY=error
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_VERBOSITY=error

# ── Default command — interactive agent REPL ──────────────────────────────────
CMD ["python", "agent/agent.py"]
