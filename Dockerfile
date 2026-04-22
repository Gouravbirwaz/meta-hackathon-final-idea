# KisanAgent — HuggingFace Spaces Docker SDK
# Multi-stage build: builder (uv sync) → runtime (slim)
# Deploys FastAPI on port 7860 (HF Spaces requirement)

# ── Stage 1: Build environment ─────────────────────────────
FROM python:3.11-slim AS builder

# Install uv — fast Python package manager
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Sync dependencies (frozen lockfile, no dev extras)
RUN uv sync --frozen --no-dev

# ── Stage 2: Runtime image ─────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security (HF Spaces best practice)
RUN useradd -m -u 1000 kisan
USER kisan

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=kisan:kisan /app/.venv /app/.venv

# Copy application code
COPY --chown=kisan:kisan . .

# Activate uv venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Logging configuration
ENV LOG_LEVEL=info

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Health check for HF Spaces readiness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

# Run FastAPI server on HF Spaces port
CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--log-level", "info", \
     "--workers", "1"]
