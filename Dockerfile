# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="viral-bench-local"
LABEL org.opencontainers.image.description="Self-hosted Viral-Bench API stack"
LABEL org.opencontainers.image.source="https://github.com/jmanhype/viral-bench-local"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (yt-dlp needs ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install . 2>&1 | tail -5

# Copy app
COPY . .

# ─── research-api (port 8001) ───────────────────────────────────
FROM base AS research-api
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8001/health || exit 1
CMD ["uvicorn", "services.research.app:app", "--host", "0.0.0.0", "--port", "8001"]

# ─── scraper-api (port 8010) ───────────────────────────────────
FROM base AS scraper-api
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8010/health || exit 1
CMD ["uvicorn", "services.scraper.app:app", "--host", "0.0.0.0", "--port", "8010"]

# ─── browser-worker (port 8020) ────────────────────────────────
FROM base AS browser-worker
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    && rm -rf /var/lib/apt/lists/*
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/lib/playwright
EXPOSE 8020
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8020/health || exit 1
CMD ["uvicorn", "services.browser.app:app", "--host", "0.0.0.0", "--port", "8020"]

# ─── publisher (port 8031) ─────────────────────────────────────
FROM base AS publisher
EXPOSE 8031
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8031/health || exit 1
CMD ["uvicorn", "services.publisher.app:app", "--host", "0.0.0.0", "--port", "8031"]

# ─── renderer (port 8030) ─────────────────────────────────────
FROM base AS renderer
EXPOSE 8030
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8030/health || exit 1
CMD ["uvicorn", "services.renderer.app:app", "--host", "0.0.0.0", "--port", "8030"]
