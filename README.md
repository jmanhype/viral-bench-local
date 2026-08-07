# Viral-Bench Local API Replacements

Self-hosted drop-in replacements for Lightreel, ScrapeCreators, and Doublespeed APIs.
Zero recurring costs. Runs on macOS + Ubuntu 3090 GPU server.

## Architecture

```
Mac (orchestration)          3090 Server (GPU)
├── research-api :8001       ├── ComfyUI :8188
├── scraper-api  :8010       └── SDXL/FLUX workflows
├── mcp-server   :8020
├── renderer     :8030
├── Postgres     :5432
├── Qdrant       :6333
├── Redis        :6379
└── MinIO        :9000
```

## Services

| Service | Port | Replaces | Status |
|---------|------|----------|--------|
| research-api | 8001 | Lightreel `/v1/chat` | 🔨 Building |
| scraper-api | 8010 | ScrapeCreators (3 endpoints) | 🔨 Building |
| mcp-server | 8020 | Doublespeed MCP | 📋 Planned |
| renderer | 8030 | Doublespeed slide rendering | 📋 Planned |

## Quick Start

```bash
cd ~/viral-bench-local
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Start all services
docker compose up -d postgres qdrant redis minio
python -m services.research.app &
python -m services.scraper.app &
```

## Viral-Bench Integration

Set env vars to point at local services:
```bash
export LIGHTREEL_API_URL="http://127.0.0.1:8001/v1/chat"
# ScrapeCreators/Doublespeed need /etc/hosts + Caddy or small source patch
```
