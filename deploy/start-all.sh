#!/bin/bash
# Start all Viral-Bench Local API services
set -e

cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
export PATH="$(pwd)/.venv/bin:$PATH"

echo "=== Starting Viral-Bench Local APIs ==="

# Research API (Lightreel replacement)
echo "[1/3] Starting research-api on :8001..."
MODELSCOPE_API_KEY="${MODELSCOPE_API_KEY:?Set MODELSCOPE_API_KEY env var}" \
  $VENV -m uvicorn services.research.app:app --host 0.0.0.0 --port 8001 &
PID_RESEARCH=$!

# Scraper API (ScrapeCreators replacement)
echo "[2/3] Starting scraper-api on :8010..."
$VENV -m uvicorn services.scraper.app:app --host 0.0.0.0 --port 8010 &
PID_SCRAPER=$!

# MCP Server (Doublespeed replacement)
echo "[3/3] Starting mcp-server on :8020..."
MCP_AUTH_TOKEN="${MCP_AUTH_TOKEN:-local-dev-token}" \
  $VENV -m uvicorn services.mcp_server.app:app --host 0.0.0.0 --port 8020 &
PID_MCP=$!

echo ""
echo "All services started:"
echo "  Research API:  http://127.0.0.1:8001  (PID $PID_RESEARCH)"
echo "  Scraper API:   http://127.0.0.1:8010  (PID $PID_SCRAPER)"
echo "  MCP Server:    http://127.0.0.1:8020  (PID $PID_MCP)"
echo ""
echo "Press Ctrl+C to stop all services"

trap "kill $PID_RESEARCH $PID_SCRAPER $PID_MCP 2>/dev/null; echo 'Stopped.'" EXIT
wait
