#!/usr/bin/env bash
# Viral-Bench Local — Start All Services
# Usage: ./start-all.sh [--stop]

set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv/bin/python"
LOGDIR="/tmp/vbl-logs"
mkdir -p "$LOGDIR"

stop_services() {
    echo "Stopping all VBL services..."
    for port in 8001 8010 8020 8030 8031; do
        pid=$(lsof -i :$port -t 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill $pid 2>/dev/null && echo "  Killed :$port (PID $pid)" || true
        fi
    done
    echo "Done."
}

if [ "${1:-}" = "--stop" ]; then
    stop_services
    exit 0
fi

# Check venv
if [ ! -f "$VENV" ]; then
    echo "ERROR: Virtual environment not found. Run: uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt"
    exit 1
fi

echo "Starting Viral-Bench Local services..."

# Research API (:8001)
if ! lsof -i :8001 -t >/dev/null 2>&1; then
    MODELSCOPE_API_KEY="${MODELSCOPE_API_KEY:?Set MODELSCOPE_API_KEY env var or in .env}" \
    nohup $VENV -m uvicorn services.research.app:app --host "${RESEARCH_HOST:-127.0.0.1}" --port 8001 > "$LOGDIR/research.log" 2>&1 &
    echo "  ✅ Research API  :8001 (PID $!)"
else
    echo "  ⏭️  Research API  :8001 (already running)"
fi

# Scraper API (:8010)
if ! lsof -i :8010 -t >/dev/null 2>&1; then
    PATH="$(pwd)/.venv/bin:$PATH" \
    SCRAPER_API_KEY="${SCRAPER_API_KEY:?Set SCRAPER_API_KEY in .env before starting the scraper}" \
    nohup $VENV -m uvicorn services.scraper.app:app --host "${SCRAPER_HOST:-127.0.0.1}" --port 8010 > "$LOGDIR/scraper.log" 2>&1 &
    echo "  ✅ Scraper API   :8010 (PID $!)"
else
    echo "  ⏭️  Scraper API   :8010 (already running)"
fi

# MCP Server (:8020)
if ! lsof -i :8020 -t >/dev/null 2>&1; then
    MCP_AUTH_TOKEN="${MCP_AUTH_TOKEN:?Set MCP_AUTH_TOKEN in .env before starting the MCP server}" \
    nohup $VENV services/mcp-server/app.py > "$LOGDIR/mcp.log" 2>&1 &
    echo "  ✅ MCP Server    :8020 (PID $!)"
else
    echo "  ⏭️  MCP Server    :8020 (already running)"
fi

# Publisher (:8030)
if ! lsof -i :8030 -t >/dev/null 2>&1; then
    nohup $VENV services/publisher/app.py > "$LOGDIR/publisher.log" 2>&1 &
    echo "  ✅ Publisher     :8030 (PID $!)"
else
    echo "  ⏭️  Publisher     :8030 (already running)"
fi

# Renderer (:8031)
if ! lsof -i :8031 -t >/dev/null 2>&1; then
    COMFYUI_URL="${COMFYUI_URL:-http://gpu-server:8188}" \
    nohup $VENV services/renderer/app.py > "$LOGDIR/renderer.log" 2>&1 &
    echo "  ✅ Renderer      :8031 (PID $!)"
else
    echo "  ⏭️  Renderer      :8031 (already running)"
fi

echo ""
echo "Logs: $LOGDIR/"
echo "Stop: ./start-all.sh --stop"
echo ""

# Health check
sleep 2
echo "Health checks:"
for entry in "8001:Research" "8010:Scraper" "8020:MCP" "8030:Publisher" "8031:Renderer"; do
    port="${entry%%:*}"
    name="${entry##*:}"
    status=$(curl -s --max-time 3 http://127.0.0.1:$port/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo "DOWN")
    printf "  %-10s :%s %s\n" "$name" "$port" "$status"
done
