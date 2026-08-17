"""
ego-browser Bridge Service — Host-side proxy for Docker containers.

ego-browser is a macOS-native Mach-O binary with embedded Chromium.
It CANNOT run inside a Linux Docker container. This bridge runs on
the Mac host and exposes a REST API that containers can call to drive
ego-browser for Flux 3 Discord generation, Google Flow, etc.

Usage (host):
    python services/ego-bridge/app.py

Usage (from container):
    curl http://host.docker.internal:8040/v1/navigate -d '{"url":"https://..."}'
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ego-bridge")

app = FastAPI(title="ego-browser Bridge", version="0.1.0")

EGO_BROWSER = os.environ.get("EGO_BROWSER", str(Path.home() / ".local/bin/ego-browser"))
DEFAULT_PROFILE = os.environ.get("EGO_PROFILE", "Default")
TASK_SPACE_NAME = "docker-bridge"


def _run_ego(js_code: str, timeout: int = 60) -> str:
    """Run ego-browser nodejs heredoc and return stdout."""
    if not Path(EGO_BROWSER).exists():
        raise RuntimeError(f"ego-browser not found at {EGO_BROWSER}")

    script = f"""
const task = await useOrCreateTaskSpace('{TASK_SPACE_NAME}')
{js_code}
"""
    result = subprocess.run(
        [EGO_BROWSER, "nodejs", "-e", script],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        logger.error(f"ego-browser failed: {result.stderr[:500]}")
        raise RuntimeError(result.stderr[:1000])
    # ego-browser cliLog() writes to stderr, not stdout
    return result.stderr or result.stdout


# ─── Models ───────────────────────────────────────────────────────────────────

class NavigateRequest(BaseModel):
    url: str
    wait: bool = True
    timeout: int = 30

class EvaluateRequest(BaseModel):
    js: str
    timeout: int = 30

class ClickRequest(BaseModel):
    selector: str
    label: Optional[str] = None

class TypeRequest(BaseModel):
    selector: str
    text: str

class ScreenshotRequest(BaseModel):
    output_path: Optional[str] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    ego_exists = Path(EGO_BROWSER).exists()
    return {
        "status": "ok" if ego_exists else "degraded",
        "ego_browser": EGO_BROWSER,
        "ego_installed": ego_exists,
        "profile": DEFAULT_PROFILE,
    }


@app.post("/v1/navigate")
async def navigate(req: NavigateRequest):
    """Navigate to a URL using ego-browser."""
    js = f"""
await openOrReuseTab('{req.url}', {{ wait: {str(req.wait).lower()}, timeout: {req.timeout} }})
cliLog(JSON.stringify(await pageInfo()))
"""
    try:
        output = _run_ego(js, timeout=req.timeout + 10)
        # Parse last JSON line from output
        for line in reversed(output.strip().split("\n")):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return {"output": output}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/snapshot")
async def snapshot():
    """Get semantic page snapshot."""
    js = "cliLog(await snapshotText())"
    try:
        output = _run_ego(js)
        return {"snapshot": output}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/evaluate")
async def evaluate(req: EvaluateRequest):
    """Evaluate JavaScript in the browser page context."""
    js = f"cliLog(await js(String.raw`{req.js}`))"
    try:
        output = _run_ego(js, timeout=req.timeout + 10)
        return {"result": output.strip()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/click")
async def click(req: ClickRequest):
    """Click an element by selector or ref."""
    label_opt = f", {{ label: '{req.label}' }}" if req.label else ""
    js = f"await click('{req.selector}'{label_opt}); cliLog('clicked')"
    try:
        output = _run_ego(js)
        return {"status": "clicked", "output": output.strip()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/type")
async def type_text(req: TypeRequest):
    """Type text into an input element.

    Resolves the target robustly (exact selector first, else the Discord
    message box via aria-label^=), then focuses and types — so automated
    callers don't depend on a brittle exact aria-label value.
    """
    escaped_sel = req.selector.replace("\\", "\\\\").replace("'", "\\'")
    escaped_text = req.text.replace("\\", "\\\\").replace("'", "\\'")
    script_body = (
        "(() => {"
        "const s1 = '" + escaped_sel + "';"
        "let s = 'div[contenteditable=true][aria-label^=\"Message\"]';"
        "try { if (document.querySelector(s1)) s = s1; } catch (e) {}"
        "const el = document.querySelector(s);"
        "if (el) el.focus();"
        "return s;"
        "})()"
    )
    # Resolve + focus inside the page context (js()), then type via ego helper.
    js = (
        "const sel = await js(String.raw`" + script_body + "`);"
        f"await fillInput(sel, '{escaped_text}');"
        "cliLog('typed');"
    )
    try:
        output = _run_ego(js)
        return {"status": "typed", "output": output.strip()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/screenshot")
async def screenshot(req: ScreenshotRequest):
    """Capture a screenshot."""
    path = req.output_path or "/tmp/ego_screenshot.png"
    js = f"await captureScreenshot('{path}'); cliLog('{path}')"
    try:
        output = _run_ego(js)
        return {"path": path, "output": output.strip()}
    except Exception as e:
        raise HTTPException(500, str(e))


class DownloadRequest(BaseModel):
    url: str
    output_path: str


@app.post("/v1/download")
async def download_file(req: DownloadRequest):
    """Download a file from the host. Works for Discord CDN signed URLs
    since the host has direct internet access (unlike Docker containers)."""
    import subprocess, os

    dest = req.output_path
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    r = subprocess.run(
        ["curl", "-sL", "-m", "120", "-o", dest, req.url],
        capture_output=True, text=True, timeout=130,
    )
    if r.returncode != 0:
        raise HTTPException(500, f"Download failed: {r.stderr[:200]}")

    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    return {"path": dest, "size": size}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("EGO_BRIDGE_PORT", "8040"))
    logger.info(f"Starting ego-browser bridge on :{port}")
    logger.info(f"ego-browser binary: {EGO_BROWSER}")
    logger.info(f"Chrome profile: {DEFAULT_PROFILE}")
    uvicorn.run(app, host="0.0.0.0", port=port)
