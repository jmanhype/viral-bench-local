"""
H3 Bridge Service — Host-side SSH/SCP proxy for Docker containers.

Docker Desktop Mac containers cannot SSH to LAN IPs (192.168.1.143).
This service runs on the Mac HOST and exposes REST endpoints for H3
operations. Factory containers call these via host.docker.internal:8041.

Mirrors lib/h3_pipeline.py logic but runs where SSH actually works.

Start: .venv/bin/python services/h3-bridge/app.py
"""
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("h3-bridge")

app = FastAPI(title="H3 Bridge", version="0.1.0")

# ─── Config (same env vars as lib/h3_pipeline.py) ─────────────────────────────
GPU_HOST = os.environ.get("H3_GPU_HOST", "192.168.1.143")
GPU_USER = os.environ.get("H3_GPU_USER", "straughter")
REMOTE_HOME = os.environ.get("H3_REMOTE_HOME", f"/home/{GPU_USER}/Wan2GP")
REMOTE_RUN_DIR = REMOTE_HOME
REMOTE_OUTPUTS_DIR = f"{REMOTE_HOME}/outputs"

WGP_CMD = (
    "source venv/bin/activate && "
    "PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True "
    "python3 wgp.py --process {job} --profile 3 --attention sdpa"
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\.]+$")


def _safe_id(value: str, what: str) -> str:
    if not _SAFE_ID_RE.match(value):
        raise ValueError(f"Unsafe {what}: '{value}'")
    return value


def _ssh(remote_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           f"{GPU_USER}@{GPU_HOST}", remote_cmd]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _scp_to(local: str, remote: str, timeout: int = 60) -> subprocess.CompletedProcess:
    cmd = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           local, f"{GPU_USER}@{GPU_HOST}:{remote}"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _scp_from(remote: str, local: str, timeout: int = 120) -> subprocess.CompletedProcess:
    cmd = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           f"{GPU_USER}@{GPU_HOST}:{remote}", local]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ─── Models ────────────────────────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    job_json: dict = Field(..., description="Complete WGP job config")


class RetrieveRequest(BaseModel):
    remote_filename: str = Field(..., description="Filename in Wan2GP/outputs/")
    local_path: str = Field(..., description="Local destination path")


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check SSH connectivity to the 3090."""
    try:
        r = _ssh("echo ok", timeout=10)
        reachable = r.returncode == 0 and "ok" in r.stdout
        return {
            "status": "ok" if reachable else "degraded",
            "ssh_reachable": reachable,
            "gpu_host": GPU_HOST,
            "gpu_user": GPU_USER,
            "remote_home": REMOTE_HOME,
        }
    except Exception as e:
        return {"status": "error", "ssh_reachable": False, "error": str(e)}


@app.post("/v1/h3/submit")
async def submit(req: SubmitRequest):
    """Submit an H3 job: write JSON, SCP to 3090, launch in tmux."""
    job_id = uuid.uuid4().hex[:12]
    session_name = f"h3-{job_id}"
    _safe_id(session_name, "session name")

    # Write job JSON to temp file
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix=f"h3_{job_id}_", delete=False
    ) as tf:
        json.dump(req.job_json, tf, ensure_ascii=False, indent=2)
        tf.write("\n")
        temp_path = tf.name

    remote_job = f"{REMOTE_RUN_DIR}/{job_id}.json"

    try:
        # SCP the job file
        logger.info(f"SCP {temp_path} -> {remote_job}")
        r = _scp_to(temp_path, remote_job)
        if r.returncode != 0:
            raise RuntimeError(f"SCP failed: {r.stderr.strip()}")

        # Launch wgp.py in tmux
        safe_job = _safe_id(f"{job_id}.json", "job file")
        launch_cmd = f"cd {REMOTE_RUN_DIR} && tmux new-session -d -s {session_name} '{WGP_CMD.format(job=safe_job)}'"
        logger.info(f"Launching tmux session {session_name}")
        r = _ssh(launch_cmd, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"tmux launch failed: {r.stderr.strip()}")

        logger.info(f"Submitted: session={session_name}")
        return {"session_name": session_name}

    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@app.get("/v1/h3/status/{session_name}")
async def status(session_name: str):
    """Poll H3 job status via tmux pane capture."""
    _safe_id(session_name, "session name")

    try:
        # Check if tmux session exists
        r = _ssh(f"tmux has-session -t {session_name} 2>/dev/null && echo yes || echo no")
        if r.returncode != 0 or "yes" not in r.stdout:
            # Session gone — check if output was produced
            r2 = _ssh(
                f"ls -1t {REMOTE_OUTPUTS_DIR}/*.mp4 2>/dev/null | head -1"
            )
            if r2.returncode == 0 and r2.stdout.strip():
                latest = r2.stdout.strip().split("/")[-1]
                return {
                    "status": "complete",
                    "progress": 1.0,
                    "output_file": latest,
                    "error": None,
                }
            return {
                "status": "failed",
                "progress": 0.0,
                "output_file": None,
                "error": "tmux session not found",
            }

        # Capture pane output for progress parsing
        r = _ssh(f"tmux capture-pane -p -t {session_name} | tail -c 4000")
        pane = r.stdout if r.returncode == 0 else ""

        # Parse denoising progress: "H3 denoising:  45%|..." or "[9/20]"
        progress = 0.0
        import re as _re
        pct_match = _re.search(r"(\d+)%", pane)
        step_match = _re.search(r"\[(\d+)/(\d+)\]", pane)
        if pct_match:
            progress = int(pct_match.group(1)) / 100.0
        elif step_match:
            done, total = int(step_match.group(1)), int(step_match.group(2))
            progress = done / total if total > 0 else 0.0

        # Check for completion markers
        if "Task 1 completed" in pane or "Queue completed" in pane:
            # Get the output filename
            r2 = _ssh(f"ls -1t {REMOTE_OUTPUTS_DIR}/*.mp4 2>/dev/null | head -1")
            output_file = r2.stdout.strip().split("/")[-1] if r2.returncode == 0 and r2.stdout.strip() else None
            return {
                "status": "complete",
                "progress": 1.0,
                "output_file": output_file,
                "error": None,
            }

        # Check for failure
        if "Traceback" in pane or "CUDA out of memory" in pane or "Error" in pane:
            error_lines = [l for l in pane.split("\n") if "Error" in l or "Traceback" in l]
            return {
                "status": "failed",
                "progress": progress,
                "output_file": None,
                "error": "; ".join(error_lines[:3]) or "unknown error",
            }

        return {
            "status": "running",
            "progress": progress,
            "output_file": None,
            "error": None,
        }

    except subprocess.TimeoutExpired:
        return {"status": "running", "progress": 0.0, "output_file": None, "error": "SSH timeout"}
    except Exception as e:
        logger.error(f"Status check failed for {session_name}: {e}")
        return {"status": "error", "progress": 0.0, "output_file": None, "error": str(e)}


@app.post("/v1/h3/retrieve")
async def retrieve(req: RetrieveRequest):
    """SCP an output file from 3090 to local path."""
    safe_name = _safe_id(req.remote_filename, "filename")
    remote = f"{REMOTE_OUTPUTS_DIR}/{safe_name}"
    local = req.local_path

    # Ensure local directory exists
    Path(local).parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Retrieving {remote} -> {local}")
        r = _scp_from(remote, local, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"SCP retrieve failed: {r.stderr.strip()}")

        size = Path(local).stat().st_size if Path(local).exists() else 0
        logger.info(f"Retrieved: {local} ({size} bytes)")
        return {"local_path": local, "size": size}

    except Exception as e:
        logger.error(f"Retrieve failed: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("H3_BRIDGE_PORT", "8041"))
    logger.info(f"Starting H3 bridge on :{port} -> {GPU_USER}@{GPU_HOST}")
    uvicorn.run(app, host="0.0.0.0", port=port)
