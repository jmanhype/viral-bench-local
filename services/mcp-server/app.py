"""
Viral-Bench MCP Server — Doublespeed-compatible replacement.

Implements Streamable HTTP transport (protocol 2025-06-18) with tools for
TikTok carousel publishing: image generation, slide rendering, draft
management, and post queuing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import textwrap
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import Context, FastMCP
from PIL import Image, ImageDraw, ImageFont
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("viral-bench-mcp")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RENDERS_DIR = Path("/tmp/vbl-renders")
DRAFTS_DB = Path("/tmp/vbl-drafts/drafts.db")
AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "local-dev-token")
HOST = os.environ.get("MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("MCP_PORT", "8020"))
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://192.168.1.143:8188")

RENDERS_DIR.mkdir(parents=True, exist_ok=True)
DRAFTS_DB.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory session state (keyed by MCP session id)
# ---------------------------------------------------------------------------

_session_state: dict[str, dict[str, Any]] = {}


def _get_session(ctx: Context) -> dict[str, Any]:
    """Return per-session mutable state dict."""
    sid = ctx.session.client_id if hasattr(ctx.session, "client_id") else "default"
    if sid not in _session_state:
        _session_state[sid] = {}
    return _session_state[sid]


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DRAFTS_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drafts (
            id TEXT PRIMARY KEY,
            draft_name TEXT,
            caption TEXT,
            account_id TEXT,
            scene_data TEXT,
            music_link TEXT,
            surface_draft_entry INTEGER DEFAULT 0,
            share_url TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            draft_id TEXT,
            status TEXT DEFAULT 'queued',
            created_at TEXT,
            FOREIGN KEY (draft_id) REFERENCES drafts(id)
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Pillow helpers
# ---------------------------------------------------------------------------

def _placeholder_image(prompt: str, width: int = 1080, height: int = 1920) -> Path:
    """Generate a solid-colour placeholder PNG with prompt text overlay."""
    # Deterministic colour from prompt hash
    h = hashlib.md5(prompt.encode()).hexdigest()
    bg = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    # Try to use a decent font size; fall back to default
    font_size = max(24, min(48, width // 20))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    wrapped = textwrap.fill(prompt[:300], width=40)
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), wrapped, fill="white", font=font, align="center")

    fname = f"placeholder_{uuid.uuid4().hex[:12]}.png"
    out = RENDERS_DIR / fname
    img.save(out, "PNG")
    return out


def _render_slide(slide: dict, scale: float = 1.0) -> Path:
    """Composite text over a background image (or solid colour)."""
    width = int(1080 * scale)
    height = int(1920 * scale)

    bg_url = slide.get("background_image_url")
    if bg_url and Path(bg_url).exists():
        img = Image.open(bg_url).convert("RGB").resize((width, height))
    else:
        img = Image.new("RGB", (width, height), (30, 30, 30))

    draw = ImageDraw.Draw(img)
    text = slide.get("text", "")
    position = slide.get("position", "middle")

    font_size = max(20, min(56, width // 18))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    wrapped = textwrap.fill(text[:500], width=35)
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2

    if position == "top":
        y = int(height * 0.1)
    elif position == "bottom":
        y = int(height * 0.85) - th
    else:  # middle
        y = (height - th) // 2

    # Text shadow for readability
    draw.text((x + 2, y + 2), wrapped, fill="black", font=font, align="center")
    draw.text((x, y), wrapped, fill="white", font=font, align="center")

    fname = f"slide_{uuid.uuid4().hex[:12]}.png"
    out = RENDERS_DIR / fname
    img.save(out, "PNG")
    return out


# ---------------------------------------------------------------------------
# ComfyUI integration
# ---------------------------------------------------------------------------

def _build_sdxl_txt2img_workflow(prompt: str, width: int, height: int) -> dict:
    """Build a minimal SDXL txt2img workflow JSON for ComfyUI API."""
    # Ensure dimensions are multiples of 8 (required by VAE)
    width = max(64, (width // 8) * 8)
    height = max(64, (height // 8) * 8)

    workflow = {
        # CheckpointLoaderSimple
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            }
        },
        # CLIPTextEncode (positive prompt)
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["3", 1]
            }
        },
        # CLIPTextEncode (negative prompt)
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "blurry, low quality, distorted, watermark, text",
                "clip": ["3", 1]
            }
        },
        # EmptyLatentImage
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        # KSampler
        "3_sampler": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(uuid.uuid4().int % (2**32)),
                "steps": 25,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["3", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        # VAEDecode
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3_sampler", 0],
                "vae": ["3", 2]
            }
        },
        # SaveImage
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "vbl",
                "images": ["8", 0]
            }
        }
    }
    return {"prompt": workflow}


async def _generate_via_comfyui(prompt: str, width: int, height: int) -> Path | None:
    """Try to generate an image via ComfyUI. Returns path on success, None on failure."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)) as client:
            # Submit prompt
            workflow = _build_sdxl_txt2img_workflow(prompt, width, height)
            resp = await client.post(f"{COMFYUI_URL}/prompt", json=workflow)
            resp.raise_for_status()
            data = resp.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                logger.warning("[comfyui] No prompt_id in response: %s", data)
                return None

            logger.info("[comfyui] Submitted prompt %s (%dx%d)", prompt_id, width, height)

            # Poll /history/{prompt_id} until complete (max 120s)
            deadline = time.monotonic() + 120
            output_filename = None
            subfolder = ""
            img_type = "output"
            while time.monotonic() < deadline:
                await asyncio.sleep(2)
                hist_resp = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
                if hist_resp.status_code != 200:
                    continue
                hist = hist_resp.json()
                if prompt_id not in hist:
                    continue
                entry = hist[prompt_id]
                outputs = entry.get("outputs", {})
                images = outputs.get("images", [])
                if images:
                    output_filename = images[0].get("filename")
                    subfolder = images[0].get("subfolder", "")
                    img_type = images[0].get("type", "output")
                    break
                # Check for execution error
                status_info = entry.get("status", {})
                if status_info.get("status_str") == "error":
                    logger.error("[comfyui] Generation error: %s", status_info)
                    return None

            if not output_filename:
                logger.warning("[comfyui] Timed out waiting for prompt %s", prompt_id)
                return None

            # Download generated image
            view_params = {"filename": output_filename, "type": img_type}
            if subfolder:
                view_params["subfolder"] = subfolder
            dl_resp = await client.get(f"{COMFYUI_URL}/view", params=view_params)
            dl_resp.raise_for_status()

            fname = f"comfyui_{uuid.uuid4().hex[:12]}.png"
            out_path = RENDERS_DIR / fname
            out_path.write_bytes(dl_resp.content)
            logger.info("[comfyui] Saved generated image to %s", out_path)
            return out_path

    except Exception as exc:
        logger.warning("[comfyui] Failed (%s), falling back to Pillow", exc)
        return None


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Simple bearer-token auth for all routes except /health."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse({"error": "Missing or invalid Authorization header"}, status_code=401)

        token = auth_header[len("Bearer "):]
        if token != AUTH_TOKEN:
            return JSONResponse({"error": "Invalid token"}, status_code=403)

        return await call_next(request)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "viral-bench-mcp",
    instructions="Viral-Bench MCP server — Doublespeed-compatible TikTok carousel publishing pipeline.",
    host=HOST,
    port=PORT,
    stateless_http=True,
    json_response=False,
    streamable_http_path="/mcp",
)


# ---- Tool: set_product ----

@mcp.tool()
async def set_product(product_id: str, ctx: Context) -> dict:
    """Set the active product ID for this session."""
    state = _get_session(ctx)
    state["product_id"] = product_id
    return {"success": True, "product_id": product_id}


# ---- Tool: list_style_presets ----

STYLE_PRESETS = [
    {
        "id": "tiktok-bg",
        "name": "TikTok Background",
        "description": "Full-bleed background optimised for TikTok carousel slides",
        "width": 1080,
        "height": 1920,
    },
    {
        "id": "tiktok-stroke",
        "name": "TikTok Stroke",
        "description": "Text stroke style for high-contrast TikTok overlays",
        "stroke_color": "#FFFFFF",
        "stroke_width": 3,
    },
    {
        "id": "clean-white",
        "name": "Clean White",
        "description": "Minimalist white background with dark text",
        "bg_color": "#FFFFFF",
        "text_color": "#1A1A1A",
    },
    {
        "id": "dark-gradient",
        "name": "Dark Gradient",
        "description": "Dark gradient background for premium feel",
        "gradient": ["#1A1A2E", "#16213E"],
    },
]


@mcp.tool()
async def list_style_presets() -> dict:
    """Return available style presets for carousel slides."""
    return {"presets": STYLE_PRESETS}


# ---- Tool: generate_image ----

@mcp.tool()
async def generate_image(
    prompt: str,
    width: int = 1080,
    height: int = 1920,
    model: str = "sdxl",
    image_url: str | None = None,
) -> dict:
    """Generate an image from a text prompt. Uses ComfyUI/SDXL when available, falls back to placeholder."""
    backend = "pillow-fallback"
    out = await _generate_via_comfyui(prompt, width, height)
    if out is not None:
        backend = "comfyui"
    else:
        logger.info("[generate_image] Using Pillow fallback for prompt: %s", prompt[:60])
        out = _placeholder_image(prompt, width, height)

    url = f"http://{HOST}:{PORT}/renders/{out.name}"
    return {
        "success": True,
        "image_url": url,
        "local_path": str(out),
        "width": width,
        "height": height,
        "model": model,
        "backend": backend,
    }


# ---- Tool: render_slides ----

@mcp.tool()
async def render_slides(
    scene_data: dict,
    scale: float = 1.0,
    max_slides: int = 10,
) -> dict:
    """Render carousel slides by compositing text over backgrounds."""
    slides = scene_data.get("slides", [])
    if not slides:
        return {"success": False, "error": "No slides in scene_data"}

    slides = slides[:max_slides]
    previews = []
    for i, slide in enumerate(slides):
        out = _render_slide(slide, scale)
        previews.append({
            "index": i,
            "url": f"http://{HOST}:{PORT}/renders/{out.name}",
            "local_path": str(out),
        })

    return {
        "success": True,
        "slide_count": len(previews),
        "previews": previews,
    }


# ---- Tool: upsert_slideshow_draft ----

@mcp.tool()
async def upsert_slideshow_draft(
    draft_name: str,
    caption: str,
    account_id: str,
    scene_data: dict,
    music_link: str | None = None,
    surface_draft_entry: bool = False,
    create_share_link: bool = False,
) -> dict:
    """Create or update a slideshow draft in the local drafts database."""
    now = datetime.now(timezone.utc).isoformat()
    group_id = hashlib.sha256(f"{account_id}:{draft_name}".encode()).hexdigest()[:16]
    share_url = f"http://{HOST}:{PORT}/share/{group_id}" if create_share_link else None

    conn = _db()
    existing = conn.execute("SELECT id FROM drafts WHERE id = ?", (group_id,)).fetchone()

    if existing:
        conn.execute("""
            UPDATE drafts SET
                draft_name=?, caption=?, account_id=?, scene_data=?,
                music_link=?, surface_draft_entry=?, share_url=?, updated_at=?
            WHERE id=?
        """, (draft_name, caption, account_id, json.dumps(scene_data),
              music_link, int(surface_draft_entry), share_url, now, group_id))
    else:
        conn.execute("""
            INSERT INTO drafts (id, draft_name, caption, account_id, scene_data,
                                music_link, surface_draft_entry, share_url,
                                created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (group_id, draft_name, caption, account_id, json.dumps(scene_data),
              music_link, int(surface_draft_entry), share_url, now, now))

    conn.commit()
    conn.close()

    return {
        "success": True,
        "group_id": group_id,
        "share_url": share_url,
        "action": "updated" if existing else "created",
    }


# ---- Tool: queue_post ----

@mcp.tool()
async def queue_post(draft_id: str) -> dict:
    """Queue a draft for publishing. Creates a job record."""
    conn = _db()
    draft = conn.execute("SELECT id FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if not draft:
        conn.close()
        return {"success": False, "error": f"Draft {draft_id} not found"}

    job_id = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO jobs (id, draft_id, status, created_at) VALUES (?, ?, 'queued', ?)",
        (job_id, draft_id, now),
    )
    conn.commit()
    conn.close()

    return {"success": True, "job_id": job_id, "status": "queued"}


# ---- Tool: list_posts ----

@mcp.tool()
async def list_posts(limit: int = 20) -> dict:
    """List recent drafts and their job statuses."""
    conn = _db()
    rows = conn.execute("""
        SELECT d.id, d.draft_name, d.caption, d.account_id, d.created_at, d.updated_at,
               j.id as job_id, j.status as job_status
        FROM drafts d
        LEFT JOIN jobs j ON j.draft_id = d.id
        ORDER BY d.updated_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    posts = [dict(r) for r in rows]
    return {"posts": posts, "count": len(posts)}


# ---------------------------------------------------------------------------
# Custom routes: health check + static file serving for renders
# ---------------------------------------------------------------------------

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request):
    return JSONResponse({
        "status": "ok",
        "service": "viral-bench-mcp",
        "protocol": "2025-06-18",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@mcp.custom_route("/renders/{filename}", methods=["GET"])
async def serve_render(request: Request):
    from starlette.responses import FileResponse
    filename = request.path_params["filename"]
    fpath = RENDERS_DIR / filename
    if not fpath.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(fpath), media_type="image/png")


@mcp.custom_route("/share/{group_id}", methods=["GET"])
async def share_draft(request: Request):
    group_id = request.path_params["group_id"]
    conn = _db()
    row = conn.execute("SELECT * FROM drafts WHERE id = ?", (group_id,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "Draft not found"}, status_code=404)
    return JSONResponse(dict(row))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    app = mcp.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware)

    print(f"[viral-bench-mcp] Starting on {HOST}:{PORT}")
    print(f"[viral-bench-mcp] Auth token: {'set via env' if os.environ.get('MCP_AUTH_TOKEN') else 'using default'}")
    print(f"[viral-bench-mcp] Renders dir: {RENDERS_DIR}")
    print(f"[viral-bench-mcp] Drafts DB: {DRAFTS_DB}")
    print(f"[viral-bench-mcp] ComfyUI URL: {COMFYUI_URL}")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
