"""
TikTok Publisher Service — FastAPI service on port 8030.

Publishes queued drafts to TikTok via Content Posting API v2,
with Playwright-based browser automation fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import os
import sys

if not __package__:
    # Direct-script launch (`python services/publisher/app.py`): put the parent
    # `services/` dir on the path so the shared helper resolves.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from secure_env import effective_host
else:
    from services.secure_env import effective_host

logger = logging.getLogger("vbl-publisher")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DRAFTS_DB = Path("/tmp/vbl-drafts/drafts.db")
HOST = effective_host("PUBLISHER_HOST")
PORT = int(os.environ.get("PUBLISHER_PORT", "8030"))

# TikTok API config
TIKTOK_API_BASE = "https://open.tiktokapis.com"
TIKTOK_CLIENT_KEY = os.environ.get("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")

# Rate limiting: 6 requests/min/user token
_rate_lock = asyncio.Lock()
_last_request_time: float = 0.0
MIN_REQUEST_INTERVAL = 10.5  # seconds between requests (6/min = 10s gap)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PublishRequest(BaseModel):
    draft_id: str
    account_id: str
    access_token: str


class PublishResponse(BaseModel):
    publish_id: str
    status: str
    post_url: str | None = None
    method: str = "api"  # "api" or "browser"


class StatusResponse(BaseModel):
    publish_id: str
    status: str
    fail_reason: str | None = None
    post_url: str | None = None


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    """Connect to the shared drafts database."""
    if not DRAFTS_DB.exists():
        raise HTTPException(status_code=503, detail=f"Drafts DB not found at {DRAFTS_DB}")
    conn = sqlite3.connect(str(DRAFTS_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_publish_columns(conn: sqlite3.Connection) -> None:
    """Add publish tracking columns to jobs table and drafts table if missing."""
    cursor = conn.execute("PRAGMA table_info(jobs)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "publish_id" not in existing_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN publish_id TEXT")
    if "tiktok_status" not in existing_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN tiktok_status TEXT DEFAULT 'pending'")
    if "post_url" not in existing_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN post_url TEXT")
    if "fail_reason" not in existing_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN fail_reason TEXT")
    if "publish_method" not in existing_cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN publish_method TEXT DEFAULT 'api'")

    # Ensure drafts table has updated_at column (referenced by list_drafts query)
    cursor = conn.execute("PRAGMA table_info(drafts)")
    draft_cols = {row[1] for row in cursor.fetchall()}
    if "updated_at" not in draft_cols:
        conn.execute("ALTER TABLE drafts ADD COLUMN updated_at TIMESTAMP")

    conn.commit()


# ---------------------------------------------------------------------------
# TikTok Content Posting API v2
# ---------------------------------------------------------------------------

async def _rate_limit_wait() -> None:
    """Enforce 6 requests/min rate limit."""
    global _last_request_time
    async with _rate_lock:
        now = asyncio.get_event_loop().time()
        elapsed = now - _last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time = asyncio.get_event_loop().time()


async def _tiktok_init_photo_post(
    access_token: str,
    image_urls: list[str],
    caption: str,
) -> dict:
    """
    Initialize a photo carousel post via TikTok Content Posting API v2.

    POST /v2/post/publish/content/init/
    media_type=PHOTO, source_info.source=PULL_FROM_URL
    """
    await _rate_limit_wait()

    payload = {
        "post_info": {
            "title": caption[:150],  # TikTok title limit
            "description": caption,
            "privacy_level": "SELF_ONLY",  # Private until audit passes
            "disable_comment": False,
            "disable_duet": False,
            "disable_stitch": False,
        },
        "source_info": {
            "source": "PULL_FROM_URL",
            "photo_cover_index": 0,
            "photo_images": image_urls,
        },
        "media_type": "PHOTO",
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        resp = await client.post(
            f"{TIKTOK_API_BASE}/v2/post/publish/content/init/",
            json=payload,
            headers=headers,
        )

        if resp.status_code != 200:
            logger.error("TikTok init failed: %s %s", resp.status_code, resp.text)
            return {"success": False, "error": resp.text, "status_code": resp.status_code}

        data = resp.json()
        publish_id = data.get("data", {}).get("publish_id")
        if not publish_id:
            return {"success": False, "error": f"No publish_id in response: {data}"}

        return {"success": True, "publish_id": publish_id}


async def _tiktok_check_status(access_token: str, publish_id: str) -> dict:
    """
    Check publish status via GET /v2/post/publish/status/fetch/
    """
    await _rate_limit_wait()

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
    }

    params = {"publish_id": publish_id}

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        resp = await client.get(
            f"{TIKTOK_API_BASE}/v2/post/publish/status/fetch/",
            params=params,
            headers=headers,
        )

        if resp.status_code != 200:
            return {"status": "UNKNOWN", "error": resp.text}

        data = resp.json()
        pub_data = data.get("data", {})
        return {
            "status": pub_data.get("status", "UNKNOWN"),
            "fail_reason": pub_data.get("fail_reason"),
            "post_url": pub_data.get("item_id"),  # item_id can be used to construct URL
        }


# ---------------------------------------------------------------------------
# Browser automation fallback (Playwright)
# ---------------------------------------------------------------------------

async def _browser_publish(draft: dict, screenshots_dir: Path) -> dict:
    """
    Fallback: Use Playwright to upload images to TikTok Creator Portal.
    Returns publish info or error.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {"success": False, "error": "Playwright not installed"}

    screenshots_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = screenshots_dir / f"publish_{uuid.uuid4().hex[:12]}.png"

    scene_data = json.loads(draft["scene_data"]) if isinstance(draft["scene_data"], str) else draft["scene_data"]
    slides = scene_data.get("slides", [])
    caption = draft.get("caption", "")

    if not slides:
        return {"success": False, "error": "No slides in draft"}

    # Collect local image paths from rendered slides
    image_paths = []
    for slide in slides:
        bg = slide.get("background_image_url")
        if bg and Path(bg).exists():
            image_paths.append(bg)

    if not image_paths:
        return {"success": False, "error": "No renderable images found for browser upload"}

    logger.info("[browser-publish] Launching headless Chromium for %d images", len(image_paths))

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            )
            page = await context.new_page()

            # Navigate to TikTok upload page
            await page.goto("https://www.tiktok.com/upload", timeout=30000)
            await page.wait_for_timeout(3000)

            # Take screenshot of current state
            await page.screenshot(path=str(screenshot_path), full_page=False)
            logger.info("[browser-publish] Screenshot saved: %s", screenshot_path)

            # Try to find and use the file upload input
            file_input = await page.query_selector('input[type="file"]')
            if file_input:
                await file_input.set_input_files(image_paths[:10])  # Max 10 images
                await page.wait_for_timeout(5000)
                logger.info("[browser-publish] Files uploaded via input")
            else:
                logger.warning("[browser-publish] No file input found, may need manual intervention")

            # Try to fill caption
            caption_selectors = [
                '[data-text="true"]',
                '.ql-editor',
                '[contenteditable="true"]',
                'textarea',
            ]
            caption_filled = False
            for sel in caption_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        await el.fill(caption[:2200])  # TikTok caption limit
                        caption_filled = True
                        break
                except Exception:
                    continue

            if not caption_filled:
                logger.warning("[browser-publish] Could not fill caption automatically")

            # Take final screenshot
            final_screenshot = screenshots_dir / f"publish_final_{uuid.uuid4().hex[:12]}.png"
            await page.screenshot(path=str(final_screenshot), full_page=False)

            # Note: We don't auto-submit — save as draft in TikTok
            # The user would need to review and submit manually, or we'd need
            # to handle TikTok's specific UI which changes frequently
            logger.info("[browser-publish] Upload prepared. Manual submission may be required.")

            await browser.close()

            return {
                "success": True,
                "publish_id": f"browser_{uuid.uuid4().hex[:16]}",
                "method": "browser",
                "screenshot": str(screenshot_path),
                "final_screenshot": str(final_screenshot),
                "note": "Upload prepared via browser. Review in TikTok Creator Portal.",
            }

    except Exception as exc:
        logger.error("[browser-publish] Failed: %s", exc)
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure DB schema has publish tracking columns on startup."""
    try:
        conn = _db()
        _ensure_publish_columns(conn)
        conn.close()
        logger.info("Publisher service ready. DB: %s", DRAFTS_DB)
    except Exception as exc:
        logger.warning("Could not initialize DB on startup: %s", exc)
    yield


app = FastAPI(
    title="Viral-Bench Publisher",
    description="TikTok publisher service for Viral-Bench carousel drafts",
    version="1.0.0",
    lifespan=lifespan,
)


# ---- GET /health ----

@app.get("/health")
async def health():
    """Service health check."""
    db_ok = DRAFTS_DB.exists()
    return {
        "status": "ok",
        "service": "vbl-publisher",
        "port": PORT,
        "db_available": db_ok,
        "tiktok_api_configured": bool(TIKTOK_CLIENT_KEY),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---- GET /drafts ----

@app.get("/drafts")
async def list_drafts(limit: int = 50):
    """List pending drafts from SQLite."""
    conn = _db()
    rows = conn.execute("""
        SELECT d.id, d.draft_name, d.caption, d.account_id,
               d.created_at, d.updated_at,
               j.id as job_id, j.status as job_status,
               j.publish_id, j.tiktok_status
        FROM drafts d
        LEFT JOIN jobs j ON j.draft_id = d.id
        ORDER BY d.updated_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    drafts = [dict(r) for r in rows]
    return {"drafts": drafts, "count": len(drafts)}


# ---- POST /publish ----

@app.post("/publish")
async def publish_draft(req: PublishRequest):
    """
    Publish a draft to TikTok.

    1. Read draft from SQLite
    2. Try TikTok Content Posting API v2
    3. Fall back to Playwright browser automation if API fails
    4. Store publish_id + status back in DB
    """
    conn = _db()
    _ensure_publish_columns(conn)

    # Fetch draft
    draft = conn.execute("SELECT * FROM drafts WHERE id = ?", (req.draft_id,)).fetchone()
    if not draft:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Draft {req.draft_id} not found")

    draft_dict = dict(draft)

    # Parse scene_data to get image URLs
    scene_data = json.loads(draft_dict["scene_data"]) if isinstance(draft_dict["scene_data"], str) else draft_dict["scene_data"]
    slides = scene_data.get("slides", [])

    # Collect publicly accessible image URLs for API method
    image_urls = []
    for slide in slides:
        url = slide.get("background_image_url", "")
        if url and url.startswith("http"):
            image_urls.append(url)

    publish_id = None
    status = "pending"
    method = "api"
    post_url = None
    fail_reason = None

    # Try TikTok API first
    api_success = False
    if image_urls and req.access_token:
        try:
            result = await _tiktok_init_photo_post(req.access_token, image_urls, draft_dict.get("caption", ""))
            if result.get("success"):
                publish_id = result["publish_id"]
                status = "uploaded"
                api_success = True
                logger.info("[publish] TikTok API success: publish_id=%s", publish_id)
            else:
                fail_reason = result.get("error", "Unknown API error")
                logger.warning("[publish] TikTok API failed: %s", fail_reason)
        except Exception as exc:
            fail_reason = str(exc)
            logger.error("[publish] TikTok API exception: %s", exc)

    # Fallback to browser automation
    if not api_success:
        logger.info("[publish] Falling back to browser automation")
        method = "browser"
        screenshots_dir = Path("/tmp/vbl-screenshots")
        browser_result = await _browser_publish(draft_dict, screenshots_dir)

        if browser_result.get("success"):
            publish_id = browser_result["publish_id"]
            status = "browser_prepared"
            post_url = browser_result.get("screenshot")
            fail_reason = browser_result.get("note")
        else:
            status = "failed"
            fail_reason = browser_result.get("error", "Browser automation failed")

    # Create/update job record
    job_id = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).isoformat()

    # Ensure publish_id is always set for status lookups
    # When browser publish fails, use job_id as the canonical identifier
    effective_publish_id = publish_id or job_id

    # Check if a job already exists for this draft
    existing_job = conn.execute(
        "SELECT id FROM jobs WHERE draft_id = ? ORDER BY created_at DESC LIMIT 1",
        (req.draft_id,),
    ).fetchone()

    if existing_job:
        conn.execute("""
            UPDATE jobs SET
                status=?, publish_id=?, tiktok_status=?, post_url=?,
                fail_reason=?, publish_method=?
            WHERE id=?
        """, (status, effective_publish_id, status, post_url, fail_reason, method, existing_job["id"]))
        job_id = existing_job["id"]
    else:
        conn.execute("""
            INSERT INTO jobs (id, draft_id, status, created_at, publish_id, tiktok_status, post_url, fail_reason, publish_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (job_id, req.draft_id, status, now, effective_publish_id, status, post_url, fail_reason, method))

    conn.commit()
    conn.close()

    return PublishResponse(
        publish_id=effective_publish_id,
        status=status,
        post_url=post_url,
        method=method,
    )


# ---- GET /status/{publish_id} ----

@app.get("/status/{publish_id}")
async def get_status(publish_id: str, access_token: str = ""):
    """
    Check TikTok post publish status.

    If access_token is provided, queries TikTok API directly.
    Otherwise returns cached status from DB.
    """
    conn = _db()
    _ensure_publish_columns(conn)

    # Look up in our DB first — try publish_id, then fall back to job id
    row = conn.execute(
        "SELECT * FROM jobs WHERE publish_id = ? ORDER BY created_at DESC LIMIT 1",
        (publish_id,),
    ).fetchone()

    if not row:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ? ORDER BY created_at DESC LIMIT 1",
            (publish_id,),
        ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Publish ID {publish_id} not found")

    job = dict(row)

    # If we have an access token and it's an API publish, check live status
    if access_token and job.get("publish_method") == "api" and not publish_id.startswith("browser_"):
        try:
            live_status = await _tiktok_check_status(access_token, publish_id)
            new_status = live_status.get("status", job["tiktok_status"])

            # Update DB with fresh status
            conn.execute(
                "UPDATE jobs SET tiktok_status=?, fail_reason=?, post_url=? WHERE publish_id=?",
                (new_status, live_status.get("fail_reason"), live_status.get("post_url"), publish_id),
            )
            conn.commit()

            job["tiktok_status"] = new_status
            job["fail_reason"] = live_status.get("fail_reason")
            job["post_url"] = live_status.get("post_url")
        except Exception as exc:
            logger.warning("[status] TikTok status check failed: %s", exc)

    conn.close()

    return StatusResponse(
        publish_id=publish_id,
        status=job.get("tiktok_status", "unknown"),
        fail_reason=job.get("fail_reason"),
        post_url=job.get("post_url"),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print(f"[vbl-publisher] Starting on {HOST}:{PORT}")
    print(f"[vbl-publisher] Drafts DB: {DRAFTS_DB}")
    print(f"[vbl-publisher] TikTok API: {'configured' if TIKTOK_CLIENT_KEY else 'NOT configured (will use browser fallback)'}")

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
