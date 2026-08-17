"""Browser Timeline Watcher — scrolls TikTok/Instagram feeds to discover trending content.

Long-running worker that:
1. Launches headless Playwright chromium (or ego-browser for X/Twitter)
2. Navigates to TikTok For You / Instagram Explore
3. Scrolls feed, extracts post cards as they appear
4. Deduplicates against corpus DB (by URL or video ID)
5. Inserts new discoveries into corpus via direct SQLite upsert
6. Sleeps between scroll cycles to avoid rate limits
7. Runs continuously or on a cron schedule (--once mode)

Wrapped in lightweight FastAPI for control endpoints (/health, /scan, /stats, /recent, /stop).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

import sys
import os

if not __package__:
    # Direct-script launch (`python services/browser-worker/app.py`): put the
    # parent `services/` dir on the path so the shared helper resolves.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from secure_env import effective_host
else:
    from services.secure_env import effective_host

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("browser-worker")

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.environ.get("VBL_DATA_DIR", os.path.expanduser("~/viral-bench-local/data")), "corpus.db")
SCRAPER_API = os.environ.get("SCRAPER_API_URL", "http://127.0.0.1:8010")
RESEARCH_API = os.environ.get("RESEARCH_API_URL", "http://127.0.0.1:8001")
EGO_BROWSER = os.environ.get("EGO_BROWSER", str(Path.home() / ".local/bin/ego-browser"))
WORKER_PORT = int(os.environ.get("BROWSER_WORKER_PORT", "8012"))
AUTO_ANALYZE = os.environ.get("AUTO_ANALYZE", "true").lower() == "true"

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]

VIEWPORTS = [
    {"width": 1280, "height": 800},
    {"width": 1440, "height": 900},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864},
]

# ── Stats tracking ───────────────────────────────────────────────────────────
class WorkerStats:
    def __init__(self):
        self.total_discovered = 0
        self.total_deduped = 0
        self.by_platform: dict[str, int] = {}
        self.by_hour: dict[str, int] = {}
        self.last_scan_time: str | None = None
        self.scan_count = 0
        self.browser_state = "stopped"
        self.errors: list[str] = []

    def record(self, platform: str, discovered: int, deduped: int):
        self.total_discovered += discovered
        self.total_deduped += deduped
        self.by_platform[platform] = self.by_platform.get(platform, 0) + discovered
        hour = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:00")
        self.by_hour[hour] = self.by_hour.get(hour, 0) + discovered
        self.last_scan_time = datetime.now(timezone.utc).isoformat()
        self.scan_count += 1

    @property
    def dedup_rate(self) -> float:
        total = self.total_discovered + self.total_deduped
        return self.total_deduped / total if total > 0 else 0.0

    def summary(self) -> dict:
        return {
            "total_discovered": self.total_discovered,
            "total_deduped": self.total_deduped,
            "dedup_rate": round(self.dedup_rate, 4),
            "by_platform": self.by_platform,
            "by_hour": dict(sorted(self.by_hour.items())[-24:]),
            "last_scan_time": self.last_scan_time,
            "scan_count": self.scan_count,
            "browser_state": self.browser_state,
        }


stats = WorkerStats()
recent_posts: list[dict[str, Any]] = []
MAX_RECENT = 100

# ── Corpus DB helpers ────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def post_exists(post_id: str, post_url: str) -> bool:
    """Check if a post already exists in the corpus."""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT 1 FROM posts WHERE id = ? OR post_url = ? LIMIT 1",
            (post_id, post_url),
        ).fetchone()
        conn.close()
        return row is not None
    except Exception as e:
        logger.warning("Dedup check failed: %s", e)
        return False


def insert_post(post: dict[str, Any]) -> bool:
    """Insert a post into the corpus. Returns True if inserted, False if skipped."""
    now = datetime.now(timezone.utc).isoformat()
    defaults = {
        "caption": "", "transcript": "", "hook": "",
        "views": 0, "likes": 0, "comments": 0, "shares": 0, "saves": 0,
        "engagement_rate": 0.0, "created_at": now,
    }
    row = {**defaults, **post}
    if "id" not in row or not row["id"]:
        return False

    sql = """
    INSERT INTO posts (
        id, platform, post_url, creator_handle, caption, transcript, hook,
        format, topic, views, likes, comments, shares, saves,
        engagement_rate, published_at, created_at
    ) VALUES (
        :id, :platform, :post_url, :creator_handle, :caption, :transcript, :hook,
        :format, :topic, :views, :likes, :comments, :shares, :saves,
        :engagement_rate, :published_at, :created_at
    ) ON CONFLICT(id) DO NOTHING
    """
    try:
        conn = _get_conn()
        cur = conn.execute(sql, row)
        conn.commit()
        inserted = cur.rowcount > 0
        conn.close()
        return inserted
    except Exception as e:
        logger.error("Insert failed for %s: %s", row.get("id"), e)
        return False


# ── Post extraction helpers ──────────────────────────────────────────────────

def _make_post_id(platform: str, url: str, handle: str = "") -> str:
    """Generate a deterministic post ID from URL or handle+hash."""
    # Try to extract video ID from URL
    tiktok_match = re.search(r'/video/(\d+)', url)
    if tiktok_match:
        return f"tiktok_{tiktok_match.group(1)}"

    ig_match = re.search(r'/(?:p|reel|reels)/([A-Za-z0-9_-]+)', url)
    if ig_match:
        return f"instagram_{ig_match.group(1)}"

    # Fallback: hash the URL
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"{platform}_{h}"


def _parse_count(text: str) -> int:
    """Parse '1.2M', '45K', '1,234' etc. into integer."""
    if not text:
        return 0
    text = text.strip().replace(",", "")
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    for suffix, mult in multipliers.items():
        if text.upper().endswith(suffix):
            try:
                return int(float(text[:-1]) * mult)
            except ValueError:
                return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _compute_engagement(views: int, likes: int, comments: int, shares: int, saves: int) -> float:
    if views <= 0:
        return 0.0
    return round((likes + comments + shares + saves) / views, 6)


def _trigger_analysis(post_url: str, post_id: str) -> None:
    """Fire-and-forget POST to research API to queue VLM analysis."""
    import urllib.request
    try:
        data = json.dumps({"post_url": post_url, "post_id": post_id, "async_mode": True}).encode()
        req = urllib.request.Request(
            f"{RESEARCH_API}/v1/analyze",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        logger.info("Queued VLM analysis for %s", post_id)
    except Exception as e:
        logger.debug("Analysis trigger failed for %s: %s", post_id, e)


# ── TikTok scraper ───────────────────────────────────────────────────────────

TIKTOK_CARD_SELECTORS = [
    'div[data-e2e="recommend_list_item_container"]',
    'div.tiktok-j2a19r-SwiperContent',
    'div[class*="DivItemContainer"]',
    'div[class*="recommend-card"]',
    'div[data-testid="recommend-list-item"]',
]

TIKTOK_LINK_SELECTOR = 'a[href*="/video/"]'


async def scrape_tiktok(page, max_scrolls: int = 10, min_posts: int = 20) -> list[dict[str, Any]]:
    """Scroll TikTok For You page and extract posts via API interception + DOM fallback.

    Strategy: Intercept /api/recommend/item_list responses which contain full structured
    data (aweme_id, stats, author, desc). Fall back to DOM extraction if API yields nothing.
    """
    discovered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    api_items: list[dict[str, Any]] = []

    # Set up API response interception
    async def _capture_response(response):
        if "api/recommend/item_list" in response.url:
            try:
                body = await response.json()
                for item in body.get("itemList", []):
                    api_items.append(item)
            except Exception:
                pass

    page.on("response", _capture_response)

    url = "https://www.tiktok.com/foryou"
    logger.info("Navigating to %s", url)
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await asyncio.sleep(random.uniform(3, 5))

    # Check for login wall — try to dismiss
    content = await page.content()
    if "login" in content.lower() and "sign up" in content.lower():
        logger.warning("TikTok login wall detected — attempting to dismiss")
        try:
            close_btn = page.locator('button:has-text("Log in"), div[class*="close"], button[aria-label="Close"]').first
            if await close_btn.is_visible(timeout=2000):
                await close_btn.click()
                await asyncio.sleep(1)
        except Exception:
            pass

    for scroll_i in range(max_scrolls):
        logger.info("TikTok scroll %d/%d (API items=%d, inserted=%d)",
                     scroll_i + 1, max_scrolls, len(api_items), len(discovered))

        # Process captured API items
        for item in api_items:
            video_id = str(item.get("id") or item.get("aweme_id") or "")
            if not video_id or video_id in seen_ids:
                continue
            seen_ids.add(video_id)

            handle = (item.get("author") or {}).get("uniqueId", "")
            caption = item.get("desc") or ""
            stats_data = item.get("stats") or {}
            views = int(stats_data.get("playCount", 0) or stats_data.get("play_count", 0) or 0)
            likes = int(stats_data.get("diggCount", 0) or stats_data.get("digg_count", 0) or 0)
            comments = int(stats_data.get("commentCount", 0) or stats_data.get("comment_count", 0) or 0)
            shares = int(stats_data.get("shareCount", 0) or stats_data.get("share_count", 0) or 0)
            saves = int(stats_data.get("collectCount", 0) or stats_data.get("collect_count", 0) or 0)

            duration = (item.get("video") or {}).get("duration", 0) or 0
            fmt = "short" if duration < 30 else ("medium" if duration < 90 else "long")

            create_time = item.get("createTime") or 0
            published_at = ""
            if create_time:
                try:
                    published_at = datetime.fromtimestamp(int(create_time), tz=timezone.utc).isoformat()
                except (ValueError, OSError):
                    pass

            post_url = f"https://www.tiktok.com/@{handle}/video/{video_id}" if handle else ""
            post_id = f"tiktok_{video_id}"

            if post_exists(post_id, post_url):
                stats.total_deduped += 1
                continue

            engagement_rate = _compute_engagement(views, likes, comments, shares, saves)

            post = {
                "id": post_id,
                "platform": "tiktok",
                "post_url": post_url,
                "creator_handle": handle,
                "caption": caption,
                "hook": caption.split(".")[0].split("!")[0].split("?")[0][:200] if caption else "",
                "views": views,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "saves": saves,
                "engagement_rate": engagement_rate,
                "format": fmt,
                "topic": "",
                "published_at": published_at,
            }

            if insert_post(post):
                discovered.append(post)
                recent_posts.insert(0, post)
                if len(recent_posts) > MAX_RECENT:
                    recent_posts.pop()
                # Auto-trigger VLM analysis for new TikTok posts
                if AUTO_ANALYZE and post.get("post_url"):
                    _trigger_analysis(post["post_url"], post["id"])
        api_items.clear()

        if len(discovered) >= min_posts:
            logger.info("Reached min_posts=%d, stopping scroll", min_posts)
            break

        # Human-like scroll to trigger more API loads
        scroll_px = random.randint(600, 1200)
        await page.evaluate(f"window.scrollBy(0, {scroll_px})")
        delay = random.uniform(2.0, 5.0)
        await asyncio.sleep(delay)

    # Fallback: if API interception yielded nothing, try DOM extraction
    if not discovered:
        logger.info("API interception yielded 0 posts, falling back to DOM extraction")
        dom_posts = await _tiktok_dom_fallback(page)
        for post in dom_posts:
            if post_exists(post["id"], post.get("post_url", "")):
                stats.total_deduped += 1
                continue
            if insert_post(post):
                discovered.append(post)
                recent_posts.insert(0, post)
                if len(recent_posts) > MAX_RECENT:
                    recent_posts.pop()

    page.remove_listener("response", _capture_response)
    return discovered


async def _tiktok_dom_fallback(page) -> list[dict[str, Any]]:
    """Fallback: extract posts from visible DOM when API interception fails."""
    cards_data = await page.evaluate("""
    () => {
        const results = [];
        const feedVideos = document.querySelectorAll('[data-e2e="feed-video"]');
        for (const card of feedVideos) {
            const userLink = card.querySelector('a[href*="/@"]');
            const handle = userLink ? userLink.getAttribute('href').split('/@')[1].split('/')[0] : '';
            const descEl = card.querySelector('[data-e2e="browse-video-desc"]') ||
                          card.querySelector('span[class*="desc"]');
            const caption = descEl ? descEl.textContent.trim() : '';
            const img = card.querySelector('img');
            const imgAlt = img ? img.alt : '';
            const cardId = card.id || '';
            results.push({
                cardId: cardId,
                handle: handle,
                caption: caption || imgAlt,
            });
        }
        return results;
    }
    """)

    posts = []
    for card in cards_data:
        handle = card.get("handle", "")
        caption = card.get("caption", "")
        card_id = card.get("cardId", "")
        # Without a real video ID from the API, we use a hash-based ID
        post_id = f"tiktok_dom_{hashlib.md5((handle + caption).encode()).hexdigest()[:12]}"
        post_url = f"https://www.tiktok.com/@{handle}" if handle else ""

        posts.append({
            "id": post_id,
            "platform": "tiktok",
            "post_url": post_url,
            "creator_handle": handle,
            "caption": caption,
            "hook": caption.split(".")[0].split("!")[0].split("?")[0][:200] if caption else "",
            "views": 0, "likes": 0, "comments": 0, "shares": 0, "saves": 0,
            "engagement_rate": 0.0,
            "format": "short",
            "topic": "",
            "published_at": "",
        })
    return posts


# ── Instagram scraper ────────────────────────────────────────────────────────

async def scrape_instagram(page, max_scrolls: int = 10, min_posts: int = 20) -> list[dict[str, Any]]:
    """Scroll Instagram Explore page and extract visible post cards."""
    discovered: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    url = "https://www.instagram.com/explore/"
    logger.info("Navigating to %s", url)
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await asyncio.sleep(random.uniform(3, 5))

    # Check for login wall
    content = await page.content()
    if "log in" in content.lower() or "sign up" in content.lower():
        logger.warning("Instagram login wall detected — limited scraping possible")

    for scroll_i in range(max_scrolls):
        logger.info("Instagram scroll %d/%d (found %d so far)", scroll_i + 1, max_scrolls, len(discovered))

        cards_data = await page.evaluate("""
        () => {
            const results = [];
            // Instagram explore grid items
            const links = document.querySelectorAll('a[href*="/p/"], a[href*="/reel/"], a[href*="/reels/"]');
            
            for (const link of links) {
                const href = link.href || link.getAttribute('href') || '';
                if (!href.match(/\\/(p|reel|reels)\\/[A-Za-z0-9_-]+/)) continue;
                
                const img = link.querySelector('img');
                const alt = img ? img.alt : '';
                
                // Try to find metrics
                const spans = link.querySelectorAll('span');
                let views_text = '0';
                for (const s of spans) {
                    const t = s.textContent.trim();
                    if (t.match(/^[\\d,.]+[KMB]?$/)) {
                        views_text = t;
                        break;
                    }
                }
                
                results.push({
                    url: href.startsWith('http') ? href : 'https://www.instagram.com' + href,
                    caption: alt,
                    views_text: views_text,
                });
            }
            return results;
        }
        """)

        for card in cards_data:
            post_url = card.get("url", "")
            if not post_url or post_url in seen_urls:
                continue
            seen_urls.add(post_url)

            caption = card.get("caption", "")
            views = _parse_count(card.get("views_text", "0"))
            post_id = _make_post_id("instagram", post_url)

            if post_exists(post_id, post_url):
                stats.total_deduped += 1
                continue

            # Determine format from URL
            fmt = "reel" if "/reel" in post_url else "post"

            post = {
                "id": post_id,
                "platform": "instagram",
                "post_url": post_url,
                "creator_handle": "",
                "caption": caption,
                "hook": caption.split(".")[0].split("!")[0].split("?")[0][:200] if caption else "",
                "views": views,
                "likes": 0,
                "comments": 0,
                "shares": 0,
                "saves": 0,
                "engagement_rate": 0.0,
                "format": fmt,
                "topic": "",
                "published_at": "",
            }

            if insert_post(post):
                discovered.append(post)
                recent_posts.insert(0, post)
                if len(recent_posts) > MAX_RECENT:
                    recent_posts.pop()
                # Auto-trigger VLM analysis for new Instagram posts
                if AUTO_ANALYZE and post.get("post_url"):
                    _trigger_analysis(post["post_url"], post["id"])

        if len(discovered) >= min_posts:
            break

        scroll_px = random.randint(500, 1000)
        await page.evaluate(f"window.scrollBy(0, {scroll_px})")
        delay = random.uniform(3.0, 6.0)  # Instagram is stricter
        await asyncio.sleep(delay)

    return discovered


# ── Browser management ───────────────────────────────────────────────────────

class BrowserManager:
    """Manages Playwright browser lifecycle with stealth settings."""

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._running = False

    async def start(self):
        """Launch headless browser with stealth config."""
        from playwright.async_api import async_playwright

        self.playwright = await async_playwright().start()
        viewport = random.choice(VIEWPORTS)
        ua = random.choice(USER_AGENTS)

        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )

        self.context = await self.browser.new_context(
            viewport={"width": viewport["width"], "height": viewport["height"]},  # type: ignore[arg-type]
            user_agent=ua,
            locale="en-US",
            timezone_id="America/New_York",
            permissions=[],
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "sec-ch-ua": '"Chromium";v="125", "Not A(Brand";v="24"',
                "sec-ch-ua-platform": '"macOS"',
            },
        )

        # Stealth: override navigator.webdriver
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            window.chrome = { runtime: {} };
        """)

        self.page = await self.context.new_page()
        self._running = True
        stats.browser_state = "running"
        logger.info("Browser started (viewport=%s, ua=%.40s...)", viewport, ua)

    async def stop(self):
        """Gracefully shut down browser."""
        self._running = False
        stats.browser_state = "stopping"
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            logger.warning("Error during browser shutdown: %s", e)
        finally:
            self.page = None
            self.context = None
            self.browser = None
            self.playwright = None
            stats.browser_state = "stopped"
            logger.info("Browser stopped")

    @property
    def running(self) -> bool:
        return self._running and self.page is not None

    async def ensure_running(self):
        """Restart browser if it crashed."""
        if not self.running:
            logger.info("Browser not running, starting...")
            await self.start()


browser_mgr = BrowserManager()


# ── Scan orchestration ───────────────────────────────────────────────────────

async def run_scan(platform: str = "tiktok", max_scrolls: int = 10, min_posts: int = 20) -> dict[str, Any]:
    """Execute a single scan cycle."""
    result = {"platform": platform, "discovered": 0, "deduped_before": stats.total_deduped, "errors": []}

    try:
        await browser_mgr.ensure_running()
        page = browser_mgr.page

        if platform == "tiktok":
            posts = await scrape_tiktok(page, max_scrolls=max_scrolls, min_posts=min_posts)
        elif platform == "instagram":
            posts = await scrape_instagram(page, max_scrolls=max_scrolls, min_posts=min_posts)
        else:
            result["errors"].append(f"Unknown platform: {platform}")
            return result

        result["discovered"] = len(posts)
        result["deduped_after"] = stats.total_deduped
        stats.record(platform, len(posts), stats.total_deduped - result["deduped_before"])

    except Exception as e:
        logger.error("Scan error (%s): %s", platform, e, exc_info=True)
        result["errors"].append(str(e))
        stats.errors.append(f"{datetime.now(timezone.utc).isoformat()}: {e}")
        # Attempt browser restart on next call
        await browser_mgr.stop()

    return result


# ── Continuous loop ──────────────────────────────────────────────────────────

_loop_task: asyncio.Task | None = None
_loop_stop_event = asyncio.Event()


async def continuous_loop(platform: str = "tiktok", max_scrolls: int = 10, min_posts: int = 20, interval: float = 300):
    """Run scan cycles continuously with sleep between them."""
    logger.info("Starting continuous loop (platform=%s, interval=%ds)", platform, interval)
    _loop_stop_event.clear()

    while not _loop_stop_event.is_set():
        try:
            result = await run_scan(platform, max_scrolls, min_posts)
            logger.info("Scan complete: %d new posts discovered", result["discovered"])
        except Exception as e:
            logger.error("Loop scan error: %s", e, exc_info=True)

        try:
            await asyncio.wait_for(_loop_stop_event.wait(), timeout=interval)
            break  # Stop event was set
        except asyncio.TimeoutError:
            pass  # Normal: interval elapsed, do another scan

    logger.info("Continuous loop stopped")


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Browser worker starting up")
    yield
    logger.info("Browser worker shutting down")
    _loop_stop_event.set()
    await browser_mgr.stop()


app = FastAPI(title="Browser Timeline Watcher", lifespan=lifespan)


class ScanRequest(BaseModel):
    platform: str = Field(default="tiktok", pattern="^(tiktok|instagram)$")
    max_scrolls: int = Field(default=10, ge=1, le=100)
    min_posts: int = Field(default=20, ge=1, le=500)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "browser-worker",
        "browser_state": stats.browser_state,
        "last_scan_time": stats.last_scan_time,
        "total_discovered": stats.total_discovered,
        "scan_count": stats.scan_count,
    }


@app.post("/scan")
async def trigger_scan(req: ScanRequest):
    """Trigger an immediate scan cycle."""
    result = await run_scan(req.platform, req.max_scrolls, req.min_posts)
    return result


@app.get("/stats")
async def get_stats():
    return stats.summary()


@app.get("/recent")
async def get_recent(limit: int = 20):
    return {"count": len(recent_posts), "posts": recent_posts[:limit]}


@app.post("/stop")
async def stop_worker():
    """Gracefully stop the browser and worker loop."""
    _loop_stop_event.set()
    await browser_mgr.stop()
    return {"status": "stopped"}


# ── CLI entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Browser Timeline Watcher")
    parser.add_argument("--platform", choices=["tiktok", "instagram"], default="tiktok")
    parser.add_argument("--max-scrolls", type=int, default=10)
    parser.add_argument("--min-posts", type=int, default=20)
    parser.add_argument("--once", action="store_true", help="Single scan then exit")
    parser.add_argument("--interval", type=float, default=300, help="Seconds between scans (continuous mode)")
    parser.add_argument("--port", type=int, default=WORKER_PORT, help="API port")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server instead of CLI scan")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        logger.info("Starting browser-worker API on port %d", args.port)
        uvicorn.run(app, host=effective_host("BROWSER_WORKER_HOST"), port=args.port)
        return

    # CLI mode
    async def cli_run():
        await browser_mgr.start()
        try:
            if args.once:
                result = await run_scan(args.platform, args.max_scrolls, args.min_posts)
                print(f"Discovered: {result['discovered']} posts")
                if result.get("errors"):
                    print(f"Errors: {result['errors']}")
            else:
                await continuous_loop(
                    platform=args.platform,
                    max_scrolls=args.max_scrolls,
                    min_posts=args.min_posts,
                    interval=args.interval,
                )
        finally:
            await browser_mgr.stop()

    asyncio.run(cli_run())


if __name__ == "__main__":
    main()
