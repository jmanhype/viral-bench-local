"""SQLite FTS5-based UGC corpus store for the Research API.

Provides lexical search over scraped social-media posts using BM25 ranking
weighted by engagement_rate. Serves as the MVP corpus until Qdrant is available.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from services.secure_env import require_secret

logger = logging.getLogger(__name__)

DB_DIR = Path(os.environ.get("VBL_DATA_DIR", os.path.expanduser("~/viral-bench-local/data")))
DB_PATH = DB_DIR / "corpus.db"

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id              TEXT PRIMARY KEY,
    platform        TEXT,
    post_url        TEXT,
    creator_handle  TEXT,
    caption         TEXT DEFAULT '',
    transcript      TEXT DEFAULT '',
    hook            TEXT DEFAULT '',
    format          TEXT,
    topic           TEXT,
    views           INTEGER DEFAULT 0,
    likes           INTEGER DEFAULT 0,
    comments        INTEGER DEFAULT 0,
    shares          INTEGER DEFAULT 0,
    saves           INTEGER DEFAULT 0,
    engagement_rate REAL DEFAULT 0.0,
    published_at    TEXT,
    created_at      TEXT,
    -- VLM video analysis fields (populated asynchronously after discovery)
    vlm_hook        TEXT DEFAULT '',   -- hook type identified by VLM
    vlm_format      TEXT DEFAULT '',   -- visual format (talking head, POV, tutorial, etc.)
    vlm_pacing      TEXT DEFAULT '',   -- pacing/energy description
    vlm_analysis    TEXT DEFAULT '',   -- full VLM analysis JSON
    video_path      TEXT DEFAULT '',   -- local path to downloaded video
    vlm_analyzed_at TEXT DEFAULT ''    -- timestamp of VLM analysis
);

CREATE VIRTUAL TABLE IF NOT EXISTS posts_fts USING fts5(
    caption, transcript, hook, topic,
    content='posts',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync with posts table
CREATE TRIGGER IF NOT EXISTS posts_ai AFTER INSERT ON posts BEGIN
    INSERT INTO posts_fts(rowid, caption, transcript, hook, topic)
    VALUES (new.rowid, new.caption, new.transcript, new.hook, new.topic);
END;

CREATE TRIGGER IF NOT EXISTS posts_ad AFTER DELETE ON posts BEGIN
    INSERT INTO posts_fts(posts_fts, rowid, caption, transcript, hook, topic)
    VALUES ('delete', old.rowid, old.caption, old.transcript, old.hook, old.topic);
END;

CREATE TRIGGER IF NOT EXISTS posts_au AFTER UPDATE ON posts BEGIN
    INSERT INTO posts_fts(posts_fts, rowid, caption, transcript, hook, topic)
    VALUES ('delete', old.rowid, old.caption, old.transcript, old.hook, old.topic);
    INSERT INTO posts_fts(rowid, caption, transcript, hook, topic)
    VALUES (new.rowid, new.caption, new.transcript, new.hook, new.topic);
END;
"""


def _get_conn() -> sqlite3.Connection:
    """Return a connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create the database directory, tables, and FTS index if needed."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = _get_conn()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        logger.info("Corpus DB initialized at %s", DB_PATH)
    finally:
        conn.close()


# ── Upsert ────────────────────────────────────────────────────────────────────

_UPSERT_SQL = """
INSERT INTO posts (
    id, platform, post_url, creator_handle, caption, transcript, hook,
    format, topic, views, likes, comments, shares, saves,
    engagement_rate, published_at, created_at
) VALUES (
    :id, :platform, :post_url, :creator_handle, :caption, :transcript, :hook,
    :format, :topic, :views, :likes, :comments, :shares, :saves,
    :engagement_rate, :published_at, :created_at
)
ON CONFLICT(id) DO UPDATE SET
    platform=excluded.platform,
    post_url=excluded.post_url,
    creator_handle=excluded.creator_handle,
    caption=excluded.caption,
    transcript=excluded.transcript,
    hook=excluded.hook,
    format=excluded.format,
    topic=excluded.topic,
    views=excluded.views,
    likes=excluded.likes,
    comments=excluded.comments,
    shares=excluded.shares,
    saves=excluded.saves,
    engagement_rate=excluded.engagement_rate,
    published_at=excluded.published_at,
    created_at=excluded.created_at
"""

_UPDATE_VLM_SQL = """
UPDATE posts SET
    vlm_hook = :vlm_hook,
    vlm_format = :vlm_format,
    vlm_pacing = :vlm_pacing,
    vlm_analysis = :vlm_analysis,
    video_path = :video_path,
    vlm_analyzed_at = :vlm_analyzed_at
WHERE id = :id
"""

_INSERT_STUB_SQL = """
INSERT OR IGNORE INTO posts (id, platform, post_url, created_at)
VALUES (:id, :platform, :post_url, :created_at)
"""


def update_vlm_analysis(post_id: str, analysis: dict[str, Any]) -> bool:
    """Store VLM analysis results for a post. Creates stub row if needed. Returns True if updated."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    try:
        # Ensure post exists (create stub if analyzed standalone)
        platform = "tiktok" if "tiktok" in post_id else ("instagram" if "instagram" in post_id else "unknown")
        post_url = analysis.get("post_url", "")
        conn.execute(_INSERT_STUB_SQL, {
            "id": post_id,
            "platform": platform,
            "post_url": post_url,
            "created_at": now,
        })

        cur = conn.execute(_UPDATE_VLM_SQL, {
            "id": post_id,
            "vlm_hook": analysis.get("hook_type", ""),
            "vlm_format": analysis.get("visual_format", ""),
            "vlm_pacing": analysis.get("pacing", ""),
            "vlm_analysis": json.dumps(analysis),
            "video_path": analysis.get("video_path", ""),
            "vlm_analyzed_at": now,
        })
        conn.commit()
        return cur.rowcount > 0
    except Exception as e:
        logger.error("VLM update failed for %s: %s", post_id, e)
        return False
    finally:
        conn.close()


def upsert_post(post: dict[str, Any]) -> None:
    """Insert or update a single post record."""
    now = datetime.now(timezone.utc).isoformat()
    defaults = {
        "caption": "", "transcript": "", "hook": "",
        "views": 0, "likes": 0, "comments": 0, "shares": 0, "saves": 0,
        "engagement_rate": 0.0, "created_at": now,
    }
    row = {**defaults, **post}
    # Ensure required fields
    if "id" not in row:
        raise ValueError("Post must have an 'id' field")

    conn = _get_conn()
    try:
        conn.execute(_UPSERT_SQL, row)
        conn.commit()
    finally:
        conn.close()


def upsert_posts(posts: list[dict[str, Any]]) -> int:
    """Batch upsert multiple posts. Returns count inserted."""
    if not posts:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    defaults = {
        "caption": "", "transcript": "", "hook": "",
        "views": 0, "likes": 0, "comments": 0, "shares": 0, "saves": 0,
        "engagement_rate": 0.0, "created_at": now,
    }
    rows = [{**defaults, **p} for p in posts]

    conn = _get_conn()
    try:
        conn.executemany(_UPSERT_SQL, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


# ── Search ────────────────────────────────────────────────────────────────────

_SEARCH_SQL = """
SELECT
    p.id, p.platform, p.post_url, p.creator_handle, p.caption,
    p.transcript, p.hook, p.format, p.topic, p.vlm_hook, p.vlm_format,
    p.views, p.likes, p.comments, p.shares, p.saves,
    p.engagement_rate, p.published_at, p.created_at,
    bm25(posts_fts) AS bm25_score
FROM posts_fts
JOIN posts p ON p.rowid = posts_fts.rowid
WHERE posts_fts MATCH :query
ORDER BY (bm25(posts_fts) * -1) + (p.engagement_rate * 0.5) ASC
LIMIT :limit
"""


def search_posts(query: str, limit: int = 20, creator_handles: list[str] = None) -> list[dict[str, Any]]:
    """Full-text search with BM25 + engagement_rate weighting.

    Returns posts sorted by combined relevance score (lower BM25 = better,
    higher engagement_rate = better).
    
    Args:
        query: Search query text
        limit: Max results to return
        creator_handles: Optional list of creators to filter by
    """
    if not query or not query.strip():
        return []

    # Sanitize query for FTS5 — wrap each token in double quotes for safety
    tokens = query.strip().split()
    fts_query = " OR ".join(f'"{t}"' for t in tokens if t)
    if not fts_query:
        return []

    # Build creator filter if provided
    creator_filter = ""
    params = {"query": fts_query, "limit": limit}
    if creator_handles:
        placeholders = ",".join(f"@h{i}" for i in range(len(creator_handles)))
        creator_filter = f"AND p.creator_handle IN ({placeholders})"
        for i, handle in enumerate(creator_handles):
            params[f"h{i}"] = handle

    sql = _SEARCH_SQL.replace(
        "ORDER BY",
        f"{creator_filter}\nORDER BY"
    )

    conn = _get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d.pop("bm25_score", None)
            results.append(d)
        return results
    except sqlite3.OperationalError as e:
        logger.warning("FTS search failed for query=%r: %s", query, e)
        return []
    finally:
        conn.close()


# ── Seeding from Scraper API ─────────────────────────────────────────────────

SCRAPER_BASE = os.environ.get("SCRAPER_API_URL", "http://127.0.0.1:8010")


def get_scraper_api_key() -> str:
    """Return the configured Scraper API key, failing CLOSED if unset.

    Lazy (not module-level) so importing the corpus module never crashes the
    research service when the key is absent — the failure surfaces only when a
    scraper call actually needs the credential.
    """
    return require_secret("SCRAPER_API_KEY", hint="Set SCRAPER_API_KEY in .env to enable corpus seeding.")


async def seed_from_scraper_api(
    handles: list[str],
    max_per_handle: int = 10,
) -> dict[str, Any]:
    """Fetch recent posts from scraper API and insert into corpus.

    Returns summary of seeding results.
    """
    total_inserted = 0
    errors: list[str] = []
    per_handle: dict[str, int] = {}

    async with httpx.AsyncClient(timeout=60, headers={"x-api-key": get_scraper_api_key()}) as client:
        for handle in handles:
            try:
                url = f"{SCRAPER_BASE}/v3/tiktok/profile/videos"
                resp = await client.get(url, params={
                    "handle": handle,
                    "count": max_per_handle,
                })
                if resp.status_code != 200:
                    errors.append(f"{handle}: HTTP {resp.status_code}")
                    per_handle[handle] = 0
                    continue

                data = resp.json()
                # Scraper API returns { aweme_list: [...] } (ScrapeCreators shape)
                items = data.get("aweme_list") or data.get("itemList") or data.get("items") or data.get("data") or []
                if isinstance(items, dict):
                    items = items.get("list", [])

                posts = []
                for item in items[:max_per_handle]:
                    post = _normalize_tiktok_item(item, handle)
                    if post:
                        posts.append(post)

                count = upsert_posts(posts)
                per_handle[handle] = count
                total_inserted += count
                logger.info("Seeded %d posts from @%s", count, handle)

            except Exception as e:
                errors.append(f"{handle}: {type(e).__name__}: {e}")
                per_handle[handle] = 0
                logger.error("Seed error for @%s: %s", handle, e)

    return {
        "inserted": total_inserted,
        "per_handle": per_handle,
        "errors": errors,
    }


def _normalize_tiktok_item(item: dict, fallback_handle: str) -> dict[str, Any] | None:
    """Convert a raw TikTok API item into our posts schema."""
    try:
        video_id = str(item.get("aweme_id") or item.get("id") or item.get("video", {}).get("id", ""))
        if not video_id:
            return None

        stats = item.get("stats") or item.get("statistics") or {}
        views = int(stats.get("play_count", 0) or stats.get("playCount", 0) or stats.get("view_count", 0) or 0)
        likes = int(stats.get("digg_count", 0) or stats.get("diggCount", 0) or stats.get("like_count", 0) or 0)
        comments = int(stats.get("comment_count", 0) or stats.get("commentCount", 0) or 0)
        shares = int(stats.get("share_count", 0) or stats.get("shareCount", 0) or 0)
        saves = int(stats.get("collect_count", 0) or stats.get("collectCount", 0) or stats.get("save_count", 0) or 0)

        # Engagement rate = (likes + comments + shares + saves) / views
        total_engagement = likes + comments + shares + saves
        engagement_rate = (total_engagement / views) if views > 0 else 0.0

        author = item.get("author") or {}
        handle = (
            author.get("uniqueId")
            or author.get("unique_id")
            or author.get("nickname")
            or fallback_handle
        )

        caption = item.get("desc") or item.get("description") or ""
        # Extract first sentence as hook
        hook = caption.split(".")[0].split("!")[0].split("?")[0][:200] if caption else ""

        # Determine format from video duration or other signals
        duration = item.get("video", {}).get("duration", 0) or 0
        fmt = "short" if duration < 30 else ("medium" if duration < 90 else "long")

        # Published timestamp
        create_time = item.get("createTime") or item.get("create_time") or 0
        published_at = ""
        if create_time:
            try:
                published_at = datetime.fromtimestamp(
                    int(create_time), tz=timezone.utc
                ).isoformat()
            except (ValueError, OSError):
                pass

        post_url = f"https://www.tiktok.com/@{handle}/video/{video_id}"

        return {
            "id": f"tiktok_{video_id}",
            "platform": "tiktok",
            "post_url": post_url,
            "creator_handle": handle,
            "caption": caption,
            "transcript": "",  # Would need separate transcription
            "hook": hook,
            "format": fmt,
            "topic": "",  # Could be enriched later via LLM tagging
            "views": views,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "saves": saves,
            "engagement_rate": round(engagement_rate, 6),
            "published_at": published_at,
        }
    except Exception as e:
        logger.debug("Failed to normalize TikTok item: %s", e)
        return None
