#!/usr/bin/env python3
"""Scrape TikTok creator profiles via yt-dlp (no API dependency).
Downloads metadata + videos for a curated list of high-engagement creators."""

import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DB_PATH = os.path.expanduser("~/viral-bench-local/data/corpus.db")
VIDEO_DIR = Path(os.path.expanduser("~/viral-bench-local/data/videos"))

# Diverse creator list across niches
CREATORS = {
    # Comedy / Skits / Magic
    "zachking": "comedy",
    "khaby.lame": "comedy",
    "wisdm8": "comedy",
    "brittany_broski": "comedy",
    # Dance / Trends
    "charlidamelio": "dance",
    "addisonre": "dance",
    "jasonderulo": "dance",
    # Music / Lip Sync
    "bellapoarch": "music",
    "toniannmusic": "music",
    # Pets / Animals
    "nala_cat": "pets",
    "tuckerbudzyn": "pets",
    "realgrumpycat": "pets",
    # Food / Cooking
    "gordonramsayofficial": "food",
    "babishculinaryuniverse": "food",
    # Fitness / Sports
    "chris.hemsworth": "fitness",
    "pamela_rf": "fitness",
    "blogilates": "fitness",
    # Tech / Education
    "hankgreen": "education",
    "neildegrassetyson": "education",
    # Lifestyle / Vlog
    "emma": "lifestyle",
    "merrelltwins": "lifestyle",
    # Brands (viral marketing)
    "duolingo": "brand",
    "ryanair": "brand",
    "chipotle": "brand",
    "nba": "brand",
    # POV / Storytime
    "noahschnapp": "pov",
    "avani": "pov",
    # Illusions / Effects
    "julianbass": "vfx",
    "corridorcrew": "vfx",
}


def get_existing_ids() -> set:
    conn = sqlite3.connect(DB_PATH)
    try:
        return {r[0] for r in conn.execute("SELECT id FROM posts").fetchall()}
    finally:
        conn.close()


def scrape_creator(handle: str, existing_ids: set) -> list[dict]:
    """Use yt-dlp to get all videos from a creator profile."""
    url = f"https://www.tiktok.com/@{handle}"
    log.info(f"Scraping @{handle}...")

    # First, get metadata only (no download)
    try:
        result = subprocess.run(
            ["yt-dlp", "--flat-playlist", "--dump-json", "--no-update",
             "--socket-timeout", "30", "--no-warnings", url],
            capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        log.warning(f"@{handle} timed out")
        return []

    if not result.stdout.strip():
        log.warning(f"@{handle} returned no data")
        return []

    posts = []
    for line in result.stdout.strip().split("\n"):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        video_id = item.get("id", "")
        post_id = f"tiktok_{video_id}"

        if post_id in existing_ids:
            continue

        stats = item.get("statistics", {}) if isinstance(item.get("statistics"), dict) else {}
        # flat-playlist has limited stats; full metadata comes from --dump-json per video
        view_count = item.get("view_count", 0) or 0
        like_count = item.get("like_count", 0) or 0
        comment_count = item.get("comment_count", 0) or 0
        repost_count = item.get("repost_count", 0) or 0

        total_engagement = like_count + comment_count + repost_count
        engagement_rate = (total_engagement / view_count) if view_count > 0 else 0.0

        caption = item.get("title", "") or item.get("description", "") or ""
        duration = item.get("duration", 0) or 0
        fmt = "short" if duration < 30 else ("medium" if duration < 90 else "long")

        upload_date = item.get("upload_date", "")
        published_at = ""
        if upload_date and len(upload_date) == 8:
            published_at = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

        video_url = item.get("webpage_url", "") or f"https://www.tiktok.com/@{handle}/video/{video_id}"

        posts.append({
            "id": post_id,
            "platform": "tiktok",
            "post_url": video_url,
            "creator_handle": handle,
            "caption": caption,
            "hook": caption.split(".")[0][:200] if caption else "",
            "format": fmt,
            "topic": "",
            "views": int(view_count),
            "likes": int(like_count),
            "comments": int(comment_count),
            "shares": int(repost_count),
            "saves": 0,
            "engagement_rate": round(engagement_rate, 6),
            "published_at": published_at,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    return posts


def insert_posts(conn: sqlite3.Connection, posts: list[dict]) -> int:
    """Insert posts into the database."""
    if not posts:
        return 0

    sql = """INSERT OR IGNORE INTO posts
        (id, platform, post_url, creator_handle, caption, hook, format, topic,
         views, likes, comments, shares, saves, engagement_rate, published_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    inserted = 0
    for p in posts:
        try:
            conn.execute(sql, (
                p["id"], p["platform"], p["post_url"], p["creator_handle"],
                p["caption"], p["hook"], p["format"], p["topic"],
                p["views"], p["likes"], p["comments"], p["shares"], p["saves"],
                p["engagement_rate"], p["published_at"], p["created_at"],
            ))
            inserted += 1
        except Exception as e:
            log.warning(f"Insert failed for {p['id']}: {e}")
    conn.commit()
    return inserted


def main():
    existing_ids = get_existing_ids()
    log.info(f"Existing posts: {len(existing_ids)}")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    total_new = 0

    for i, (handle, niche) in enumerate(CREATORS.items(), 1):
        log.info(f"[{i}/{len(CREATORS)}] @{handle} ({niche})")
        posts = scrape_creator(handle, existing_ids)
        if posts:
            n = insert_posts(conn, posts)
            total_new += n
            existing_ids.update(p["id"] for p in posts)
            log.info(f"  ✅ {len(posts)} found, {n} new inserted (total: {total_new})")
        else:
            log.info(f"  ⏭️  No new posts")

        time.sleep(1)  # Rate limit

    conn.close()
    log.info(f"\nScraping complete: {total_new} new posts added to corpus")
    log.info(f"Total corpus size: {len(existing_ids)}")


if __name__ == "__main__":
    main()
