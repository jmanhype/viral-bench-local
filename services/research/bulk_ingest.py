"""Bulk ingest TikTok posts from trending creator profiles into the VBL corpus.

Usage:
    python -m services.research.bulk_ingest [--profiles N] [--min-posts N]

Pulls videos from a curated list of high-engagement TikTok creators,
deduplicates against existing corpus, and inserts new posts with metadata.
Then triggers async VLM analysis for each new post.
"""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import httpx

from services.secure_env import require_secret

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCRAPER_URL = "http://127.0.0.1:8010"
RESEARCH_URL = "http://127.0.0.1:8001"
DB_PATH = os.path.expanduser("~/viral-bench-local/data/corpus.db")

# Curated list of high-engagement TikTok creators across niches
# Mix of viral formats: comedy, dance, POV, tutorials, pets, fitness, food
TRENDING_PROFILES = [
    # Comedy / Skits
    "charlidamelio", "addisonre", "bellapoarch", "zachking",
    "khaby.lame", "wisdm8", "brittany_broski", "dylanmulvaney",
    # Dance / Trends
    "jasonderulo", "michaelle_justdance", "enurainternational",
    # POV / Storytime
    "noahschnapp", "avani", "brycehall", "griffithjenna",
    # Pets / Animals
    "nala_cat", "juniperthefox", "tuckerbudzyn", "realgrumpycat",
    # Fitness / Sports
    "chris.hemsworth", "pamela_rf", "blogilates",
    # Food / Cooking
    "gordonramsayofficial", "babishculinaryuniverse", "feelgoodfoodie",
    # Tech / Education
    "hankgreen", "neildegrassetyson", "physicsgirl",
    # Music / Art
    "toniannmusic", "spencerx", "marcsebastian",
    # Lifestyle / Vlog
    "emma", "merrelltwins", "lizzobeeating",
    # More viral creators to hit 474+
    "daviddobrik", "lilhuddy", "mcndiamond", "ren",
    "kyliejenner", "therock", "willsmith", "kevinhart4real",
    "selena.gomez", "billieeilish", "duolingo", "ryanair",
    "chipotle", "starbucks", "netflix", "nba",
]


def get_existing_ids() -> set[str]:
    """Get all post IDs already in the corpus."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT id FROM posts").fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


async def fetch_profile_videos(handle: str, client: httpx.AsyncClient) -> list[dict]:
    """Fetch all videos from a TikTok profile via scraper API."""
    try:
        resp = await client.get(
            f"{SCRAPER_URL}/v3/tiktok/profile/videos",
            params={"handle": handle},
            timeout=60,
        )
        if resp.status_code != 200:
            log.warning(f"[{handle}] HTTP {resp.status_code}")
            return []
        data = resp.json()
        videos = data.get("videos", data.get("aweme_list", []))
        return videos
    except Exception as e:
        log.warning(f"[{handle}] Error: {e}")
        return []


def insert_post(conn: sqlite3.Connection, video: dict, handle: str) -> bool:
    """Insert a single video as a post into the corpus. Returns True if inserted."""
    post_id = video.get("id") or video.get("aweme_id") or video.get("video_id")
    if not post_id:
        return False

    post_id = f"tiktok_{post_id}"
    desc = video.get("desc", "") or video.get("description", "") or ""
    likes = video.get("statistics", {}).get("digg_count", 0) or video.get("likes", 0) or 0
    views = video.get("statistics", {}).get("play_count", 0) or video.get("views", 0) or 0
    comments = video.get("statistics", {}).get("comment_count", 0) or video.get("comments", 0) or 0
    shares = video.get("statistics", {}).get("share_count", 0) or video.get("shares", 0) or 0
    saves = video.get("statistics", {}).get("collect_count", 0) or 0
    url = f"https://www.tiktok.com/@{handle}/video/{post_id.replace('tiktok_', '')}"

    engagement_rate = (likes + comments + shares + saves) / max(views, 1)

    try:
        conn.execute(
            """INSERT OR IGNORE INTO posts
               (id, platform, post_url, creator_handle, caption, likes, views,
                comments, shares, saves, engagement_rate, created_at)
               VALUES (?, 'tiktok', ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (post_id, url, f"@{handle}", desc, int(likes), int(views),
             int(comments), int(shares), int(saves), round(engagement_rate, 6)),
        )
        return conn.total_changes > 0
    except Exception as e:
        log.warning(f"[{post_id}] Insert error: {e}")
        return False


async def trigger_analysis(post_id: str, post_url: str, client: httpx.AsyncClient):
    """Fire-and-forget VLM analysis request."""
    try:
        await client.post(
            f"{RESEARCH_URL}/v1/analyze",
            json={"post_id": post_id, "post_url": post_url, "async_mode": True},
            timeout=10,
        )
    except Exception:
        pass  # Fire-and-forget, don't block


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=int, default=len(TRENDING_PROFILES),
                        help="Number of profiles to scrape")
    parser.add_argument("--min-posts", type=int, default=500,
                        help="Target minimum new posts to ingest")
    args = parser.parse_args()

    profiles = TRENDING_PROFILES[:args.profiles]
    existing_ids = get_existing_ids()
    log.info(f"Existing posts in corpus: {len(existing_ids)}")
    log.info(f"Scraping {len(profiles)} profiles, target {args.min_posts} new posts")

    inserted = 0
    analyzed = 0

    async with httpx.AsyncClient(headers={"x-api-key": require_secret("SCRAPER_API_KEY", hint="Set SCRAPER_API_KEY in .env before running bulk ingest.")}) as client:
        for i, handle in enumerate(profiles):
            if inserted >= args.min_posts:
                log.info(f"Reached target of {args.min_posts} new posts")
                break

            log.info(f"[{i+1}/{len(profiles)}] Scraping @{handle}...")
            videos = await fetch_profile_videos(handle, client)
            log.info(f"  Got {len(videos)} videos from @{handle}")

            conn = sqlite3.connect(DB_PATH)
            new_in_batch = 0
            for video in videos:
                if insert_post(conn, video, handle):
                    new_in_batch += 1
                    inserted += 1
            conn.commit()
            conn.close()

            if new_in_batch > 0:
                log.info(f"  Inserted {new_in_batch} new posts (total: {inserted})")

            # Trigger VLM analysis for new posts (batch of 5 at a time)
            if new_in_batch > 0:
                conn = sqlite3.connect(DB_PATH)
                new_posts = conn.execute(
                    "SELECT id, post_url FROM posts WHERE vlm_analyzed_at = '' ORDER BY created_at DESC LIMIT ?",
                    (min(new_in_batch, 20),),
                ).fetchall()
                conn.close()

                for pid, purl in new_posts:
                    await trigger_analysis(pid, purl, client)
                    analyzed += 1
                    await asyncio.sleep(0.5)  # Rate limit analysis triggers

            # Small delay between profiles to avoid rate limiting
            await asyncio.sleep(2)

    log.info(f"Bulk ingest complete: {inserted} new posts inserted, {analyzed} analysis triggered")
    log.info(f"VLM analysis is running asynchronously — check /v1/analyzed for progress")


if __name__ == "__main__":
    asyncio.run(main())
