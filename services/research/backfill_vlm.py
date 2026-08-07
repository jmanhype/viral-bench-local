"""Batch backfill VLM analysis for all unanalyzed corpus posts.

Usage:
    python -m services.research.backfill_vlm [--limit N] [--concurrency N] [--dry-run]

Runs as a standalone script against the research API at localhost:8001.
Respects rate limits with configurable delay between requests.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("backfill-vlm")

DB_PATH = Path("/tmp/vbl-corpus/corpus.db")
API_URL = "http://127.0.0.1:8001"


def get_unanalyzed(limit: int = 0) -> list[dict]:
    """Fetch posts that need VLM analysis."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT id, platform, post_url FROM posts
        WHERE (vlm_analyzed_at IS NULL OR vlm_analyzed_at = '')
        AND post_url LIKE '%tiktok.com/%video/%'
        ORDER BY created_at DESC
    """
    if limit > 0:
        sql += f" LIMIT {limit}"
    rows = [dict(r) for r in conn.execute(sql).fetchall()]
    conn.close()
    return rows


async def analyze_one(client: httpx.AsyncClient, post: dict, sem: asyncio.Semaphore) -> dict:
    """Submit one post for synchronous analysis. Returns result dict."""
    async with sem:
        try:
            resp = await client.post(
                f"{API_URL}/v1/analyze",
                json={
                    "post_url": post["post_url"],
                    "post_id": post["id"],
                    "async_mode": False,
                },
                timeout=300,
            )
            data = resp.json()
            status = data.get("status", "unknown")
            hook = ""
            if data.get("analysis"):
                hook = data["analysis"].get("hook_type", "")[:60]
            logger.info("[%s] %s — %s", status, post["id"], hook or "(no hook)")
            return {"id": post["id"], "status": status, "ok": status == "completed"}
        except Exception as e:
            logger.error("[error] %s — %s", post["id"], e)
            return {"id": post["id"], "status": "error", "ok": False}


async def main(limit: int = 0, concurrency: int = 2, delay: float = 5.0, dry_run: bool = False):
    posts = get_unanalyzed(limit)
    total = len(posts)
    logger.info("Found %d posts to analyze (concurrency=%d, delay=%.1fs)", total, concurrency, delay)

    if dry_run:
        for p in posts[:10]:
            print(f"  Would analyze: {p['id']} — {p['post_url']}")
        if total > 10:
            print(f"  ... and {total - 10} more")
        return

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    failed = 0
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        # Process in batches with delay between each batch
        batch_size = concurrency
        for i in range(0, total, batch_size):
            batch = posts[i : i + batch_size]
            tasks = [analyze_one(client, post, sem) for post in batch]
            results = await asyncio.gather(*tasks)

            for r in results:
                if r["ok"]:
                    completed += 1
                else:
                    failed += 1

            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed - failed) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d done, %d failed (%.1f/min, ETA %.0f min)",
                completed + failed, total, failed, rate * 60, eta / 60,
            )

            # Delay between batches to avoid rate limiting
            if i + batch_size < total:
                await asyncio.sleep(delay)

    elapsed = time.time() - start_time
    logger.info(
        "Backfill complete: %d succeeded, %d failed out of %d in %.0f min",
        completed, failed, total, elapsed / 60,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill VLM analysis for corpus posts")
    parser.add_argument("--limit", type=int, default=0, help="Max posts to process (0=all)")
    parser.add_argument("--concurrency", type=int, default=2, help="Parallel requests")
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds between batches")
    parser.add_argument("--dry-run", action="store_true", help="List posts without analyzing")
    args = parser.parse_args()

    asyncio.run(main(limit=args.limit, concurrency=args.concurrency, delay=args.delay, dry_run=args.dry_run))
