#!/usr/bin/env python3
"""
Full corpus backfill: download videos + VLM analysis on 3090.
Processes all 1019 posts: downloads missing videos, analyzes with improved prompt.
"""
import asyncio
import base64
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/corpus.db"
VIDEO_DIR = Path("/tmp/vbl-videos")
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3-vl:30b-a3b-instruct"

SYSTEM_MSG = "You are an expert viral content analyst. Always respond with valid JSON. Use timestamps. Name specific frameworks and psychological triggers."

ANALYSIS_PROMPT = """You are a senior viral content analyst studying short-form video (TikTok/Reels/Shorts). Your job is to reverse-engineer WHY a video performed well, with the same precision a human analyst would provide.

Analyze this video and respond with valid JSON only — no markdown, no explanation, just JSON with these exact keys:

{
  "hook_type": "string — Describe the specific hook technique used in the first 0-3 seconds. Name the psychological trigger (curiosity gap, pattern interrupt, shock visual, relatable frustration, direct address question, text overlay hook, etc.). Be specific about what the viewer sees.",
  "hook_timestamp": "string — e.g. '00:00-00:03' with what happens at each beat",
  "visual_format": "string — Use industry terminology. Examples: 'talking head + B-roll', 'POV comedy skit', 'illusion reveal with behind-the-scenes', 'tutorial with satisfying payoff', 'chaotic trick-shot montage', 'single-shot life hack demonstration'",
  "on_screen_text": ["array of strings — ALL visible text overlays, captions, watermarks, and on-screen graphics"],
  "pacing": "string — Describe the editing rhythm with timestamps. e.g. 'Fast (00:00-00:06): quick cuts every 1-2s selling the illusion, then slows (00:07-00:12) for the comedic reveal'. Note transition types.",
  "energy_level": "string — one of: low, medium, high, extreme. Justify briefly.",
  "audio_style": "string — Is it trending sound, original audio, voiceover, dialogue, music, or silent? Name specific sounds if recognizable.",
  "product_visibility": "string — Any brands, products, logos, or sponsored content visible. If none, say 'none'.",
  "why_it_works": "string — Name 2-3 specific viral mechanics at play. Use framework terminology where applicable: curiosity gap, forbidden snack trope, escalating absurdity, satisfying payoff, relatable frustration, pattern interrupt, social proof, etc. Explain the viewer psychology.",
  "retention_triggers": ["array of strings — What keeps viewers watching past the first 3 seconds? e.g. 'will they succeed?', 'what is that object?', 'escalating stakes'"],
  "creator_style_notes": "string — Distinctive patterns: signature edits, recurring formats, brand voice, visual style."
}

Analyze every frame carefully. Pay attention to: camera angles, lighting, subject positioning, text placement, edit timing, and visual effects. Be specific with timestamps."""


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def extract_frames(video_path: str, num_frames=4) -> list[str]:
    """Extract evenly-spaced frames from a video file."""
    frame_dir = tempfile.mkdtemp()
    try:
        # Get video duration
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=10
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 10.0
        
        # Extract frames evenly spaced
        interval = max(duration / (num_frames + 1), 0.5)
        for i in range(num_frames):
            ts = interval * (i + 1)
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
                 "-vframes", "1", "-q:v", "2",
                 f"{frame_dir}/frame_{i:02d}.jpg"],
                capture_output=True, timeout=10
            )
        
        frames = sorted(Path(frame_dir).glob("frame_*.jpg"))
        images = []
        for f in frames:
            with open(f, "rb") as fh:
                images.append(base64.b64encode(fh.read()).decode())
        return images
    except Exception as e:
        logger.warning("Frame extraction failed for %s: %s", video_path, e)
        return []
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def download_video(post_url: str, post_id: str) -> str | None:
    """Download video via yt-dlp. Returns path or None."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = post_id.replace("/", "_").replace("\\", "_")
    output_path = str(VIDEO_DIR / f"{safe_id}.mp4")
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        return output_path
    
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[height<=720]/best", "-o", output_path,
             "--no-playlist", "--quiet", "--no-update", "--socket-timeout", "30",
             post_url],
            capture_output=True, timeout=120
        )
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
            return output_path
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning("Download failed for %s: %s", post_id, e)
    return None


async def analyze_with_ollama(video_path: str) -> dict | None:
    """Send video to Qwen3-VL on 3090 for analysis."""
    frames = extract_frames(video_path, num_frames=4)
    if not frames:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": ANALYSIS_PROMPT, "images": frames}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2, "top_p": 0.9}
                }
            )
        
        if resp.status_code != 200:
            logger.error("Ollama error: %d", resp.status_code)
            return None
        
        raw = resp.json()["message"]["content"]
        
        # Strip thinking tags if present
        think_open = "<" + "think" + ">"
        think_close = "<" + "/think" + ">"
        if think_open in raw and think_close in raw:
            raw = raw.split(think_close)[-1].strip()
        
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        
        return json.loads(cleaned)
    
    except json.JSONDecodeError:
        logger.error("JSON parse failed: %s", raw[:200] if 'raw' in dir() else "empty")
        return None
    except Exception as e:
        logger.error("Analysis error: %s", e)
        return None


async def backfill_all():
    """Download + analyze all posts."""
    db = get_db()
    
    # Get all posts ordered by likes (top performers first)
    posts = db.execute("""
        SELECT id, platform, post_url, creator_handle, likes, video_path
        FROM posts 
        ORDER BY likes DESC
    """).fetchall()
    
    total = len(posts)
    logger.info(f"Backfill: {total} posts to process")
    
    # Stats
    stats = {"downloaded": 0, "analyzed": 0, "download_fail": 0, "analyze_fail": 0, "skipped": 0}
    start_time = time.time()
    
    for i, post in enumerate(posts, 1):
        post_id = post["id"]
        post_url = post["post_url"]
        likes = post["likes"] or 0
        
        # Check if already analyzed with good data
        existing = db.execute(
            "SELECT vlm_analysis FROM posts WHERE id = ?", (post_id,)
        ).fetchone()
        if existing and existing["vlm_analysis"]:
            try:
                data = json.loads(existing["vlm_analysis"])
                # Skip if it has real analysis fields (not just video_path)
                if "hook_type" in data or "visual_format" in data:
                    stats["skipped"] += 1
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Step 1: Download video
        video_path = download_video(post_url, post_id)
        if not video_path:
            stats["download_fail"] += 1
            elapsed = time.time() - start_time
            logger.info(
                f"[{i}/{total}] ❌ Download failed: {post_id} (@{post['creator_handle']}, {likes:,} likes) "
                f"| Stats: {stats} | Elapsed: {elapsed:.0f}s"
            )
            continue
        
        stats["downloaded"] += 1
        
        # Update video_path in DB
        db.execute("UPDATE posts SET video_path = ? WHERE id = ?", (video_path, post_id))
        
        # Step 2: Analyze
        analysis = await analyze_with_ollama(video_path)
        if not analysis:
            stats["analyze_fail"] += 1
            elapsed = time.time() - start_time
            logger.info(
                f"[{i}/{total}] ❌ Analysis failed: {post_id} | Stats: {stats} | Elapsed: {elapsed:.0f}s"
            )
            continue
        
        stats["analyzed"] += 1
        now = datetime.now(timezone.utc).isoformat()
        
        # Store results
        analysis_json = json.dumps(analysis)
        db.execute("""
            UPDATE posts SET 
                vlm_analysis = ?,
                vlm_analyzed_at = ?,
                vlm_hook = ?,
                vlm_format = ?,
                vlm_pacing = ?
            WHERE id = ?
        """, (
            analysis_json,
            now,
            analysis.get("hook_type", ""),
            analysis.get("visual_format", ""),
            analysis.get("pacing", ""),
            post_id
        ))
        db.commit()
        
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate if rate > 0 else 0
        
        logger.info(
            f"[{i}/{total}] ✅ {post_id} (@{post['creator_handle']}, {likes:,} likes) "
            f"| Hook: {analysis.get('hook_type', 'N/A')[:60]} "
            f"| Analyzed: {stats['analyzed']} | "
            f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s ({eta/60:.0f}min)"
        )
    
    # Final stats
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKFILL COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Downloaded: {stats['downloaded']}")
    logger.info(f"  Analyzed: {stats['analyzed']}")
    logger.info(f"  Download failures: {stats['download_fail']}")
    logger.info(f"  Analysis failures: {stats['analyze_fail']}")
    logger.info(f"  Skipped (already done): {stats['skipped']}")
    
    db.close()


if __name__ == "__main__":
    asyncio.run(backfill_all())
