#!/usr/bin/env python3
"""Analyze top 500 new posts (from expanded corpus) with VLM on 3090."""
import asyncio
import base64
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx


def db_retry(func, max_attempts=5, base_delay=1.0):
    """Retry database operations with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                raise

DB_PATH = Path.home() / "viral-bench-local" / "data" / "corpus.db"
VIDEO_DIR = Path.home() / "viral-bench-local" / "data" / "videos"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3-vl:30b-a3b-thinking"

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
        print(f"Frame extraction failed for {video_path}: {e}")
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
        print(f"Download failed for {post_id}: {e}")
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
            print(f"Ollama error: {resp.status_code}")
            return None
        
        raw = resp.json()["message"]["content"]
        
        # Strip thinking tags using regex — build tag strings safely
        import re
        lt = chr(60)  # <
        gt = chr(62)  # >
        tags_to_strip = ["think", "thinking", "thought"]
        for tag in tags_to_strip:
            open_tag = f"{lt}{tag}{gt}"
            close_tag = f"{lt}/{tag}{gt}"
            pattern = re.compile(re.escape(open_tag) + r".*?" + re.escape(close_tag), re.DOTALL)
            raw = pattern.sub("", raw)
        raw = raw.strip()
        
        cleaned = raw.strip()
        # Remove markdown code fences
        if cleaned.startswith("```"):
            # Remove first line (```json or ```)
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        
        # Find JSON object boundaries if there's preamble text
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            cleaned = cleaned[first_brace:last_brace + 1]
        
        return json.loads(cleaned)
    
    except json.JSONDecodeError:
        print(f"JSON parse failed: {raw[:200] if raw else 'empty'}")
        return None
    except Exception as e:
        print(f"Analysis error: {e}")
        return None


async def analyze_top_500():
    """Get top 500 NEW posts (not in original 1,019), download + analyze."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    cursor = conn.cursor()
    
    # Get all posts, ordered by likes (top performers first)
    # Exclude posts that already have VLM analysis with hook_type field
    cursor.execute("""
        SELECT id, platform, post_url, creator_handle, likes, video_path, vlm_analysis
        FROM posts 
        ORDER BY likes DESC
    """)
    
    all_posts = cursor.fetchall()
    
    # Filter to NEW posts only (not in original 1,019 which have vlm_analysis)
    # and not already analyzed with new format
    new_posts = []
    for post in all_posts:
        post_id, platform, post_url, creator, likes, video_path, vlm_analysis = post
        
        # Skip if already has real analysis (with hook_type field)
        if vlm_analysis:
            try:
                analysis = json.loads(vlm_analysis)
                if "hook_type" in analysis:
                    continue  # Already analyzed
            except:
                pass
        
        # This is a NEW post
        new_posts.append({
            "id": post_id,
            "platform": platform,
            "post_url": post_url,
            "creator": creator,
            "likes": likes,
            "video_path": video_path,
        })
        
        if len(new_posts) >= 500:
            break
    
    print(f"Analyzing top {len(new_posts)} new posts")
    print(f"Likes range: {new_posts[0]['likes']:,} to {new_posts[-1]['likes']:,}")
    print()
    
    # Stats
    stats = {"downloaded": 0, "analyzed": 0, "download_fail": 0, "analyze_fail": 0}
    start_time = time.time()
    
    for i, post in enumerate(new_posts, 1):
        post_id = post["id"]
        post_url = post["post_url"]
        likes = post["likes"] or 0
        creator = post["creator"]
        
        # Step 1: Download video
        video_path = post["video_path"]
        if not video_path or not os.path.exists(video_path):
            video_path = download_video(post_url, post_id)
            if not video_path:
                stats["download_fail"] += 1
                elapsed = time.time() - start_time
                print(f"[{i}/{len(new_posts)}] ❌ Download failed: {post_id} (@{creator}, {likes:,} likes) | Stats: {stats} | Elapsed: {elapsed:.0f}s")
                continue
            
            stats["downloaded"] += 1
            
            # Update video_path in DB
            db_retry(lambda: cursor.execute("UPDATE posts SET video_path = ? WHERE id = ?", (video_path, post_id)))
        
        # Step 2: Analyze
        analysis = await analyze_with_ollama(video_path)
        if not analysis:
            stats["analyze_fail"] += 1
            elapsed = time.time() - start_time
            print(f"[{i}/{len(new_posts)}] ❌ Analysis failed: {post_id} | Stats: {stats} | Elapsed: {elapsed:.0f}s")
            continue
        
        stats["analyzed"] += 1
        now = datetime.now().isoformat()
        
        # Store results
        analysis_json = json.dumps(analysis)
        db_retry(lambda: cursor.execute("""
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
        )))
        db_retry(lambda: conn.commit())
        
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(new_posts) - i) / rate if rate > 0 else 0
        
        print(f"[{i}/{len(new_posts)}] ✅ {post_id} (@{creator}, {likes:,} likes) | Hook: {analysis.get('hook_type', 'N/A')[:60]} | Analyzed: {stats['analyzed']} | Rate: {rate:.1f}/s | ETA: {eta:.0f}s ({eta/60:.0f}min)")
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"TOP 500 ANALYSIS COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Analyzed: {stats['analyzed']}")
    print(f"  Download failures: {stats['download_fail']}")
    print(f"  Analysis failures: {stats['analyze_fail']}")
    
    conn.close()


if __name__ == "__main__":
    asyncio.run(analyze_top_500())
