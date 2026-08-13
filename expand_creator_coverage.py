#!/usr/bin/env python3
"""Expand VLM analysis across ALL creators (not just top 500).
Samples from each creator to ensure broad niche coverage."""
import asyncio
import base64
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx

DB_PATH = Path.home() / "viral-bench-local" / "data" / "corpus.db"
VIDEO_DIR = Path.home() / "viral-bench-local" / "data" / "videos"
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


def extract_frames(video_path: str, num_frames=4) -> list[str]:
    """Extract evenly-spaced frames from a video file."""
    frame_dir = tempfile.mkdtemp()
    try:
        # Check if file has video stream (skip audio-only)
        check = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10
        )
        if "video" not in check.stdout:
            return []
        
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=10
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 10.0
        
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
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        
        # Find JSON object boundaries if there's preamble text
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            cleaned = cleaned[first_brace:last_brace + 1]
        
        return json.loads(cleaned)
    
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        print(f"Raw response (first 500 chars): {raw[:500] if raw else 'empty'}")
        print(f"Cleaned (first 500 chars): {cleaned[:500] if cleaned else 'empty'}")
        return None
    except Exception as e:
        print(f"Analysis error: {e}")
        return None


async def expand_coverage(posts_per_creator=20, max_concurrent_downloads=5, max_concurrent_analyses=3):
    """Analyze top posts from EACH creator to maximize niche coverage."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    cursor = conn.cursor()
    
    # Get all creators
    creators = cursor.execute("""
        SELECT creator_handle, COUNT(*) as total
        FROM posts
        GROUP BY creator_handle
        ORDER BY total DESC
    """).fetchall()
    
    print(f"📊 Expanding VLM coverage across {len(creators)} creators")
    print(f"   Sampling top {posts_per_creator} posts per creator")
    print(f"   Parallelism: {max_concurrent_downloads} downloads, {max_concurrent_analyses} analyses\n")
    
    stats = {"downloaded": 0, "analyzed": 0, "download_fail": 0, "analyze_fail": 0}
    start_time = time.time()
    total_processed = 0
    
    # Semaphores for parallelism control
    download_sem = asyncio.Semaphore(max_concurrent_downloads)
    analyze_sem = asyncio.Semaphore(max_concurrent_analyses)
    
    async def download_task(post_url: str, post_id: str) -> str | None:
        """Parallel download wrapper."""
        async with download_sem:
            return await asyncio.to_thread(download_video, post_url, post_id)
    
    async def analyze_task(video_path: str) -> dict | None:
        """Parallel analysis wrapper."""
        async with analyze_sem:
            return await analyze_with_ollama(video_path)
    
    for creator_idx, (creator, total_posts) in enumerate(creators, 1):
        print(f"\n👤 [{creator_idx}/{len(creators)}] {creator} ({total_posts} posts)")
        
        # Get top posts for this creator (by likes, not yet analyzed)
        posts = cursor.execute("""
            SELECT id, platform, post_url, likes, video_path, vlm_analysis
            FROM posts 
            WHERE creator_handle = ?
            ORDER BY likes DESC
            LIMIT ?
        """, (creator, posts_per_creator * 2)).fetchall()
        
        creator_analyzed = 0
        pending_tasks = []
        
        for post in posts:
            post_id, platform, post_url, likes, video_path, vlm_analysis = post
            
            # Skip if already has new-format analysis
            if vlm_analysis:
                try:
                    analysis = json.loads(vlm_analysis)
                    if "hook_type" in analysis:
                        continue
                except:
                    pass
            
            # Queue download task
            if not video_path or not os.path.exists(video_path):
                pending_tasks.append({
                    "type": "download",
                    "post_id": post_id,
                    "post_url": post_url,
                    "likes": likes,
                })
            else:
                # Already has video, queue analysis
                pending_tasks.append({
                    "type": "analyze",
                    "post_id": post_id,
                    "video_path": video_path,
                    "likes": likes,
                })
            
            if len(pending_tasks) >= posts_per_creator:
                break
        
        # Batch downloads in parallel
        download_tasks = [t for t in pending_tasks if t["type"] == "download"]
        if download_tasks:
            download_results = await asyncio.gather(
                *[download_task(t["post_url"], t["post_id"]) for t in download_tasks],
                return_exceptions=True
            )
            for task, result in zip(download_tasks, download_results):
                if isinstance(result, Exception) or result is None:
                    stats["download_fail"] += 1
                    print(f"   ❌ Download failed: {task['post_id']}")
                    task["failed"] = True
                    continue
                stats["downloaded"] += 1
                task["video_path"] = result
                cursor.execute("UPDATE posts SET video_path = ? WHERE id = ?", (result, task["post_id"]))
        
        # Batch VLM analyses in parallel (downloads + already-downloaded)
        analysis_tasks = []
        for task in pending_tasks:
            if task.get("failed"):
                continue
            vp = task.get("video_path")
            if vp:
                analysis_tasks.append(task)
        
        if analysis_tasks:
            analysis_results = await asyncio.gather(
                *[analyze_task(t.get("video_path") or t.get("video_path_orig")) for t in analysis_tasks],
                return_exceptions=True
            )
            
            for task, analysis in zip(analysis_tasks, analysis_results):
                if isinstance(analysis, Exception) or analysis is None:
                    stats["analyze_fail"] += 1
                    print(f"   ❌ Analysis failed: {task['post_id']}")
                    continue
                
                stats["analyzed"] += 1
                creator_analyzed += 1
                total_processed += 1
                now = datetime.now().isoformat()
                
                analysis_json = json.dumps(analysis)
                cursor.execute("""
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
                    task["post_id"]
                ))
                
                likes_str = f"{task['likes']:,}" if task["likes"] else "0"
                print(f"   ✅ {task['post_id']} ({likes_str} likes) | Hook: {analysis.get('hook_type', 'N/A')[:50]}")
        
        conn.commit()
        
        # Summary for this creator
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"   📈 Analyzed {creator_analyzed} posts | Global: {stats['analyzed']} analyzed, {rate:.2f}/s")
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"COVERAGE EXPANSION COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Analyzed: {stats['analyzed']}")
    print(f"  Download failures: {stats['download_fail']}")
    print(f"  Analysis failures: {stats['analyze_fail']}")
    
    conn.close()


if __name__ == "__main__":
    asyncio.run(expand_coverage(posts_per_creator=20))
