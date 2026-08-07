"""Video analysis pipeline — download + VLM analysis for discovered posts.

Downloads videos via yt-dlp, sends to Qwen3.8-Max (ModelScope) for visual
analysis, and stores structured results in the corpus DB.

Designed to run asynchronously after post discovery — doesn't block the
browser worker's scroll cycle.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIR = Path(os.environ.get("VBL_VIDEO_DIR", "/tmp/vbl-videos"))
MODELSCOPE_API_KEY = os.environ.get("MODELSCOPE_API_KEY", "")
MODELSCOPE_BASE_URL = os.environ.get(
    "MODELSCOPE_BASE_URL", "https://api-inference.modelscope.ai/v1"
)
MODELSCOPE_MODEL = os.environ.get("MODELSCOPE_MODEL", "Qwen-Ambassador/Qwen3.8-Max")
MAX_VIDEO_SIZE_MB = 20  # Skip videos larger than this
YTDLP_PATH = os.environ.get("YTDLP_PATH", "yt-dlp")

# Analysis prompt — structured for consistent parsing
ANALYSIS_PROMPT = """Analyze this short-form video for viral content research. Respond in valid JSON only with these exact keys:

{
  "hook_type": "string — what grabs attention in the first 3 seconds (e.g. 'curiosity-gap teaser', 'shock visual', 'direct address question', 'text overlay hook')",
  "visual_format": "string — format classification (e.g. 'talking head', 'POV', 'tutorial', 'dance', 'skit', 'vlog', 'montage', 'product demo')",
  "on_screen_text": ["array of strings — all visible text overlays/captions"],
  "pacing": "string — editing pace description (e.g. 'rapid cuts every 1-2s', 'slow single take', 'moderate with transitions')",
  "energy_level": "string — low/medium/high/extreme",
  "audio_style": "string — voiceover/trending sound/music/dialogue/silent",
  "product_visibility": "string — any brands/products visible, or 'none'",
  "why_it_works": "string — specific reasons this video likely performed well",
  "creator_style_notes": "string — distinctive creator patterns worth noting"
}

Be specific and timestamped where relevant. Focus on actionable insights for content creators."""


async def download_video(url: str, post_id: str) -> str | None:
    """Download a video via yt-dlp. Returns local path or None on failure."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', post_id)
    output_path = str(VIDEO_DIR / f"{safe_id}.mp4")

    # Skip if already downloaded
    if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
        logger.info("Video already downloaded: %s", output_path)
        return output_path

    cmd = [
        YTDLP_PATH,
        "-f", "best[height<=720][filesize<20M]/best[height<=720]/best",
        "--max-filesize", str(MAX_VIDEO_SIZE_MB * 1024 * 1024),
        "-o", output_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace")[:500]
            logger.warning("yt-dlp failed for %s: %s", url, err_msg)
            return None

        if not Path(output_path).exists():
            # yt-dlp may have used a different extension
            candidates = list(VIDEO_DIR.glob(f"{safe_id}.*"))
            if candidates:
                output_path = str(candidates[0])
            else:
                logger.warning("Download succeeded but no file found for %s", post_id)
                return None

        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info("Downloaded %s (%.1fMB) → %s", post_id, size_mb, output_path)
        return output_path

    except asyncio.TimeoutError:
        logger.warning("yt-dlp timed out for %s", url)
        return None
    except Exception as e:
        logger.error("Download error for %s: %s", url, e)
        return None


async def analyze_with_vlm(video_path: str) -> dict[str, Any] | None:
    """Send video to Qwen3.8-Max for visual analysis via ModelScope API.

    ModelScope supports video input via base64-encoded data URLs in the
    content array of chat messages. Compresses video if > 1MB to avoid
    gateway timeouts.
    """
    if not MODELSCOPE_API_KEY:
        logger.error("MODELSCOPE_API_KEY not configured")
        return None

    import base64

    file_size = Path(video_path).stat().st_size
    if file_size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        logger.warning("Video too large (%.1fMB), skipping VLM analysis", file_size / (1024 * 1024))
        return None

    # Compress if > 1MB to avoid ModelScope gateway timeouts
    compressed_path = video_path
    if file_size > 1_000_000:
        compressed_path = str(Path(video_path).with_suffix(".compressed.mp4"))
        if not Path(compressed_path).exists():
            logger.info("Compressing video (%.1fMB) for VLM upload...", file_size / (1024 * 1024))
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", video_path,
                    "-vf", "scale=480:-2",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "30",
                    "-an",  # Drop audio to reduce size
                    "-movflags", "+faststart",
                    compressed_path,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                if proc.returncode != 0:
                    logger.warning("Compression failed, using original: %s", stderr.decode(errors="replace")[:200])
                    compressed_path = video_path
                else:
                    new_size = Path(compressed_path).stat().st_size
                    logger.info("Compressed %.1fMB → %.1fMB", file_size / (1024 * 1024), new_size / (1024 * 1024))
            except Exception as e:
                logger.warning("Compression error: %s, using original", e)
                compressed_path = video_path

    # Encode video as base64 data URL
    with open(compressed_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    # Determine MIME type
    ext = Path(compressed_path).suffix.lower()
    mime_map = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}
    mime = mime_map.get(ext, "video/mp4")
    data_url = f"data:{mime};base64,{video_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": data_url}},
                {"type": "text", "text": ANALYSIS_PROMPT},
            ],
        }
    ]

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{MODELSCOPE_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {MODELSCOPE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODELSCOPE_MODEL,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.3,  # Lower temp for structured output
                },
            )

            if resp.status_code != 200:
                logger.error("VLM API error: %d %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            raw = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(cleaned)
            logger.info("VLM analysis complete for %s", video_path)
            return result

    except json.JSONDecodeError as e:
        logger.warning("VLM response parse failed: %s", e)
        return None
    except Exception as e:
        logger.error("VLM analysis error: %s", e)
        return None


async def analyze_post(post_url: str, post_id: str) -> dict[str, Any] | None:
    """Full pipeline: download video → VLM analysis → structured result.

    Returns analysis dict ready for corpus.update_vlm_analysis(), or None on failure.
    """
    logger.info("Starting video analysis for %s (%s)", post_id, post_url)

    # Step 1: Download
    video_path = await download_video(post_url, post_id)
    if not video_path:
        logger.warning("Download failed for %s, skipping VLM analysis", post_id)
        return None

    # Step 2: VLM analysis
    analysis = await analyze_with_vlm(video_path)
    if not analysis:
        logger.warning("VLM analysis failed for %s", post_id)
        # Still store the video path even if analysis fails
        return {"video_path": video_path}

    analysis["video_path"] = video_path
    return analysis


# ── Background queue ─────────────────────────────────────────────────────────

_analysis_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
_analysis_running = False


async def _analysis_worker():
    """Background worker that processes video analysis queue."""
    global _analysis_running
    _analysis_running = True
    logger.info("Video analysis worker started")

    while _analysis_running:
        try:
            post_url, post_id = await asyncio.wait_for(
                _analysis_queue.get(), timeout=30
            )
        except asyncio.TimeoutError:
            continue

        try:
            # Import here to avoid circular imports
            from services.research import corpus

            analysis = await analyze_post(post_url, post_id)
            if analysis:
                corpus.update_vlm_analysis(post_id, analysis)
                logger.info("Stored VLM analysis for %s", post_id)
            else:
                logger.warning("Analysis produced no results for %s", post_id)
        except Exception as e:
            logger.error("Analysis worker error for %s: %s", post_id, e)
        finally:
            _analysis_queue.task_done()


def enqueue_analysis(post_url: str, post_id: str) -> bool:
    """Queue a post for async video analysis. Returns False if queue is full."""
    if _analysis_queue.qsize() > 50:
        logger.warning("Analysis queue full, dropping %s", post_id)
        return False
    _analysis_queue.put_nowait((post_url, post_id))
    return True


async def start_analysis_worker():
    """Start the background analysis worker task."""
    asyncio.create_task(_analysis_worker())


async def stop_analysis_worker():
    """Stop the background analysis worker."""
    global _analysis_running
    _analysis_running = False
