"""Video analysis pipeline — download + VLM analysis for discovered posts.

Downloads videos via yt-dlp, sends to Gemini 3.5 Flash for visual
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

# VLM provider: "gemini" (default), "ollama" (local), or "modelscope"
VLM_PROVIDER = os.environ.get("VBL_VLM_PROVIDER", "gemini")

# Gemini config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

# Ollama config (remote 3090 inference)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://3090:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:30b-a3b-instruct")

# ModelScope config (fallback)
MODELSCOPE_API_KEY = os.environ.get("MODELSCOPE_API_KEY", "")
MODELSCOPE_BASE_URL = os.environ.get(
    "MODELSCOPE_BASE_URL", "https://api-inference.modelscope.ai/v1"
)
MODELSCOPE_MODEL = os.environ.get("MODELSCOPE_MODEL", "Qwen-Ambassador/Qwen3.8-Max")
MAX_VIDEO_SIZE_MB = 20  # Skip videos larger than this
YTDLP_PATH = os.environ.get("YTDLP_PATH", "yt-dlp")

# Rate limiter for Gemini free tier (15 RPM)
import time as _time
_gemini_last_call = 0.0
_GEMINI_RPM = int(os.environ.get("GEMINI_RPM", "14"))  # Stay under 15 limit

async def _gemini_rate_limit():
    """Ensure we don't exceed Gemini free tier RPM."""
    global _gemini_last_call
    min_interval = 60.0 / _GEMINI_RPM
    elapsed = _time.monotonic() - _gemini_last_call
    if elapsed < min_interval:
        await asyncio.sleep(min_interval - elapsed)
    _gemini_last_call = _time.monotonic()

# System message for all VLM providers
VLM_SYSTEM_MESSAGE = "You are an expert viral content analyst. Always respond with valid JSON. Use timestamps. Name specific frameworks and psychological triggers."

# Improved analysis prompt — gets Qwen 3.7 Plus-level quality from local Qwen3-VL
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
    """Send video to VLM for visual analysis.

    Supports Gemini (native API with inline base64) and ModelScope
    (OpenAI-compatible with data URLs). Compresses video if > 1MB.
    """

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

    if VLM_PROVIDER == "gemini":
        return await _analyze_with_gemini(video_b64, mime)
    elif VLM_PROVIDER == "ollama":
        return await _analyze_with_ollama(video_b64, mime)
    else:
        return await _analyze_with_modelscope(data_url)


async def _analyze_with_gemini(video_b64: str, mime: str) -> dict[str, Any] | None:
    """Send video to Gemini via native API."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not configured")
        return None

    await _gemini_rate_limit()

    payload = {
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": mime, "data": video_b64}},
                {"text": ANALYSIS_PROMPT},
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 2048,
            "temperature": 0.3,
            "thinkingConfig": {"thinkingBudget": 0},  # Disable thinking to avoid truncation
        }
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                json=payload,
            )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "60"))
                logger.warning("Gemini rate limited, sleeping %ds", retry_after)
                await asyncio.sleep(retry_after)
                return None

            if resp.status_code != 200:
                logger.error("Gemini API error: %d %s", resp.status_code, resp.text[:300])
                return None

            data = resp.json()
            raw = data["candidates"][0]["content"]["parts"][0]["text"]

            # Parse JSON from response
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(cleaned)
            logger.info("Gemini VLM analysis complete")
            return result

    except json.JSONDecodeError as e:
        logger.warning("Gemini response parse failed: %s", e)
        return None
    except Exception as e:
        logger.error("Gemini analysis error: %s", e)
        return None


async def _analyze_with_ollama(video_b64: str, mime: str) -> dict[str, Any] | None:
    """Send video to Ollama for local inference on 3090.
    
    Extracts multiple key frames (start, 1/3, 2/3, end) and sends all
    to Qwen3-VL for temporal video understanding.
    """
    import base64
    import tempfile
    import subprocess as _sp
    
    try:
        # Decode video to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
            tmp_video.write(base64.b64decode(video_b64))
            tmp_video_path = tmp_video.name
        
        # Extract 4 evenly-spaced frames using ffmpeg
        # fps=N/duration gives us N frames spread across the video
        frame_dir = tempfile.mkdtemp()
        _sp.run([
            'ffmpeg', '-y', '-i', tmp_video_path,
            '-vf', 'fps=4/(10)',  # ~4 frames per 10s video (adjusts)
            '-vframes', '4', '-q:v', '2',
            f'{frame_dir}/frame_%02d.jpg'
        ], capture_output=True, check=True)
        
        # If fps approach fails, fall back to single frame
        frames = sorted(Path(frame_dir).glob('frame_*.jpg'))
        if not frames:
            # Single frame fallback
            _sp.run([
                'ffmpeg', '-i', tmp_video_path,
                '-vframes', '1', '-q:v', '2',
                f'{frame_dir}/frame_01.jpg'
            ], capture_output=True, check=True)
            frames = [Path(f'{frame_dir}/frame_01.jpg')]
        
        # Read all frames as base64
        images_b64 = []
        for frame_path in frames[:4]:  # Max 4 frames
            with open(frame_path, 'rb') as f:
                images_b64.append(base64.b64encode(f.read()).decode())
        
        # Clean up temp files
        import os, shutil
        os.unlink(tmp_video_path)
        shutil.rmtree(frame_dir)
        
        logger.info("Sending %d frames to Ollama Qwen3-VL on 3090", len(images_b64))
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": VLM_SYSTEM_MESSAGE,
                        },
                        {
                            "role": "user",
                            "content": ANALYSIS_PROMPT,
                            "images": images_b64
                        }
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2, "top_p": 0.9}
                }
            )
            
            if response.status_code != 200:
                logger.error("Ollama API error: %d %s", response.status_code, response.text[:300])
                return None
            
            data = response.json()
            raw = data.get("message", {}).get("content", "")
            
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            
            result = json.loads(cleaned)
            logger.info("Ollama VLM analysis complete (%d frames)", len(images_b64))
            return result
            
    except json.JSONDecodeError as e:
        logger.warning("Ollama response parse failed: %s", e)
        return None
    except Exception as e:
        logger.error("Ollama analysis error: %s", e)
        return None


async def _analyze_with_modelscope(data_url: str) -> dict[str, Any] | None:
    """Send video to ModelScope/Qwen via OpenAI-compatible API."""
    if not MODELSCOPE_API_KEY:
        logger.error("MODELSCOPE_API_KEY not configured")
        return None

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
                    "temperature": 0.3,
                },
            )

            if resp.status_code != 200:
                logger.error("ModelScope API error: %d %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            raw = data["choices"][0]["message"]["content"]

            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(cleaned)
            logger.info("ModelScope VLM analysis complete")
            return result

    except json.JSONDecodeError as e:
        logger.warning("ModelScope response parse failed: %s", e)
        return None
    except Exception as e:
        logger.error("ModelScope analysis error: %s", e)
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
