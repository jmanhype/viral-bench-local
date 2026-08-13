#!/usr/bin/env python3
"""Benchmark v2: Improved prompts + thinking variant vs original vs cloud models."""

import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
import shutil

import httpx

SAMPLE_URLS = [
    "https://www.tiktok.com/@zachking/video/6768504823336815877",
    "https://www.tiktok.com/@zachking/video/141928945859395584",
    "https://www.tiktok.com/@zachking/video/159516700902727680",
]

# ── Improved Prompt ──────────────────────────────────────────────────────────
# This prompt is designed to elicit the same depth as Qwen 3.7 Plus
IMPROVED_PROMPT = """You are a senior viral content analyst studying short-form video (TikTok/Reels/Shorts). Your job is to reverse-engineer WHY a video performed well, with the same precision a human analyst would provide.

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

# ── Original prompt (from first benchmark) ────────────────────────────────────
ORIGINAL_PROMPT = """Analyze this short-form video for viral content research. Respond in valid JSON only with these exact keys:

{
  "hook_type": "string — what grabs attention in the first 3 seconds",
  "visual_format": "string — format classification",
  "on_screen_text": ["array of strings — all visible text overlays/captions"],
  "pacing": "string — editing pace description",
  "energy_level": "string — low/medium/high/extreme",
  "audio_style": "string — voiceover/trending sound/music/dialogue/silent",
  "product_visibility": "string — any brands/products visible, or 'none'",
  "why_it_works": "string — specific reasons this video likely performed well",
  "creator_style_notes": "string — distinctive creator patterns worth noting"
}

Be specific and timestamped where relevant. Focus on actionable insights for content creators."""


def download_video(url: str, output_dir: str) -> str | None:
    safe_id = url.split("/")[-1]
    output_path = os.path.join(output_dir, f"{safe_id}.mp4")
    if os.path.exists(output_path):
        return output_path
    cmd = ["yt-dlp", "-f", "best[height<=720]/best", "-o", output_path, "--no-playlist", "--quiet", url]
    subprocess.run(cmd, capture_output=True)
    return output_path if os.path.exists(output_path) else None


def extract_frames(video_path: str, num_frames: int = 4) -> list[str]:
    frame_dir = tempfile.mkdtemp()
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={num_frames}/(10)",
        "-vframes", str(num_frames), "-q:v", "2",
        f"{frame_dir}/frame_%02d.jpg"
    ], capture_output=True)
    frames = sorted(Path(frame_dir).glob("frame_*.jpg"))
    if not frames:
        subprocess.run(["ffmpeg", "-i", video_path, "-vframes", "1", "-q:v", "2", f"{frame_dir}/frame_01.jpg"], capture_output=True)
        frames = [Path(f"{frame_dir}/frame_01.jpg")]
    images = []
    for f in frames[:num_frames]:
        with open(f, "rb") as fh:
            images.append(base64.b64encode(fh.read()).decode())
    shutil.rmtree(frame_dir)
    return images


async def analyze_ollama(video_path: str, model: str, prompt: str, num_frames: int = 4, label: str = "") -> dict:
    frames = extract_frames(video_path, num_frames=num_frames)
    start = time.time()
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert viral content analyst. Always respond with valid JSON. Use timestamps. Name specific frameworks and psychological triggers."},
                    {"role": "user", "content": prompt, "images": frames}
                ],
                "stream": False,
                "options": {"temperature": 0.2, "top_p": 0.9}
            }
        )
        elapsed = time.time() - start
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}", "_time_s": round(elapsed, 2)}
    data = resp.json()
    try:
        raw = data["message"]["content"]
        # Strip thinking tags if present
        if "" in raw:
            raw = raw.split("")[-1].strip()
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(cleaned)
        result["_model"] = label or model
        result["_time_s"] = round(elapsed, 2)
        result["_frames"] = num_frames
        return result
    except Exception as e:
        return {"error": str(e), "_model": label or model, "_time_s": round(elapsed, 2)}


async def benchmark_video(video_path: str, video_idx: int) -> dict:
    print(f"\n{'='*80}")
    print(f"VIDEO {video_idx}: {Path(video_path).stem}")
    print(f"{'='*80}")

    tasks = {
        # Original setup (what we tested before)
        "original_4f": analyze_ollama(video_path, "qwen3-vl:30b-a3b-instruct", ORIGINAL_PROMPT, 4, "Instruct + Original Prompt + 4 frames"),
        # Improved prompt, same model
        "improved_4f": analyze_ollama(video_path, "qwen3-vl:30b-a3b-instruct", IMPROVED_PROMPT, 4, "Instruct + Improved Prompt + 4 frames"),
        # More frames
        "improved_8f": analyze_ollama(video_path, "qwen3-vl:30b-a3b-instruct", IMPROVED_PROMPT, 8, "Instruct + Improved Prompt + 8 frames"),
    }

    results = await asyncio.gather(*tasks.values())
    results = dict(zip(tasks.keys(), results))

    # Print comparison
    print(f"\n{'Model':<50} {'Time':<8} {'Hook':<50}")
    print("-" * 108)
    for key, result in results.items():
        if "error" in result:
            print(f"{key:<50} {'ERR':<8} {str(result['error'])[:50]}")
        else:
            model = result.get("_model", key)[:50]
            time_s = result.get("_time_s", "?")
            hook = result.get("hook_type", "N/A")[:50]
            print(f"{model:<50} {time_s:<8} {hook}")

    return {"video": video_path, "results": results}


async def main():
    print("VLM BENCHMARK v2: Improving Local Qwen3-VL Output Quality")
    print("Testing: prompt engineering, system message, more frames, thinking variant")
    print("=" * 80)

    # Check if thinking model is available
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get("http://localhost:11434/api/tags")
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            print(f"\nAvailable models on 3090: {model_names}")
            if any("thinking" in m for m in model_names):
                print("✅ Thinking variant available!")
            else:
                print("⏳ Thinking variant still pulling...")
        except:
            print("❌ Cannot reach Ollama on 3090")

    # Download videos
    print("\nDownloading sample videos...")
    video_dir = tempfile.mkdtemp()
    videos = []
    for url in SAMPLE_URLS[:3]:
        path = download_video(url, video_dir)
        if path:
            videos.append(path)
            print(f"  ✓ {Path(path).stem}")

    if not videos:
        print("ERROR: No videos downloaded")
        return

    # Run benchmarks
    all_results = []
    for i, video in enumerate(videos, 1):
        result = await benchmark_video(video, i)
        all_results.append(result)

    # Save results
    output_path = "/tmp/vlm_benchmark_v2.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
