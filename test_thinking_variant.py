#!/usr/bin/env python3
"""Compare Qwen3-VL instruct vs thinking variant on the 3090."""
import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_MSG = "You are an expert viral content analyst. Always respond with valid JSON. Use timestamps. Name specific frameworks and psychological triggers."

PROMPT = """You are a senior viral content analyst studying short-form video (TikTok/Reels/Shorts). Your job is to reverse-engineer WHY a video performed well, with the same precision a human analyst would provide.

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
    frame_dir = tempfile.mkdtemp()
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={num_frames}/(10)",
        "-vframes", str(num_frames), "-q:v", "2",
        f"{frame_dir}/frame_%02d.jpg"
    ], capture_output=True)
    frames = sorted(Path(frame_dir).glob("frame_*.jpg"))
    images = []
    for f in frames[:num_frames]:
        with open(f, "rb") as fh:
            images.append(base64.b64encode(fh.read()).decode())
    shutil.rmtree(frame_dir)
    return images


async def test_model(model: str, video_path: str, label: str) -> dict:
    frames = extract_frames(video_path, 4)
    start = time.time()
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": PROMPT, "images": frames}
                ],
                "stream": False,
                "options": {"temperature": 0.2, "top_p": 0.9}
            }
        )
        elapsed = time.time() - start

    raw = resp.json()["message"]["content"]
    # Strip thinking tags — construct tag strings to avoid markdown parsing
    think_open = "<" + "think" + ">"
    think_close = "<" + "/think" + ">"
    thinking = None
    if think_open in raw and think_close in raw:
        thinking = raw.split(think_open)[1].split(think_close)[0].strip()
        raw = raw.split(think_close)[-1].strip()

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(cleaned)

    return {
        "model": label,
        "time": round(elapsed, 1),
        "result": result,
        "thinking_chars": len(thinking) if thinking else 0,
    }


async def main():
    # Use the first video from the benchmark
    video_dir = tempfile.mkdtemp()
    url = "https://www.tiktok.com/@zachking/video/6768504823336815877"
    subprocess.run(["yt-dlp", "-f", "best[height<=720]/best", "-o",
                     f"{video_dir}/test.mp4", "--no-playlist", "--quiet", url])
    video_path = f"{video_dir}/test.mp4"
    if not os.path.exists(video_path):
        print("ERROR: video download failed")
        return

    print("=" * 80)
    print("THINKING vs INSTRUCT: Qwen3-VL on 3090")
    print("=" * 80)

    # Run both sequentially (3090 can only run one at a time)
    results = []
    for model, label in [("qwen3-vl:30b-a3b-instruct", "Instruct"),
                          ("qwen3-vl:30b-a3b-thinking", "Thinking")]:
        r = await test_model(model, video_path, label)
        results.append(r)

    # Print side-by-side comparison
    for r in results:
        res = r["result"]
        print(f"\n{'─'*80}")
        print(f"📊 {r['model']} — {r['time']}s (thinking: {r['thinking_chars']} chars)")
        print(f"{'─'*80}")
        for key in ["hook_type", "hook_timestamp", "visual_format", "pacing",
                     "energy_level", "why_it_works", "retention_triggers", "creator_style_notes"]:
            val = res.get(key, "N/A")
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            print(f"  {key:<25}: {val}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in results:
        print(f"  {r['model']:<12}: {r['time']}s | thinking: {r['thinking_chars']} chars")

    shutil.rmtree(video_dir)


if __name__ == "__main__":
    asyncio.run(main())
