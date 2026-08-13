#!/usr/bin/env python3
"""VLM Benchmark: Compare Qwen3-VL (local 3090) vs Gemini 2.5 Flash vs Qwen 3.8 Max vs Qwen 3.7 Plus"""

import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

# Sample TikTok URLs for benchmarking
SAMPLE_URLS = [
    "https://www.tiktok.com/@zachking/video/6768504823336815877",
    "https://www.tiktok.com/@zachking/video/141928945859395584",
    "https://www.tiktok.com/@zachking/video/159516700902727680",
]

ANALYSIS_PROMPT = """Analyze this short-form video for viral content research. Respond in valid JSON only with these exact keys:

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
    """Download video via yt-dlp."""
    safe_id = url.split("/")[-1]
    output_path = os.path.join(output_dir, f"{safe_id}.mp4")
    
    if os.path.exists(output_path):
        return output_path
    
    cmd = [
        "yt-dlp",
        "-f", "best[height<=720][filesize<20M]/best[height<=720]/best",
        "-o", output_path,
        "--no-playlist",
        "--quiet",
        url,
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0 and os.path.exists(output_path):
        return output_path
    return None


def extract_frames(video_path: str, num_frames: int = 4) -> list[str]:
    """Extract N frames from video as base64 strings."""
    frame_dir = tempfile.mkdtemp()
    
    # Extract frames
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={num_frames}/10",
        "-vframes", str(num_frames), "-q:v", "2",
        f"{frame_dir}/frame_%02d.jpg"
    ], capture_output=True, check=True)
    
    # Read as base64
    frames = []
    for frame_path in sorted(Path(frame_dir).glob("frame_*.jpg"))[:num_frames]:
        with open(frame_path, "rb") as f:
            frames.append(base64.b64encode(f.read()).decode())
    
    # Cleanup
    import shutil
    shutil.rmtree(frame_dir)
    
    return frames


async def analyze_gemini(video_path: str) -> dict:
    """Analyze with Gemini 2.5 Flash."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "No GEMINI_API_KEY"}
    
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    
    start = time.time()
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
            params={"key": api_key},
            json={
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": "video/mp4", "data": video_b64}},
                        {"text": ANALYSIS_PROMPT},
                    ]
                }],
                "generationConfig": {
                    "maxOutputTokens": 2048,
                    "temperature": 0.3,
                }
            }
        )
        elapsed = time.time() - start
    
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}"}
    
    data = resp.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = parts[0]["text"] if parts else ""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(cleaned)
        result["_model"] = "Gemini 3.5 Flash"
        result["_time_s"] = round(elapsed, 2)
        return result
    except Exception as e:
        return {"error": str(e), "_time_s": round(elapsed, 2)}


async def analyze_ollama(video_path: str) -> dict:
    """Analyze with Qwen3-VL on local 3090."""
    frames = extract_frames(video_path, num_frames=4)
    
    start = time.time()
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen3-vl:30b-a3b-instruct",
                "messages": [{
                    "role": "user",
                    "content": ANALYSIS_PROMPT,
                    "images": frames
                }],
                "stream": False
            }
        )
        elapsed = time.time() - start
    
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}"}
    
    data = resp.json()
    try:
        raw = data["message"]["content"]
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(cleaned)
        result["_model"] = "Qwen3-VL 30B (3090)"
        result["_time_s"] = round(elapsed, 2)
        return result
    except Exception as e:
        return {"error": str(e), "_time_s": round(elapsed, 2)}


async def analyze_modelscope(video_path: str, model: str) -> dict:
    """Analyze with ModelScope (Qwen 3.8 Max or 3.7 Plus)."""
    api_key = os.environ.get("MODELSCOPE_API_KEY")
    if not api_key:
        return {"error": "No MODELSCOPE_API_KEY"}
    
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    
    data_url = f"data:video/mp4;base64,{video_b64}"
    
    start = time.time()
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            "https://api-inference.modelscope.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": data_url}},
                        {"type": "text", "text": ANALYSIS_PROMPT},
                    ],
                }],
                "max_tokens": 2048,
                "temperature": 0.3,
            }
        )
        elapsed = time.time() - start
    
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}"}
    
    data = resp.json()
    try:
        raw = data["choices"][0]["message"]["content"]
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(cleaned)
        result["_model"] = model.split("/")[-1]
        result["_time_s"] = round(elapsed, 2)
        return result
    except Exception as e:
        return {"error": str(e), "_time_s": round(elapsed, 2)}


async def benchmark_video(video_path: str, video_idx: int) -> dict:
    """Run all 4 models on one video."""
    print(f"\n{'='*70}")
    print(f"VIDEO {video_idx}: {Path(video_path).stem}")
    print(f"{'='*70}")
    
    tasks = {
        "gemini": analyze_gemini(video_path),
        "qwen3vl": analyze_ollama(video_path),
        "qwen38max": analyze_modelscope(video_path, "Qwen-Ambassador/Qwen3.8-Max"),
        "qwen37plus": analyze_modelscope(video_path, "Qwen-Ambassador/Qwen3.7-Plus"),
    }
    
    results = await asyncio.gather(*tasks.values())
    results = dict(zip(tasks.keys(), results))
    
    # Print comparison table
    print(f"\n{'Model':<25} {'Time':<8} {'Hook Type':<40}")
    print("-" * 73)
    for key, result in results.items():
        if "error" in result:
            print(f"{key:<25} {'ERROR':<8} {result['error'][:40]}")
        else:
            model = result.get("_model", key)
            time_s = result.get("_time_s", "?")
            hook = result.get("hook_type", "N/A")[:40]
            print(f"{model:<25} {time_s:<8} {hook}")
    
    return {"video": video_path, "results": results}


async def main():
    print("VLM BENCHMARK: Qwen3-VL (3090) vs Gemini 2.5 Flash vs Qwen 3.8 Max vs Qwen 3.7 Plus")
    print("=" * 80)
    
    # Check environment
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY not set")
    if not os.environ.get("MODELSCOPE_API_KEY"):
        print("WARNING: MODELSCOPE_API_KEY not set")
    
    # Download sample videos
    print("\nDownloading sample videos...")
    video_dir = tempfile.mkdtemp()
    videos = []
    
    for url in SAMPLE_URLS[:3]:
        print(f"  {url}...")
        path = download_video(url, video_dir)
        if path:
            videos.append(path)
            print(f"    ✓ Downloaded: {path}")
        else:
            print(f"    ✗ Failed")
    
    if not videos:
        print("ERROR: No videos downloaded")
        return
    
    # Run benchmarks
    all_results = []
    for i, video in enumerate(videos, 1):
        result = await benchmark_video(video, i)
        all_results.append(result)
    
    # Save results
    output_path = "/tmp/vlm_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE - Results saved to {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
