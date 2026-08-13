#!/usr/bin/env python3
"""Batch download and analyze videos from top creators."""
import asyncio
import sqlite3
import subprocess
import sys
from pathlib import Path
from curl_cffi import requests as cffi_requests

DB_PATH = Path.home() / "viral-bench-local" / "data" / "corpus.db"
VIDEO_DIR = Path.home() / "viral-bench-local" / "data" / "videos"

def download_video_cffi(url: str, post_id: str) -> str | None:
    """Download video via yt-dlp (original working method)."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(VIDEO_DIR / f"{post_id}.mp4")
    
    if Path(output_path).exists() and Path(output_path).stat().st_size > 10000:
        return output_path
    
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[height<=720]/best", "-o", output_path,
             "--no-playlist", "--quiet", "--no-update", "--socket-timeout", "30",
             url],
            capture_output=True, timeout=120
        )
        if Path(output_path).exists() and Path(output_path).stat().st_size > 10000:
            return output_path
        else:
            # Check stderr for slideshow/unavailable errors
            stderr = result.stderr.decode()
            if 'slideshow' in stderr.lower() or 'image' in stderr.lower():
                print(f"  ⊘ Slideshow (no video)")
            elif 'not available' in stderr.lower() or 'unavailable' in stderr.lower():
                print(f"  ✗ Video unavailable/deleted")
            else:
                print(f"  ✗ Download failed")
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  ✗ Error: {e}")
    return None


async def download_and_update(creator: str, limit: int = 50):
    """Download top videos for a creator."""
    db = sqlite3.connect(str(DB_PATH))
    cursor = db.cursor()
    
    # Get posts without videos
    cursor.execute("""
        SELECT id, post_url, likes
        FROM posts
        WHERE creator_handle = ?
        AND (video_path IS NULL OR video_path = '')
        ORDER BY likes DESC
        LIMIT ?
    """, (creator, limit))
    
    posts = cursor.fetchall()
    print(f"\n📥 Downloading {len(posts)} videos from {creator}...")
    
    success = 0
    for post_id, post_url, likes in posts:
        print(f"  [{likes:,} likes] {post_id}...", end=" ")
        video_path = download_video_cffi(post_url, post_id)
        
        if video_path:
            cursor.execute("UPDATE posts SET video_path = ? WHERE id = ?", 
                         (video_path, post_id))
            db.commit()
            size = Path(video_path).stat().st_size / (1024*1024)
            print(f"✓ {size:.1f}MB")
            success += 1
        else:
            print()
    
    print(f"Downloaded {success}/{len(posts)} videos")
    db.close()
    return success


async def analyze_existing_videos():
    """Run VLM analysis on all videos that don't have analysis yet."""
    from expand_creator_coverage import analyze_with_ollama
    
    db = sqlite3.connect(str(DB_PATH))
    cursor = db.cursor()
    
    # Get videos without analysis
    cursor.execute("""
        SELECT id, video_path, likes, creator_handle
        FROM posts
        WHERE video_path IS NOT NULL AND video_path != ''
        AND (vlm_analysis IS NULL OR vlm_analysis = '')
        ORDER BY likes DESC
    """)
    
    posts = cursor.fetchall()
    print(f"\n🔍 Analyzing {len(posts)} videos...")
    
    success = 0
    failed = 0
    
    for i, (post_id, video_path, likes, creator) in enumerate(posts, 1):
        print(f"[{i}/{len(posts)}] {post_id} ({creator}, {likes:,} likes)...", end=" ")
        
        analysis = await analyze_with_ollama(video_path)
        
        if analysis:
            import json
            cursor.execute("""
                UPDATE posts 
                SET vlm_analysis = ?, vlm_analyzed_at = datetime('now')
                WHERE id = ?
            """, (json.dumps(analysis), post_id))
            db.commit()
            print("✓")
            success += 1
        else:
            print("✗ (audio-only or failed)")
            failed += 1
        
        # Rate limit
        await asyncio.sleep(0.5)
    
    print(f"\nAnalyzed {success}/{len(posts)} videos ({failed} failed)")
    db.close()
    return success


async def main():
    # Download from top 5 creators
    creators = ['charlidamelio', 'khaby.lame', 'duolingo', 'jasonderulo', 'ryanair']
    
    print("=" * 80)
    print("PHASE 1: Download videos")
    print("=" * 80)
    
    for creator in creators:
        await download_and_update(creator, limit=20)
    
    print("\n" + "=" * 80)
    print("PHASE 2: Analyze all existing videos")
    print("=" * 80)
    
    await analyze_existing_videos()
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
