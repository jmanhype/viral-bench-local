#!/usr/bin/env python3
"""Background download: Download top videos per creator."""
import asyncio
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from expand_creator_coverage import download_video, DB_PATH

async def download_top_posts(creator=None, top_n=50, max_concurrent=5):
    """Download top N videos per creator (or all creators)."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    
    if creator:
        creators = [creator]
    else:
        creators = cursor.execute("""
            SELECT DISTINCT creator_handle FROM posts
        """).fetchall()
        creators = [c[0] for c in creators]
    
    print(f"📊 Downloading top {top_n} videos for {len(creators)} creators")
    
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    start_time = time.time()
    
    for creator in creators:
        # Get top posts without videos
        posts = cursor.execute("""
            SELECT id, post_url, likes
            FROM posts
            WHERE creator_handle = ?
              AND (video_path IS NULL OR video_path = '')
            ORDER BY likes DESC
            LIMIT ?
        """, (creator, top_n)).fetchall()
        
        if not posts:
            stats["skipped"] += 1
            continue
        
        print(f"\n👤 {creator}: {len(posts)} videos to download")
        
        # Parallel downloads
        sem = asyncio.Semaphore(max_concurrent)
        async def dl(post):
            async with sem:
                return await asyncio.to_thread(download_video, post[1], post[0])
        
        results = await asyncio.gather(
            *[dl(p) for p in posts],
            return_exceptions=True
        )
        
        creator_downloaded = 0
        for post, video_path in zip(posts, results):
            post_id, post_url, likes = post
            
            if isinstance(video_path, Exception) or video_path is None:
                stats["failed"] += 1
                continue
            
            if not os.path.exists(video_path):
                stats["failed"] += 1
                continue
            
            stats["downloaded"] += 1
            creator_downloaded += 1
            cursor.execute("UPDATE posts SET video_path = ? WHERE id = ?", (video_path, post_id))
        
        conn.commit()
        print(f"   ✅ Downloaded {creator_downloaded}/{len(posts)}")
        
        # Progress
        elapsed = time.time() - start_time
        print(f"   📈 Total: {stats['downloaded']} downloaded, {stats['failed']} failed ({elapsed:.0f}s)")
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"DOWNLOAD COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", help="Only download for this creator")
    parser.add_argument("--top", type=int, default=50, help="Top N videos per creator")
    parser.add_argument("--concurrent", type=int, default=5, help="Parallel downloads")
    args = parser.parse_args()
    
    asyncio.run(download_top_posts(
        creator=args.creator,
        top_n=args.top,
        max_concurrent=args.concurrent
    ))
