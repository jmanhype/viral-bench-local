#!/usr/bin/env python3
"""Fast pass: Analyze all posts that already have video files."""
import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from expand_creator_coverage import analyze_with_ollama, DB_PATH

async def analyze_existing(posts_per_batch=10, max_concurrent=3):
    """Analyze posts that already have downloaded videos."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()
    
    # Find posts with videos but no analysis
    posts = cursor.execute("""
        SELECT id, creator_handle, likes, video_path
        FROM posts
        WHERE video_path IS NOT NULL 
          AND video_path != ''
          AND (vlm_analysis IS NULL OR vlm_analysis = '')
        ORDER BY likes DESC
    """).fetchall()
    
    print(f"📊 Found {len(posts)} posts with videos but no analysis")
    
    if not posts:
        print("✅ All posts with videos are already analyzed!")
        return
    
    stats = {"analyzed": 0, "failed": 0}
    start_time = time.time()
    
    # Process in batches with parallel analysis
    for batch_start in range(0, len(posts), posts_per_batch):
        batch = posts[batch_start:batch_start + posts_per_batch]
        print(f"\n🔄 Batch {batch_start//posts_per_batch + 1}: {len(batch)} posts")
        
        # Process sequentially (GPU can't handle concurrent VLM requests)
        for post in batch:
            post_id, creator, likes, video_path = post
            
            analysis = await analyze_with_ollama(video_path)
            
            if isinstance(analysis, Exception):
                stats["failed"] += 1
                print(f"   ❌ {post_id}: {type(analysis).__name__}: {analysis}")
                continue
            if analysis is None:
                stats["failed"] += 1
                print(f"   ❌ {post_id}: Analysis returned None (likely JSON parse error)")
                continue
            
            stats["analyzed"] += 1
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
                post_id
            ))
            
            likes_str = f"{likes:,}" if likes else "0"
            print(f"   ✅ {post_id} ({likes_str} likes) | {creator} | {analysis.get('hook_type', 'N/A')[:40]}")
        
        conn.commit()
        
        # Progress stats
        elapsed = time.time() - start_time
        rate = stats["analyzed"] / elapsed if elapsed > 0 else 0
        remaining = (len(posts) - stats["analyzed"]) / rate if rate > 0 else 0
        print(f"   📈 Progress: {stats['analyzed']}/{len(posts)} ({rate:.2f}/s, ETA: {remaining/60:.1f}min)")
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"FAST PASS COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Analyzed: {stats['analyzed']}")
    print(f"  Failed: {stats['failed']}")
    
    conn.close()


if __name__ == "__main__":
    asyncio.run(analyze_existing(posts_per_batch=10, max_concurrent=3))
