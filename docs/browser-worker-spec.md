# Browser Timeline Watcher Spec

## Goal
Build ~/viral-bench-local/services/browser-worker/app.py — a service that scrolls TikTok and Instagram feeds to discover trending content and feed it into the research corpus.

## Context
- Venv: ~/viral-bench-local/.venv/bin/python
- Playwright already installed (1.62.0)
- Corpus DB at /tmp/vbl-corpus/corpus.db (SQLite FTS5)
- Research API at :8001 has POST /v1/seed for inserting posts
- Scraper API at :8010 can enrich discovered URLs with full metadata
- ego-browser at ~/.local/bin/ego-browser (Profile 5, CDP pierce=true) — MANDATORY for X/Twitter, optional for TikTok
- Install deps: uv pip install --python ~/viral-bench-local/.venv/bin/python <pkg>

## Architecture
This is NOT a FastAPI server. It's a long-running worker process that:
1. Launches a headless browser (Playwright chromium)
2. Navigates to TikTok For You page or Instagram Explore
3. Scrolls the feed, extracting post cards as they appear
4. Deduplicates against corpus (by URL or video ID)
5. Enriches new discoveries via scraper API (optional, can be lazy)
6. Inserts into corpus via research API /v1/seed or direct SQLite insert
7. Sleeps between scroll cycles to avoid rate limits
8. Runs continuously or on a cron schedule

## Post Extraction Schema
From each visible post card, extract:
- platform: "tiktok" | "instagram"
- post_url: canonical URL
- creator_handle: @username
- caption/desc: text content
- views/likes/comments/shares: if visible on card
- video_duration: if shown
- music/sound: track name if shown
- discovered_at: timestamp

## Endpoints (lightweight FastAPI wrapper for control)
The worker runs as a background task inside a FastAPI app so we can control it:

### GET /health
Service status, browser state, last scan time, posts discovered count

### POST /scan
Trigger an immediate scan cycle. Body: { "platform": "tiktok"|"instagram", "max_scrolls": 10, "min_posts": 20 }

### GET /stats
Discovery stats: total discovered, by platform, by hour, dedup rate

### GET /recent
Last N discovered posts

### POST /stop
Gracefully stop the browser and worker loop

## TikTok Feed Scraping Strategy
1. Navigate to https://www.tiktok.com/foryou (or /following)
2. Wait for video cards to load
3. Extract visible cards using DOM selectors
4. Scroll down to load more
5. Repeat until max_scrolls or min_posts reached
6. Handle login walls / CAPTCHAs gracefully (log warning, don't crash)

## Instagram Feed Scraping Strategy  
1. Navigate to https://www.instagram.com/explore/
2. Similar scroll-and-extract pattern
3. Instagram is stricter — use stealth settings, random delays

## Browser Stealth
- Use playwright-stealth or similar
- Random viewport sizes
- Human-like scroll speeds (random delays between scrolls)
- Rotate user agents
- Respect rate limits: min 2-5 seconds between scrolls

## Deduplication
Before inserting, check corpus DB:
```sql
SELECT 1 FROM posts WHERE id = ? OR post_url = ? LIMIT 1
```
Skip if exists. Track dedup rate in stats.

## Error Handling
- Browser crash → restart after 30s
- Login wall detected → log warning, pause, retry later
- Rate limited → exponential backoff
- Network error → retry with backoff
- Never crash the service — log and continue

## CLI Mode
Also support running standalone without the API wrapper:
```bash
python services/browser-worker/app.py --platform tiktok --max-scrolls 20 --once
```
`--once` does a single scan cycle and exits (good for cron).

## Verification
After building:
1. Start service, verify /health returns ok
2. Trigger POST /scan with max_scrolls=3
3. Verify posts appear in /recent
4. Verify dedup works (second scan of same content skips existing)
