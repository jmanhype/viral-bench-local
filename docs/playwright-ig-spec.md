# Playwright IG Scraper Spec

## Goal
Replace the broken gallery-dl/instaloader IG scraper with Playwright + network interception.

## Implementation
Modify ~/viral-bench-local/services/scraper/app.py to add a new `_run_playwright_ig()` function that:

1. Launches Chromium with persistent profile at `/tmp/vbl-ig-profile`
2. User logs in manually once (first run)
3. Navigates to post URL
4. Intercepts network responses matching `graphql/query` or `/api/v1/media/`
5. Parses JSON from intercepted responses for metadata
6. Falls back to DOM extraction if no API responses captured
7. Caches results same as existing IG endpoint
8. Returns same ScrapeCreators-compatible format

## Key patterns from ChatGPT research
- Listen for XHR/fetch responses, not DOM
- Use persistent browser context (cookies survive restarts)
- Stop on challenge/checkpoint pages
- Cache aggressively by shortcode
- Download media immediately (CDN URLs expire)
- One active page per request, no parallel pagination

## Files to modify
- ~/viral-bench-local/services/scraper/app.py — replace _run_gallery_dl and _run_instaloader calls with _run_playwright_ig as primary
