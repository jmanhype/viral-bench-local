# TikTok Publisher Service Spec

## Goal
Build ~/viral-bench-local/services/publisher/app.py — FastAPI service on port 8030 that publishes queued drafts to TikTok.

## Context
- MCP server at services/mcp-server/app.py stores drafts in SQLite at /tmp/vbl-drafts/drafts.db
- Viral-Bench calls upsert_slideshow_draft → queue_post via MCP, then publisher picks up queued jobs
- Venv: ~/viral-bench-local/.venv/bin/python
- Install deps with: uv pip install --python ~/viral-bench-local/.venv/bin/python <pkg>

## TikTok Content Posting API v2
Docs: https://developers.tiktok.com/doc/content-posting-api-getting-started

### Photo Carousel Flow
1. POST /v2/post/publish/content/init/ with media_type=PHOTO, source_info images array
2. Each image needs a publicly accessible HTTPS URL (PULL_FROM_URL) or direct upload
3. Rate limit: 6 requests/min/user token
4. Unaudited posts are PRIVATE until TikTok audit passes

### Required OAuth Scopes
- user.info.basic
- video.publish (for video posts)
- photo.publish (for photo carousels)

## Endpoints to Implement

### POST /publish
Accept: { draft_id, account_id, access_token }
- Read draft from SQLite (/tmp/vbl-drafts/drafts.db)
- Upload slide images to TikTok
- Create photo carousel post
- Store publish_id + status back in DB
- Return { publish_id, status, post_url? }

### GET /status/{publish_id}
Check TikTok post publish status via GET /v2/post/publish/status/fetch/

### GET /drafts
List pending drafts from SQLite

### GET /health
Return service status

## Browser Automation Fallback
If TikTok API credentials aren't available, implement Playwright-based fallback:
- Navigate to TikTok Creator Portal (tiktok.com/upload)
- Upload images, fill caption, submit as draft
- Screenshot confirmation

## CLI Script
Create ~/viral-bench-local/scripts/publish-draft.py:
- Takes draft_id as argument
- Calls POST /publish on localhost:8030
- Prints result

## Verification
After building:
1. Start service on port 8030
2. Verify GET /health returns ok
3. Verify GET /drafts returns list from SQLite
