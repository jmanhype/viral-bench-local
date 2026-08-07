# Transcript Extraction Spec

## Goal
Add transcript/caption extraction to the scraper API so research can analyze what creators actually say in videos.

## Implementation

### New endpoint: GET /v2/tiktok/video/transcript
- Accepts `url` query param (TikTok video URL)
- Uses yt-dlp with `--write-auto-sub --sub-lang en --skip-download --sub-format vtt`
- Parses VTT subtitle file into clean text
- Returns `{ url, transcript, language, auto_generated }`
- Cache results in SQLite (same cache table, key by URL + "_transcript")

### Modify existing video endpoint
- Add optional `include_transcript=true` query param to `/v2/tiktok/video`
- When true, fetch transcript alongside metadata
- Add `transcript` field to response

### VTT Parser
```python
def parse_vtt(vtt_content: str) -> str:
    """Strip VTT timestamps and deduplicate lines."""
    lines = []
    seen = set()
    for line in vtt_content.split('\n'):
        line = line.strip()
        if not line or line.startswith('WEBVTT') or '-->' in line or line.isdigit():
            continue
        # Strip HTML tags
        import re
        clean = re.sub(r'<[^>]+>', '', line)
        if clean and clean not in seen:
            seen.add(clean)
            lines.append(clean)
    return ' '.join(lines)
```

### Error handling
- No subtitles available → return `{ transcript: null, reason: "no_subtitles" }`
- Non-English only → return transcript with `language` field set
- yt-dlp fails → return error, don't crash

## Files to modify
- ~/viral-bench-local/services/scraper/app.py — add endpoint + modify existing
- Test with real TikTok URLs after implementation
