# Content Critique Endpoint Spec

## Goal
Add a POST /v1/critique endpoint to the research API that analyzes TikTok accounts, scripts, or video concepts — matching LightReel's content critique feature.

## Implementation

### New endpoint: POST /v1/critique
Request body:
```json
{
  "type": "account" | "script" | "video_concept",
  "content": "handle name OR script text OR concept description",
  "context": "optional additional context (niche, goals, target audience)"
}
```

Response:
```json
{
  "score": 72,
  "strengths": ["strong hook in first 2 seconds", "clear CTA"],
  "weaknesses": ["no pattern interrupt after hook", "CTA too generic"],
  "recommendations": ["add visual surprise at 3s mark", "use specific number in CTA"],
  "comparable_posts": [{"url": "...", "hook": "...", "views": 1200000}],
  "evidence_count": 5
}
```

### Logic
1. For `type=account`: scrape profile via scraper API, get recent videos, analyze patterns against corpus
2. For `type=script`: search corpus for similar hooks/formats, compare structure
3. For `type=video_concept`: search corpus for comparable content, evaluate novelty

Use existing `search_corpus()` and `call_llm()` functions. Build a critique-specific system prompt that asks for structured JSON output.

### System prompt template
```
You are a TikTok content strategist. Analyze the following {type} against current trending patterns.

Evidence from trending content:
{evidence_json}

Content to critique:
{content}

{context}

Respond in JSON with: score (0-100), strengths (array), weaknesses (array), recommendations (array).
Be specific and actionable. Reference evidence posts by URL when relevant.
```

## Files to modify
- ~/viral-bench-local/services/research/app.py — add /v1/critique endpoint
- Test with real handles/scripts after implementation
