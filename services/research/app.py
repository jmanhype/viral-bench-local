"""Lightreel-compatible Research API — local replacement.

Implements: POST /v1/chat
Request:  { question: str, response_fields?: { name: { type, description } } }
Response: { conversationId: str, answer: str | dict }

Backed by Qwen3.8-Max via ModelScope for synthesis.
Corpus: starts empty, seeded via scraper-api + manual ingestion.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Literal
from uuid import uuid4

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from services.research import corpus

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    corpus.init_db()
    logger.info("Corpus DB initialized")
    # Start background video analysis worker
    from services.research.video_analyzer import start_analysis_worker, stop_analysis_worker
    await start_analysis_worker()
    logger.info("Video analysis worker started")
    yield
    await stop_analysis_worker()
    logger.info("Video analysis worker stopped")


app = FastAPI(title="Local Lightreel Compatibility API", lifespan=lifespan)

# ─── Config ────────────────────────────────────────────────────────────────────
MODELSCOPE_API_KEY = os.environ.get("MODELSCOPE_API_KEY", "")
MODELSCOPE_BASE_URL = os.environ.get(
    "MODELSCOPE_BASE_URL", "https://api-inference.modelscope.ai/v1"
)
MODELSCOPE_MODEL = os.environ.get("MODELSCOPE_MODEL", "Qwen-Ambassador/Qwen3.8-Max")
MAX_RESPONSE_FIELDS = 5


# ─── Request/Response models ──────────────────────────────────────────────────
class ResponseField(BaseModel):
    type: Literal["string", "array"]
    description: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_id: str | None = None
    response_fields: dict[str, ResponseField] | None = None


class ChatResponse(BaseModel):
    conversationId: str
    answer: str | dict[str, Any]


# ─── Critique models ─────────────────────────────────────────────────────────
class CritiqueRequest(BaseModel):
    type: Literal["account", "script", "video_concept"]
    content: str = Field(min_length=1)
    context: str | None = None


class ComparablePost(BaseModel):
    url: str
    hook: str
    views: int


class CritiqueResponse(BaseModel):
    score: int
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    comparable_posts: list[ComparablePost]
    evidence_count: int


# ─── LLM client ───────────────────────────────────────────────────────────────
async def call_llm(messages: list[dict], max_tokens: int = 4096) -> str:
    """Call ModelScope chat completions API."""
    if not MODELSCOPE_API_KEY:
        return "ERROR: MODELSCOPE_API_KEY not configured"

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{MODELSCOPE_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {MODELSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODELSCOPE_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
        )
        if resp.status_code != 200:
            logger.error("LLM call failed: %s %s", resp.status_code, resp.text[:200])
            return f"ERROR: LLM returned {resp.status_code}"

        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ─── Evidence retrieval (SQLite FTS5 corpus) ──────────────────────────────────
async def retrieve_evidence(question: str, limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve relevant posts from the local UGC corpus via FTS5 search."""
    return corpus.search_posts(question, limit=limit)


# ─── Synthesis ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a social media research analyst specializing in TikTok and Instagram content trends.

You analyze retrieved social-media evidence to answer marketing research questions.

Rules:
- Use only the supplied evidence records when available.
- When no evidence is available, provide general knowledge but clearly state it's not grounded in current data.
- Distinguish observed facts from interpretation.
- Do not infer a performance trend from one post.
- Every specific example must include its source URL or evidence ID.
- Preserve index alignment for related arrays (hooks[i] ↔ why_it_works[i] ↔ example_urls[i]).
- Return exactly the requested fields and no others.
- For array fields, return JSON arrays even when empty.
- If evidence is insufficient, return an empty value rather than inventing one.
- Be specific: cite actual hooks, formats, creator names, and metrics when available."""


async def synthesize(
    question: str,
    evidence: list[dict[str, Any]],
    fields: dict[str, ResponseField] | None,
) -> str | dict[str, Any]:
    """Produce a grounded answer conforming to the requested contract."""

    # Limit evidence to avoid gateway timeouts; summarize compactly
    ev_compact = []
    for e in evidence[:5]:
        ev_compact.append({
            "hook": e.get("hook", ""),
            "format": e.get("format", ""),
            "views": e.get("views", 0),
            "engagement_rate": round(e.get("engagement_rate", 0), 4),
            "creator": e.get("creator_handle", ""),
            "url": e.get("post_url", ""),
        })
    evidence_text = json.dumps(ev_compact, indent=1) if ev_compact else "No evidence available in local corpus yet."

    if fields:
        field_contract = "\n".join(
            f"- {name} ({spec.type}): {spec.description}"
            for name, spec in fields.items()
        )
        user_msg = f"""Question: {question}

Evidence:
{evidence_text}

Requested structured output fields:
{field_contract}

Return ONLY valid JSON with exactly these keys: {list(fields.keys())}
Array fields must be JSON arrays. String fields must be strings."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw = await call_llm(messages)

        # Try to parse as JSON
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(cleaned)
            # Validate structure
            validated: dict[str, Any] = {}
            for name, spec in fields.items():
                val = result.get(name)
                if spec.type == "array":
                    validated[name] = val if isinstance(val, list) else []
                else:
                    validated[name] = str(val) if val is not None else ""
            return validated
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Structured output parse failed: %s, returning raw", e)
            # Return a best-effort structure
            return {name: ([] if spec.type == "array" else raw[:500]) for name, spec in fields.items()}
    else:
        user_msg = f"""Question: {question}

Evidence:
{evidence_text}

Provide a thorough, specific answer."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return await call_llm(messages)


# ─── Route ─────────────────────────────────────────────────────────────────────
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
) -> ChatResponse:
    if request.response_fields and len(request.response_fields) > MAX_RESPONSE_FIELDS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"response_fields supports at most {MAX_RESPONSE_FIELDS} fields",
                    "type": "invalid_request",
                }
            },
        )

    evidence = await retrieve_evidence(request.question)
    answer = await synthesize(request.question, evidence, request.response_fields)

    return ChatResponse(
        conversationId=request.conversation_id or f"local-{uuid4().hex[:12]}",
        answer=answer,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "research-api", "model": MODELSCOPE_MODEL}


# ─── Seed endpoint ─────────────────────────────────────────────────────────────
class SeedRequest(BaseModel):
    handles: list[str] = Field(min_length=1)
    max_per_handle: int = Field(default=10, ge=1, le=100)


class SeedResponse(BaseModel):
    inserted: int
    per_handle: dict[str, int]
    errors: list[str]


@app.post("/v1/seed", response_model=SeedResponse)
async def seed_corpus(request: SeedRequest) -> SeedResponse:
    """Seed the corpus from scraper API for given creator handles."""
    result = await corpus.seed_from_scraper_api(
        handles=request.handles,
        max_per_handle=request.max_per_handle,
    )
    return SeedResponse(**result)


# ─── Critique endpoint ────────────────────────────────────────────────────────

CRITIQUE_SYSTEM_PROMPT = """\
You are a TikTok content strategist. Analyze the following {type} against current trending patterns.

Evidence from trending content:
{evidence_json}

Content to critique:
{content}

{context}

Respond in JSON with: score (0-100), strengths (array), weaknesses (array), recommendations (array).
Be specific and actionable. Reference evidence posts by URL when relevant.\
"""


async def _gather_evidence_for_critique(
    critique_type: str,
    content: str,
) -> list[dict[str, Any]]:
    """Gather relevant evidence depending on the critique type."""
    if critique_type == "account":
        # For account critiques, try to scrape the profile first, then search corpus
        handle = content.strip().lstrip("@")
        try:
            await corpus.seed_from_scraper_api(handles=[handle], max_per_handle=10)
        except Exception as e:
            logger.warning("Scraper fetch failed for @%s during critique: %s", handle, e)
        # Search corpus for this creator's posts + general trending patterns
        evidence = corpus.search_posts(handle, limit=10)
        if len(evidence) < 5:
            # Supplement with broader trending content
            evidence.extend(corpus.search_posts("trending viral hook", limit=10))
        return evidence

    elif critique_type == "script":
        # Extract key phrases from the script to search for similar content
        words = content.split()[:10]
        query = " ".join(words)
        evidence = corpus.search_posts(query, limit=15)
        if len(evidence) < 3:
            evidence.extend(corpus.search_posts("hook format viral", limit=10))
        return evidence

    else:  # video_concept
        evidence = corpus.search_posts(content, limit=15)
        if len(evidence) < 3:
            evidence.extend(corpus.search_posts("trending concept format", limit=10))
        return evidence


def _compact_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim evidence records to the fields needed for critique context."""
    compact = []
    for e in evidence[:10]:
        compact.append({
            "hook": e.get("hook", ""),
            "format": e.get("format", ""),
            "views": e.get("views", 0),
            "engagement_rate": round(e.get("engagement_rate", 0), 4),
            "creator": e.get("creator_handle", ""),
            "url": e.get("post_url", ""),
        })
    return compact


def _extract_comparable_posts(evidence: list[dict[str, Any]]) -> list[ComparablePost]:
    """Build comparable_posts from evidence, sorted by views descending."""
    posts = []
    for e in evidence[:5]:
        url = e.get("post_url", "")
        if url:
            posts.append(ComparablePost(
                url=url,
                hook=e.get("hook", ""),
                views=e.get("views", 0),
            ))
    posts.sort(key=lambda p: p.views, reverse=True)
    return posts


@app.post("/v1/critique", response_model=CritiqueResponse)
async def critique(request: CritiqueRequest) -> CritiqueResponse:
    """Analyze a TikTok account, script, or video concept against trending patterns."""
    evidence = await _gather_evidence_for_critique(request.type, request.content)
    ev_compact = _compact_evidence(evidence)
    evidence_json = json.dumps(ev_compact, indent=1) if ev_compact else "No evidence available in local corpus yet."

    context_line = f"Additional context: {request.context}" if request.context else ""
    system_msg = CRITIQUE_SYSTEM_PROMPT.format(
        type=request.type.replace("_", " "),
        evidence_json=evidence_json,
        content=request.content,
        context=context_line,
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Critique this {request.type.replace('_', ' ')} and respond with valid JSON only."},
    ]

    raw = await call_llm(messages, max_tokens=2048)

    # Parse LLM response
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning("Critique LLM parse failed: %s — returning defaults", e)
        parsed = {}

    score = int(parsed.get("score", 50)) if isinstance(parsed.get("score"), (int, float)) else 50
    strengths = parsed.get("strengths", []) if isinstance(parsed.get("strengths"), list) else []
    weaknesses = parsed.get("weaknesses", []) if isinstance(parsed.get("weaknesses"), list) else []
    recommendations = parsed.get("recommendations", []) if isinstance(parsed.get("recommendations"), list) else []

    comparable = _extract_comparable_posts(evidence)

    return CritiqueResponse(
        score=max(0, min(100, score)),
        strengths=[str(s) for s in strengths],
        weaknesses=[str(w) for w in weaknesses],
        recommendations=[str(r) for r in recommendations],
        comparable_posts=comparable,
        evidence_count=len(evidence),
    )


# ─── Video Analysis endpoints ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    post_url: str = Field(min_length=1)
    post_id: str | None = None
    async_mode: bool = Field(default=True, description="If true, queue for background analysis. If false, block until complete.")


class AnalyzeResponse(BaseModel):
    status: str  # "queued" | "completed" | "failed"
    post_id: str
    analysis: dict[str, Any] | None = None
    message: str = ""


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest) -> AnalyzeResponse:
    """Download and VLM-analyze a video post. Queue async or block for result."""
    from services.research.video_analyzer import analyze_post, enqueue_analysis

    post_id = request.post_id or request.post_url.split("/")[-1].split("?")[0]

    if request.async_mode:
        ok = enqueue_analysis(request.post_url, post_id)
        return AnalyzeResponse(
            status="queued" if ok else "queue_full",
            post_id=post_id,
            message="Queued for background analysis" if ok else "Analysis queue full, try again later",
        )
    else:
        analysis = await analyze_post(request.post_url, post_id)
        if analysis:
            analysis["post_url"] = request.post_url  # ensure URL is in analysis dict for stub insert
            corpus.update_vlm_analysis(post_id, analysis)
            return AnalyzeResponse(status="completed", post_id=post_id, analysis=analysis)
        return AnalyzeResponse(status="failed", post_id=post_id, message="Analysis failed")


@app.get("/v1/analyzed")
async def list_analyzed(limit: int = 20):
    """List posts that have been VLM-analyzed."""
    conn = corpus._get_conn()
    try:
        rows = conn.execute(
            "SELECT id, platform, post_url, creator_handle, vlm_hook, vlm_format, "
            "vlm_pacing, vlm_analyzed_at FROM posts WHERE vlm_analyzed_at != '' "
            "ORDER BY vlm_analyzed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {"count": len(rows), "posts": [dict(r) for r in rows]}
    finally:
        conn.close()


# ─── Insights / Correlation Analysis ─────────────────────────────────────────

@app.get("/v1/insights")
async def get_insights(
    dimension: str = "all",
    min_n: int = 2,
    top_k: int = 15,
    tier_analysis: bool = True,
):
    """Engagement ↔ visual pattern correlation analysis.

    Dimensions: hook, format, pacing, energy, audio, all
    Returns ranked categories by avg engagement rate with sample sizes.
    When tier_analysis=true, also compares top-10% vs bottom-50% ER posts.
    """
    import json as _json
    import statistics as _stats
    from collections import defaultdict

    conn = corpus._get_conn()
    try:
        rows = conn.execute(
            "SELECT vlm_hook, vlm_format, vlm_pacing, likes, views, engagement_rate, "
            "comments, shares, saves, vlm_analysis FROM posts WHERE vlm_analyzed_at != '' AND views > 0"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {"error": "No analyzed posts with engagement data"}

    def _norm(text: str | None, max_len: int = 40) -> str:
        if not text:
            return "unknown"
        return text.lower().strip()[:max_len]

    def _compute_stats(items: list[dict]) -> dict:
        ers = [i["er"] for i in items]
        views = sorted([i["views"] for i in items])
        likes = [i["likes"] for i in items]
        return {
            "n": len(items),
            "avg_er": round(_stats.mean(ers), 6),
            "median_er": round(_stats.median(ers), 6),
            "median_views": int(_stats.median(views)),
            "avg_likes": round(_stats.mean(likes), 1),
            "total_likes": sum(likes),
        }

    # ── Per-dimension aggregation ──
    dims = {}
    dim_map = {
        "hook": lambda r: _norm(r[0]),
        "format": lambda r: _norm(r[1]),
        "pacing": lambda r: _norm(r[2]),
        "energy": lambda r: (_json.loads(r[9]).get("energy_level", "unknown").lower() if r[9] else "unknown"),
        "audio": lambda r: (_json.loads(r[9]).get("audio_style", "unknown").lower()[:30] if r[9] else "unknown"),
    }

    target_dims = list(dim_map.keys()) if dimension == "all" else [dimension]

    for dim_name in target_dims:
        if dim_name not in dim_map:
            continue
        buckets: dict[str, list[dict]] = defaultdict(list)
        extractor = dim_map[dim_name]
        for r in rows:
            try:
                key = extractor(r)
            except Exception:
                key = "unknown"
            buckets[key].append({"er": r[5], "views": r[4], "likes": r[3]})

        ranked = []
        for cat, items in buckets.items():
            if len(items) < min_n:
                continue
            s = _compute_stats(items)
            s["category"] = cat
            ranked.append(s)
        ranked.sort(key=lambda x: -x["avg_er"])
        dims[dim_name] = ranked[:top_k]

    result = {"total_posts": len(rows), "dimensions": dims}

    # ── Tier analysis: top 10% vs bottom 50% ──
    if tier_analysis:
        all_ers = sorted([r[5] for r in rows], reverse=True)
        top_cutoff = all_ers[max(0, len(all_ers) // 10)]
        bot_cutoff = all_ers[len(all_ers) // 2]

        top_posts = [r for r in rows if r[5] >= top_cutoff]
        bot_posts = [r for r in rows if r[5] <= bot_cutoff]

        def _tier_patterns(posts, label):
            patterns = {}
            for dim_name, extractor in dim_map.items():
                buckets: dict[str, int] = defaultdict(int)
                for r in posts:
                    try:
                        key = extractor(r)
                    except Exception:
                        key = "unknown"
                    if key != "unknown":
                        buckets[key] += 1
                total = sum(buckets.values()) or 1
                top3 = sorted(buckets.items(), key=lambda x: -x[1])[:5]
                patterns[dim_name] = [
                    {"pattern": k, "count": v, "pct": round(v / total * 100, 1)}
                    for k, v in top3
                ]
            return patterns

        result["tier_analysis"] = {
            "top_10pct": {
                "n": len(top_posts),
                "min_er": round(top_cutoff, 6),
                "patterns": _tier_patterns(top_posts, "top"),
            },
            "bottom_50pct": {
                "n": len(bot_posts),
                "max_er": round(bot_cutoff, 6),
                "patterns": _tier_patterns(bot_posts, "bottom"),
            },
        }

    # ── Cross-correlation: hook × energy ──
    if dimension in ("all", "cross"):
        cross_buckets: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            try:
                analysis = _json.loads(r[9]) if r[9] else {}
                hook = _norm(r[0], 30)
                energy = analysis.get("energy_level", "unknown").lower()
                key = f"{hook}|{energy}"
            except Exception:
                continue
            cross_buckets[key].append({"er": r[5], "views": r[4], "likes": r[3]})

        cross_ranked = []
        for combo, items in cross_buckets.items():
            if len(items) < min_n:
                continue
            s = _compute_stats(items)
            parts = combo.split("|")
            s["hook"] = parts[0]
            s["energy"] = parts[1]
            s["combo"] = combo
            cross_ranked.append(s)
        cross_ranked.sort(key=lambda x: -x["avg_er"])
        result["cross_correlations"] = {
            "hook_x_energy": cross_ranked[:top_k],
        }

    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("RESEARCH_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
