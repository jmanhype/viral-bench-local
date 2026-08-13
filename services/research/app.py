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
from typing import Any, Literal, Optional
from uuid import uuid4

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
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
# VLM/text provider: "gemini" (default) or "modelscope"
VBL_PROVIDER = os.environ.get("VBL_PROVIDER", "gemini")

# Gemini config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

# ModelScope config (fallback)
MODELSCOPE_API_KEY = os.environ.get("MODELSCOPE_API_KEY", "")
MODELSCOPE_BASE_URL = os.environ.get(
    "MODELSCOPE_BASE_URL", "https://api-inference.modelscope.ai/v1"
)
MODELSCOPE_MODEL = os.environ.get("MODELSCOPE_MODEL", "Qwen-Ambassador/Qwen3.7-Max")

ACTIVE_MODEL = GEMINI_MODEL if VBL_PROVIDER == "gemini" else MODELSCOPE_MODEL
MAX_RESPONSE_FIELDS = 5


# ─── Request/Response models ──────────────────────────────────────────────────
class ResponseField(BaseModel):
    type: Literal["string", "array"]
    description: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_id: str | None = None
    response_fields: dict[str, ResponseField] | None = None
    niche: str | None = Field(default=None, description="Filter retrieval to a specific niche (e.g. dance, comedy, brand)")


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
    """Call LLM API (Gemini or ModelScope)."""
    if VBL_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            return "ERROR: GEMINI_API_KEY not configured"

        # Convert OpenAI-style messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": 0.7,
                    }
                },
            )
            if resp.status_code == 429:
                logger.warning("Gemini rate limited")
                return "ERROR: Rate limited"
            if resp.status_code != 200:
                logger.error("Gemini call failed: %s %s", resp.status_code, resp.text[:200])
                return f"ERROR: LLM returned {resp.status_code}"

            data = resp.json()
            try:
                parts = data["candidates"][0]["content"].get("parts", [])
                # Gemini 3.5 includes thoughtSignature parts — find the text one
                for part in parts:
                    if "text" in part and part["text"]:
                        return part["text"]
                return "ERROR: No text in response"
            except (KeyError, IndexError) as e:
                logger.error("Gemini response parse error: %s — raw: %s", e, str(data)[:500])
                return f"ERROR: Parse failed: {e}"
    else:
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


# ─── Niche-to-creator mapping ────────────────────────────────────────────────
NICHE_CREATORS = {
    'comedy': ['khaby.lame', 'wisdm8', 'brittany_broski'],
    'magic/vfx': ['zachking'],
    'dance': ['charlidamelio', 'jasonderulo', 'addisonre'],
    'music': ['bellapoarch', 'toniannmusic'],
    'pets': ['nala_cat', 'tuckerbudzyn', 'realgrumpycat'],
    'food': ['gordonramsayofficial', 'babishculinaryuniverse'],
    'fitness': ['chris.hemsworth', 'pamela_rf', 'blogilates'],
    'education': ['hankgreen', 'neildegrassetyson'],
    'lifestyle': ['emma', 'merrelltwins'],
    'brand': ['duolingo', 'ryanair', 'chipotle'],
    'vfx': ['julianbass'],
}


def get_creators_for_niche(niche: str | None) -> list[str] | None:
    """Return creator handles for a niche, or None if not specified."""
    if not niche:
        return None
    return NICHE_CREATORS.get(niche.lower())


# ─── Evidence retrieval (SQLite FTS5 corpus) ──────────────────────────────────
async def retrieve_evidence(question: str, limit: int = 20, niche: str | None = None) -> list[dict[str, Any]]:
    """Retrieve relevant posts from the local UGC corpus via FTS5 search."""
    creator_handles = get_creators_for_niche(niche)
    return corpus.search_posts(question, limit=limit, creator_handles=creator_handles)


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
            "hook": e.get("vlm_hook") or e.get("hook", ""),
            "format": e.get("vlm_format") or e.get("format", ""),
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

    evidence = await retrieve_evidence(request.question, niche=request.niche)
    answer = await synthesize(request.question, evidence, request.response_fields)

    return ChatResponse(
        conversationId=request.conversation_id or f"local-{uuid4().hex[:12]}",
        answer=answer,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "research-api", "model": ACTIVE_MODEL, "provider": VBL_PROVIDER}


# ─── Viral Score endpoint ────────────────────────────────────────────────────
@app.get("/v1/score")
async def viral_score(
    hook: str,
    niche: Optional[str] = None,
    format: Optional[str] = None,
):
    """Score a hook against the corpus — predict virality."""
    from services.research.viral_score import score_hook
    result = score_hook(hook, niche=niche, format_type=format)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


# ─── Agent endpoint ──────────────────────────────────────────────────────────
class AgentRequest(BaseModel):
    niche: str = Field(min_length=1, description="Content niche: dance, comedy, brand, etc.")
    goal: str = Field(default="", description="Optional marketing goal")
    max_rounds: int = Field(default=5, ge=1, le=10)
    min_score: float = Field(default=5.0, ge=0, le=10)
    custom_direction: str = Field(default="", description="Creative direction / constraints")


class AgentResponse(BaseModel):
    status: Literal["success", "failed"]
    brief: Optional[dict] = None
    error: Optional[str] = None


async def _generate_brief_impl(
    niche: str,
    goal: str = "",
    max_rounds: int = 5,
    min_score: float = 5.0,
    custom_direction: str = "",
) -> AgentResponse:
    """Shared implementation for brief generation."""
    from services.agent.autonomous_agent import AgentConfig, generate_brief

    config = AgentConfig(
        niche=niche,
        goal=goal,
        max_rounds=max_rounds,
        min_score=min_score,
        custom_direction=custom_direction,
    )

    brief = await generate_brief(config)
    if brief:
        return AgentResponse(status="success", brief=brief.model_dump())
    return AgentResponse(status="failed", error="No high-scoring content found")


@app.post("/v1/agent/brief", response_model=AgentResponse)
async def agent_brief_post(request: AgentRequest) -> AgentResponse:
    """Generate a viral content brief for a niche (POST with JSON body)."""
    return await _generate_brief_impl(
        niche=request.niche,
        goal=request.goal,
        max_rounds=request.max_rounds,
        min_score=request.min_score,
        custom_direction=request.custom_direction,
    )


@app.get("/v1/agent/brief", response_model=AgentResponse)
async def agent_brief_get(
    niche: str = Query(..., description="Content niche: dance, comedy, brand, etc."),
    goal: str = Query(default="", description="Optional marketing goal"),
    max_rounds: int = Query(default=5, ge=1, le=10),
    min_score: float = Query(default=5.0, ge=0, le=10),
    custom_direction: str = Query(default="", description="Creative direction / constraints"),
) -> AgentResponse:
    """Generate a viral content brief for a niche (GET with query params)."""
    return await _generate_brief_impl(
        niche=niche,
        goal=goal,
        max_rounds=max_rounds,
        min_score=min_score,
        custom_direction=custom_direction,
    )


# ─── Batch brief endpoint ─────────────────────────────────────────────────────
class BatchBriefRequest(BaseModel):
    briefs: list[AgentRequest] = Field(min_length=1, max_length=20)


class BatchBriefResponse(BaseModel):
    status: Literal["success", "partial", "failed"]
    briefs: list[dict]
    stats: dict


@app.post("/v1/agent/batch_brief", response_model=BatchBriefResponse)
async def agent_batch_brief(request: BatchBriefRequest) -> BatchBriefResponse:
    """Generate multiple briefs with deduplication to avoid hook/style repetition."""
    from services.agent.autonomous_agent import AgentConfig, generate_brief
    import asyncio
    
    # Parallel generation with semaphores to avoid overwhelming the API
    semaphore = asyncio.Semaphore(8)  # Max 8 concurrent brief generations
    
    async def generate_one(req: AgentRequest) -> dict:
        async with semaphore:
            config = AgentConfig(
                niche=req.niche,
                goal=req.goal,
                max_rounds=req.max_rounds,
                min_score=req.min_score,
                custom_direction=req.custom_direction,
            )
            
            brief = await generate_brief(config)
            if brief:
                return {
                    "niche": req.niche,
                    "goal": req.goal,
                    "brief": brief.model_dump(),
                    "success": True
                }
            return {"niche": req.niche, "goal": req.goal, "success": False}
    
    # Generate all briefs in parallel
    tasks = [generate_one(req) for req in request.briefs]
    raw_results = await asyncio.gather(*tasks)
    
    # Deduplicate based on hook similarity (more relaxed)
    results = []
    used_hooks = []
    used_styles = set()
    failed = []
    
    for r in raw_results:
        if not r["success"]:
            failed.append({"niche": r["niche"], "goal": r["goal"], "reason": "generation failed"})
            continue
        
        hook = r["brief"]["hook"]
        style_name = r["brief"]["visual_direction"].get("style_name") if r["brief"].get("visual_direction") else None
        
        # Check for semantic similarity (relaxed: word overlap > 85%)
        hook_words = set(hook.lower().split())
        is_duplicate = False
        for used in used_hooks:
            used_words = set(used.lower().split())
            if hook_words and used_words:
                overlap = len(hook_words & used_words) / min(len(hook_words), len(used_words))
                if overlap > 0.85:
                    is_duplicate = True
                    break
        
        if is_duplicate or (style_name and style_name in used_styles):
            failed.append({"niche": r["niche"], "goal": r["goal"], "reason": "duplicate"})
            continue
        
        used_hooks.append(hook)
        if style_name:
            used_styles.add(style_name)
        del r["success"]
        results.append(r)
    
    status = "success" if len(results) == len(request.briefs) else ("partial" if results else "failed")
    
    return BatchBriefResponse(
        status=status,
        briefs=results,
        stats={
            "requested": len(request.briefs),
            "generated": len(results),
            "failed": len(failed),
            "unique_hooks": len(used_hooks),
            "unique_styles": len(used_styles),
            "failures": failed,
        }
    )


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

@app.get("/v1/top_hooks")
async def get_top_hooks(
    niche: str | None = None,
    limit: int = 10,
    min_engagement: float = 0.1,
):
    """Return actual high-performing hook text from corpus.
    
    Returns raw caption text (not VLM pattern labels) sorted by engagement rate.
    Used by autonomous_agent to seed hook generation.
    """
    conn = corpus._get_conn()
    try:
        # Filter by niche if provided
        niche_filter = ""
        params = {"limit": limit, "min_er": min_engagement}
        
        if niche:
            creators = NICHE_CREATORS.get(niche.lower())
            if creators:
                placeholders = ",".join(f":c{i}" for i in range(len(creators)))
                niche_filter = f"AND creator_handle IN ({placeholders})"
                for i, c in enumerate(creators):
                    params[f"c{i}"] = c
        
        # Get top hooks by engagement rate with actual caption text
        sql = f"""
            SELECT caption, format, views, engagement_rate, creator_handle
            FROM posts
            WHERE caption IS NOT NULL AND caption != ''
              AND engagement_rate >= :min_er
              {niche_filter}
            ORDER BY engagement_rate DESC
            LIMIT :limit
        """
        
        rows = conn.execute(sql, params).fetchall()
        
        hooks = []
        for r in rows:
            # Extract first line or first 100 chars as the hook
            caption = r["caption"]
            hook_text = caption.split('\n')[0][:150] if caption else ""
            
            hooks.append({
                "hook": hook_text,
                "format": r["format"],
                "views": r["views"],
                "engagement_rate": round(r["engagement_rate"], 3),
                "creator": r["creator_handle"],
            })
        
        return {"niche": niche, "count": len(hooks), "hooks": hooks}
    finally:
        conn.close()


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

    def _norm_hook(text: str | None) -> str:
        """Normalize free-text hook descriptions into categorical buckets."""
        h = (text or "").lower()
        if not h or h == "unknown":
            return "unknown"
        if "curiosity-gap" in h or "curiosity gap" in h:
            if "shock" in h:
                return "curiosity-gap → shock"
            if "text" in h:
                return "curiosity-gap + text overlay"
            if "direct address" in h or "direct-address" in h:
                return "curiosity-gap + direct address"
            return "curiosity-gap"
        if "shock" in h:
            return "shock visual"
        if "direct address" in h or "direct-address" in h:
            return "direct address"
        if "text overlay" in h or "text-overlay" in h:
            return "text overlay hook"
        if "relatable" in h:
            return "relatable scenario"
        if "skill" in h or "misdirection" in h:
            return "skill/misdirection"
        if "warm" in h or "emotional" in h:
            return "emotional/warm"
        return "other"

    def _norm_energy(analysis_json: str | None) -> str:
        """Extract and normalize energy level from VLM analysis JSON."""
        if not analysis_json:
            return "unknown"
        try:
            e = _json.loads(analysis_json).get("energy_level", "unknown").lower().strip()
        except Exception:
            return "unknown"
        if "extreme" in e:
            return "extreme"
        if "high" in e:
            return "high"
        if "medium-high" in e or "medium high" in e:
            return "medium-high"
        if "medium" in e:
            return "medium"
        if "low" in e:
            return "low"
        return "unknown"

    def _norm_pacing(text: str | None) -> str:
        """Normalize free-text pacing descriptions into categorical buckets."""
        p = (text or "").lower()
        if not p or p == "unknown":
            return "unknown"
        if "rapid cut" in p:
            return "rapid cuts"
        if "single take" in p or "continuous" in p or "uncut" in p:
            return "single take"
        if "slow" in p and "escalat" in p:
            return "slow build → escalate"
        if "moderate" in p:
            return "moderate cuts"
        if "fast" in p:
            return "fast paced"
        return "other"

    def _norm_format(text: str | None) -> str:
        """Normalize free-text format descriptions into categorical buckets."""
        f = (text or "").lower()
        if not f or f == "unknown":
            return "unknown"
        if "skit" in f or "comedy" in f:
            return "skit/comedy"
        if "vlog" in f or "selfie" in f or "talking head" in f:
            return "vlog/talking head"
        if "dance" in f or "lip-sync" in f or "lip sync" in f:
            return "dance/lip-sync"
        if "performance" in f or "concert" in f:
            return "performance"
        if "tutorial" in f or "how-to" in f or "diy" in f:
            return "tutorial/DIY"
        if "montage" in f:
            return "montage"
        if "split-screen" in f or "split screen" in f:
            return "split screen"
        return "other"

    def _norm_audio(analysis_json: str | None) -> str:
        """Extract and normalize audio style from VLM analysis JSON."""
        if not analysis_json:
            return "unknown"
        try:
            a = _json.loads(analysis_json).get("audio_style", "unknown").lower().strip()
        except Exception:
            return "unknown"
        if "dialogue" in a or "talk" in a:
            return "dialogue/talking"
        if "music" in a or "song" in a:
            return "music"
        if "sound effect" in a or "sfx" in a:
            return "sound effects"
        if "voiceover" in a or "narration" in a:
            return "voiceover"
        if "lip-sync" in a or "lip sync" in a:
            return "lip-sync"
        if "silent" in a or "no audio" in a:
            return "silent"
        return "other"

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
        "hook": lambda r: _norm_hook(r[0]),
        "format": lambda r: _norm_format(r[1]),
        "pacing": lambda r: _norm_pacing(r[2]),
        "energy": lambda r: _norm_energy(r[9]),
        "audio": lambda r: _norm_audio(r[9]),
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
                hook = _norm_hook(r[0])
                energy = _norm_energy(r[9])
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
