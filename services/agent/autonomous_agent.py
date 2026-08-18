"""Autonomous viral content agent — insights → generate → score → brief.

Generates hook candidates directly from top-performing patterns in the corpus,
then scores them to find the best one. No external LLM dependency.

Flow:
1. Fetch top-performing patterns from VBL corpus (niche-scoped)
2. Generate hook candidates from proven patterns (template-based)
3. Score each hook against corpus nearest-neighbors
4. Select the best scorer
5. Output a complete content brief

Output: Ready-to-produce brief. Human handles video creation + posting.
"""
import asyncio
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Optional

from pathlib import Path
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Service URLs
RESEARCH_URL = "http://127.0.0.1:8001"
SGOS_URL = "http://127.0.0.1:8420"

# Video length: every video is capped at 20 seconds.
# H3 runs at 24fps → 20s = 480 frames. All script beats and dialogue
# timestamps are planned on this grid.
VIDEO_DURATION_S = 20
H3_FPS = 24
H3_FRAME_COUNT = VIDEO_DURATION_S * H3_FPS  # 480

# Defaults
MAX_ROUNDS = 3
MIN_SCORE = 5.0

# Diversity tracking: cache recently returned hooks to avoid repetition
from collections import defaultdict
import time

_recent_hooks = defaultdict(list)  # niche -> [(timestamp, hook_normalized), ...]
_COOLDOWN_SECONDS = 300  # 5 minutes
_RECENT_PENALTY = 4.5  # Score penalty for recently returned hooks

def _record_hook_return(niche: str, hook_text: str):
    """Record that a hook was returned for this niche."""
    import re
    normalized = re.sub(r'\W+', '', hook_text.lower())
    now = time.time()
    _recent_hooks[niche].append((now, normalized))
    # Clean old entries
    _recent_hooks[niche] = [(t, h) for t, h in _recent_hooks[niche] if now - t < _COOLDOWN_SECONDS]

def _get_recent_penalty(niche: str, hook_text: str) -> float:
    """Get score penalty for recently returned hooks."""
    import re
    normalized = re.sub(r'\W+', '', hook_text.lower())
    now = time.time()
    for t, h in _recent_hooks.get(niche, []):
        if h == normalized:
            # Penalty decays over time
            age_seconds = now - t
            decay = 1.0 - (age_seconds / _COOLDOWN_SECONDS)
            return _RECENT_PENALTY * decay
    return 0.0


class AgentConfig(BaseModel):
    niche: str = Field(description="Content niche: dance, comedy, brand, fitness, food, etc.")
    goal: str = Field(default="", description="Optional marketing goal")
    max_rounds: int = MAX_ROUNDS
    min_score: float = MIN_SCORE
    custom_direction: str = Field(default="", description="Creative direction / constraints")
    style_id: str = Field(default="", description="Pin a visual style by id (franchise lock) — e.g. '2000s-atlanta-trap-house'. Empty = auto-match.")
    character_id: str = Field(default="", description="Pin a franchise character by id (character lock) — e.g. 'unc-ray'. Empty = archetypes only.")


class HookCandidate(BaseModel):
    hook: str
    format_type: str
    script: str
    caption: str
    hashtags: list[str] = []
    viral_score: float = 0
    predicted_er: dict = {}
    nearest_neighbors: list[dict] = []
    pattern_dna: list[dict] = []


def _detect_hood_scenario(hook_lower: str) -> str:
    """Detect which hood skit scenario a hook belongs to."""
    if "mama" in hook_lower or "caught" in hook_lower:
        return "mama"
    if "block" in hook_lower or "friday" in hook_lower or "neighbor" in hook_lower:
        return "block"
    if "cookout" in hook_lower or "function" in hook_lower:
        return "function"
    if "graduated" in hook_lower or "only one" in hook_lower:
        return "family"
    if "grew up" in hook_lower or "tell me" in hook_lower:
        return "nostalgia"
    if "food" in hook_lower or "slaps" in hook_lower:
        return "food"
    if "dj" in hook_lower or "aux" in hook_lower or "playlist" in hook_lower:
        return "dj"
    if "barbershop" in hook_lower or "barber" in hook_lower or "haircut" in hook_lower:
        return "barbershop"
    if "corner store" in hook_lower or "bodega" in hook_lower or "credit" in hook_lower:
        return "cornerstore"
    return "general"


def _hook_subject(hook: str) -> str:
    """Pull a short concrete subject out of the hook so dialogue can share a topic.

    dreamingtulpa's dialogue works because every line is about the SAME specific
    thing. Our hooks carry that subject — we just have to strip the meta framing
    ("POV:", "this is what ... really looks like") down to the noun/event people
    are actually reacting to. Returns a cleaned phrase; falls back to the hook.
    """
    import re
    h = hook.strip()
    # Strip format prefixes
    h = re.sub(r"^(POV:|POV\s|When|Y'all tell me|Real talk:|Hot take:|Storytime:?)\s*",
               "", h, flags=re.IGNORECASE)
    # Strip meta framing that wraps the real subject
    h = re.sub(r"^(this is what|this is how|what)\s+", "", h, flags=re.IGNORECASE)
    h = re.sub(r"\s+(really )?(looks like|is like|actually happened|happened|went down)$",
               "", h, flags=re.IGNORECASE)
    h = re.sub(r"[.!?]+$", "", h).strip()
    # Drop a leading article for a tighter noun phrase
    h = re.sub(r"^(the|a|an)\s+", "", h, flags=re.IGNORECASE)
    return h or hook.strip()


def generate_dialogue(scenario: str, hook: str, guide: Optional[dict] = None, premise: str = "") -> list[dict]:
    """Generate shootable dialogue lines timed to script beats.

    Each line: {time, character, line} — character includes delivery hints
    like (VO) or (off-camera). Returns [] for formats that don't use dialogue.

    Common-sense rule: spoken lines must share ONE subject — the concrete thing
    the hook is about. `subject` is the hook's topic with meta framing stripped,
    so the exchange reads like a real conversation, not canned one-liners.

    Style-aware: when a style's `dialogue_guide` is supplied and the style is
    NOT a hood/street style, dialogue is built from the guide's `examples` in the
    guide's `tone` instead of the hardcoded hood templates. Banned_phrases are
    always filtered out, regardless of style.
    """
    subject = _hook_subject(hook)
    banned = [str(b).lower() for b in (guide or {}).get("banned_phrases") or []]

    # Style-aware: non-hood styles use their guide's examples/tone, not hood lines.
    if guide and not _is_hood_style(guide):
        return _guide_dialogue(guide, hook, subject, scenario, premise)

    DIALOGUES = {
        "block": [
            {"time": "0-2s", "character": "NARRATOR (VO)", "line": hook},
            {"time": "2-7s", "character": "KID", "line": "DJ! DJ! Play that one song!"},
            {"time": "7-15s", "character": "UNCLE ON THE GRILL", "line": "Ain't nobody touching this grill but me. I don't care who you are."},
            {"time": "7-15s", "character": "AUNTIE", "line": "I made four types of potato salad — y'all gon' try ALL of 'em."},
            {"time": "7-15s", "character": "COUSIN (showing up late)", "line": "Man, y'all ALWAYS eat without me!"},
            {"time": "15-20s", "character": "NARRATOR (VO)", "line": "And yeah — the cops shut it down at 9:47. Every single time."},
        ],
        "mama": [
            {"time": "0-2s", "character": "NARRATOR (VO)", "line": hook},
            {"time": "2-7s", "character": "MAMA (off-camera)", "line": "Boy, WHERE you at?! It is ELEVEN o'clock!"},
            {"time": "7-15s", "character": "YOU (whispering)", "line": "I'm outside..."},
            {"time": "7-15s", "character": "MAMA (off-camera)", "line": "Outside?! You better be inside in five seconds."},
            {"time": "15-20s", "character": "YOU (to camera)", "line": "Y'all already know how this ended."},
        ],
        "function": [
            {"time": "0-2s", "character": "NARRATOR (VO)", "line": hook},
            {"time": "2-7s", "character": "COUSIN", "line": "Man, the food table got a VIP section now?"},
            {"time": "7-15s", "character": "UNCLE ON THE GRILL", "line": "You ain't touching this grill, lil man."},
            {"time": "7-15s", "character": "AUNTIE", "line": "He brought STORE-BOUGHT potato salad... to my cookout."},
            {"time": "15-20s", "character": "CROWD", "line": "Ooooooh!"},
        ],
        "family": [
            {"time": "0-2s", "character": "NARRATOR (VO)", "line": hook},
            {"time": "2-7s", "character": "AUNTIE", "line": "Baby, you the first one. Make us proud."},
            {"time": "7-15s", "character": "YOU (VO)", "line": "They said I couldn't. So I did it twice."},
            {"time": "15-20s", "character": "MAMA", "line": "I always knew."},
        ],
        "nostalgia": [
            {"time": "0-2s", "character": "NARRATOR (VO)", "line": hook},
            {"time": "7-15s", "character": "YOU", "line": "Corner store had a policy: no shoes, no shirt — still got service."},
            {"time": "7-15s", "character": "FRIEND", "line": "Man, the block party DJ had ONE job."},
            {"time": "15-20s", "character": "YOU (to camera)", "line": "If you know this one... we family."},
        ],
        "food": [
            {"time": "0-2s", "character": "NARRATOR (VO)", "line": hook},
            {"time": "2-7s", "character": "SKEPTIC", "line": "It's just a plate of food, bro."},
            {"time": "7-15s", "character": "YOU", "line": "You ain't even tasted it yet."},
            {"time": "15-20s", "character": "SKEPTIC (after one bite)", "line": "...okay. I'm wrong. I'm wrong."},
        ],
        # POV: the hook rides in the narrator VO — on screen the characters must
        # say things people actually SAY. Every line shares one subject: the
        # hook's concrete topic (meta framing stripped). Never echo "POV:"
        # verbatim — nobody walks around saying that out loud.
        "pov": [
            {"time": "0-2s", "character": "YOU (to camera)", "line": "Y'all see what just happened?"},
            {"time": "7-15s", "character": "FRIEND", "line": f"Wait — {subject}?"},
            {"time": "15-20s", "character": "YOU", "line": "Yep. Every. Single. Time."},
        ],
        "dj": [
            {"time": "0-2s", "character": "NARRATOR (VO)", "line": hook},
            {"time": "2-7s", "character": "KID", "line": "DJ! Play the one everybody waiting on!"},
            {"time": "7-15s", "character": "DJ", "line": "Nah, I control the AUX tonight. Y'all gone wait."},
            {"time": "7-15s", "character": "UNCLE ON THE GRILL", "line": "Boy, play some Isley Brothers before I unplug you."},
            {"time": "15-20s", "character": "CROWD", "line": "PLAY. THE. SONG."},
        ],
        "barbershop": [
            {"time": "0-2s", "character": "NARRATOR (VO)", "line": hook},
            {"time": "2-7s", "character": "BARBER", "line": "You see this chair? Four generations sat in this chair."},
            {"time": "7-15s", "character": "BARBER", "line": "I don't fix haircuts. I fix decisions."},
            {"time": "7-15s", "character": "FRIEND", "line": "Bro, he just described my whole life in one cut."},
            {"time": "15-20s", "character": "YOU", "line": "Barbershop debates undefeated. Undefeated."},
        ],
        "cornerstore": [
            {"time": "0-2s", "character": "NARRATOR (VO)", "line": hook},
            {"time": "2-7s", "character": "KID", "line": "Mr. Hassan, can I get two Arizona and a Honey Bun?"},
            {"time": "7-15s", "character": "STORE OWNER", "line": "You owe me from Tuesday. Pay your mama first."},
            {"time": "7-15s", "character": "STORE OWNER", "line": "And don't touch the glass. You know the glass rule."},
            {"time": "15-20s", "character": "YOU (VO)", "line": "Man held the whole block together. One register at a time."},
        ],
        # NARRATOR (VO) lines use the hook — that's legitimate voiceover copy,
        # it's read, not acted on screen. Spoken lines share the hook's subject.
        "general": [
            {"time": "0-2s", "character": "NARRATOR (VO)", "line": hook},
            {"time": "7-15s", "character": "FRIEND", "line": f"Hold on — {subject}?"},
            {"time": "15-20s", "character": "YOU", "line": "Yep. Every single time."},
        ],
    }
    result = DIALOGUES.get(scenario, [])
    if banned:
        filtered = []
        for d in result:
            if not any(b in d["line"].lower() for b in banned):
                filtered.append(d)
        result = filtered
    return result


# Keyword → (tone, banned_phrases) for inferring dialogue tone from style_id/franchise names
_TONE_HINTS = {
    "kaiju": ("ominous military narrator or pilot tech-speak", ["Y'all", "no cap", "fr fr", "bruh"]),
    "godzilla": ("ominous military narrator or pilot tech-speak", ["Y'all", "no cap", "fr fr", "bruh"]),
    "gojira": ("ominous military narrator or pilot tech-speak", ["Y'all", "no cap", "fr fr", "bruh"]),
    "tokusatsu": ("dramatic narrator, heroic declarations", ["Y'all", "no cap", "fr fr"]),
    "mecha": ("pilot tech-speak, mission briefing tone", ["Y'all", "no cap", "fr fr"]),
    "super-robot": ("ominous military narrator or pilot tech-speak", ["Y'all", "no cap", "fr fr", "bruh"]),
    "sitcom": ("warm ensemble banter, comedic timing", ["Target emerging", "All units", "brace for impact"]),
    "comedy": ("casual humor, self-aware wit", ["Target emerging", "All units", "brace for impact"]),
    "laugh": ("warm ensemble banter, comedic timing", ["Target emerging", "All units"]),
    "horror": ("whispered dread, fragmented sentences", ["Y'all", "no cap", "fr fr", "bruh", "lol"]),
    "vhs": ("eerie narration, unsettling calm", ["Y'all", "no cap", "fr fr", "bruh"]),
    "space": ("retro-futuristic narration, formal scientific tone", ["Y'all", "no cap", "fr fr", "bruh"]),
    "astronaut": ("mission control formality, awe-struck wonder", ["Y'all", "no cap", "fr fr"]),
    "futurism": ("retro-futuristic narration, optimistic technology tone", ["Y'all", "no cap", "fr fr"]),
    "music": ("lyrical, rhythmic, matches beat energy", ["Target emerging", "All units"]),
    "mtv": ("energetic VJ-style hype, music video energy", ["Target emerging", "All units"]),
    "retro": ("period-appropriate speech, nostalgic warmth", ["Y'all", "no cap", "fr fr", "bruh"]),
    "vintage": ("period-appropriate speech, formal or stylized", ["Y'all", "no cap", "fr fr"]),
    "anime": ("stylized speech, dramatic declarations", ["no cap", "fr fr"]),
    "noir": ("hardboiled narration, cynical inner monologue", ["Y'all", "no cap", "fr fr", "bruh", "lol"]),
    "fantasy": ("archaic/formal speech, mythic tone", ["no cap", "fr fr", "bruh"]),
    "punk": ("rebellious edge, raw energy", ["sir", "ma'am"]),
    "western": ("frontier drawl, laconic stoicism", ["no cap", "fr fr"]),
    "military": ("clipped commands, operational brevity", ["Y'all", "no cap", "fr fr"]),
    "documentary": ("measured narrator, factual authority", ["Y'all", "no cap", "fr fr"]),
    "news": ("broadcast authority, urgent clarity", ["Y'all", "no cap", "fr fr"]),
    "cart": ("high-energy competition commentary", ["Target emerging", "All units"]),
    "kart": ("high-energy competition commentary", ["Target emerging", "All units"]),
}


def _infer_tone_from_ids(style_id: str, franchise: str = "") -> Optional[dict]:
    """Infer a dialogue_guide from style_id/franchise (two-tier).

    TIER 1 (fast cache): keyword match over _TONE_HINTS — returned without an
    API call. TIER 2 (dynamic LLM): if no keyword matches, ask the LLM what
    tone/banned phrases suit this style name and cache the result for later.
    FALLBACK: neutral default if the LLM is unavailable/fails.
    """
    combined = f"{style_id} {franchise}".lower().replace("-", " ").replace("_", " ")
    for keyword, (tone, banned) in _TONE_HINTS.items():
        if keyword in combined:
            return {"tone": tone, "banned_phrases": banned, "examples": []}

    # Tier 2: dynamic LLM inference for arbitrary style names.
    guide = _run_async(_llm_infer_tone(style_id, franchise)) or {}
    tone = guide.get("tone")
    if not tone:
        print(f"[TONE] No tone inferred for '{style_id}' (franchise '{franchise}'); using neutral default", flush=True)
        return {"tone": "neutral in-scene narration", "banned_phrases": [], "examples": []}

    # Cache so the same style doesn't hit the LLM again.
    _TONE_HINTS[combined.strip() or style_id] = (tone, guide.get("banned_phrases") or [])
    return {"tone": tone, "banned_phrases": guide.get("banned_phrases") or [], "examples": []}


async def _llm_infer_tone(style_id: str, franchise: str) -> dict:
    """Ask the LLM for dialogue tone + banned phrases for an arbitrary style name."""
    try:
        from services.research.app import call_llm
    except Exception as exc:  # noqa: BLE001
        print(f"[TONE] LLM client unavailable ({exc}); neutral fallback", flush=True)
        return {}
    user_msg = (
        f'Given this video aesthetic/style name: "{style_id}" (franchise: "{franchise}"), '
        f"what dialogue tone would be appropriate? What phrases should be BANNED because they "
        f"clash with this aesthetic? Respond in JSON: "
        f'{{"tone": "description of appropriate dialogue tone", "banned_phrases": ["phrase1", "phrase2"]}}'
    )
    messages = [
        {"role": "system", "content": "You infer dialogue tone for AI video aesthetics. Reply only with valid JSON."},
        {"role": "user", "content": user_msg},
    ]
    try:
        raw = await call_llm(messages, max_tokens=256)
    except Exception as exc:
        print(f"[TONE] LLM tone inference failed: {exc}; neutral fallback", flush=True)
        return {}
    if not raw or (isinstance(raw, str) and raw.strip().lower().startswith("error")):
        print("[TONE] LLM returned an error; neutral fallback", flush=True)
        return {}
    tone = _parse_tone_json(raw)
    print(f"[TONE] Inferred for '{style_id}': {tone}", flush=True)
    return tone


def _parse_tone_json(raw: str) -> dict:
    """Parse {tone, banned_phrases} from the LLM, tolerating fences/extra text."""
    if not raw:
        return {}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract a JSON object substring.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    if not isinstance(data, dict):
        return {}
    banned = data.get("banned_phrases") or []
    if isinstance(banned, str):
        banned = [banned]
    return {"tone": str(data.get("tone", "")), "banned_phrases": [str(b) for b in banned]}


def _is_hood_style(guide: Optional[dict]) -> bool:
    """True if the style's dialogue_guide is a hood/street/community style that
    legitimately uses the existing casual 'y'all' templates."""
    if not guide:
        return False
    tone = (guide.get("tone") or "").lower()
    return ("hood" in tone or "street" in tone or "community" in tone or "real talk" in tone)


def _is_helpful_guide(guide: Optional[dict]) -> bool:
    """True if a dialogue_guide is usable: has a tone and at least one example."""
    return bool(guide and (guide.get("tone") or (guide.get("examples") or [])))


def _run_async(coro):
    """Run an async coroutine from a sync context, tolerating an already-running
    event loop (fresh thread + own loop) — safe inside generate_brief's loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result = {}
    def _runner():
        result["value"] = asyncio.run(coro)
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    return result.get("value")


def _generate_dynamic_dialogue_sync(
    premise: str,
    tone: str,
    banned_phrases: Optional[list] = None,
    num_lines: int = 3,
    force_scene: bool = False,
    emphasize_scene: bool = False,
) -> list[str]:
    """Dynamic LLM dialogue for a video scene (sync wrapper)."""
    coro = _generate_dynamic_dialogue(premise, tone, banned_phrases, num_lines, force_scene, emphasize_scene)
    return _run_async(coro) or []


async def _generate_dynamic_dialogue(
    premise: str,
    tone: str,
    banned_phrases: Optional[list] = None,
    num_lines: int = 3,
    force_scene: bool = False,
    emphasize_scene: bool = False,
) -> list[str]:
    """Generate dialogue lines with the LLM (the same client used for briefs).

    Prompt asks for `num_lines` short lines (under 15 words) spoken by a
    character in the scene described by `premise`, in `tone`, never using
    `banned_phrases`. Returns a JSON array of strings. Falls back to the style's
    `examples`/guide when the LLM call fails or key is missing.
    """
    # Lazy import to avoid a circular import: research.app imports this module.
    try:
        from services.research.app import call_llm
    except Exception as exc:  # noqa: BLE001
        print(f"[DIALOGUE] LLM client unavailable ({exc}); falling back", flush=True)
        return []

    banned = ", ".join(banned_phrases or []) or "none"
    scene_emphasis = (
        " Each line MUST directly reference the scene you just described (reuse "
        "its key nouns and setting) — do not write generic lines."
        if (force_scene or emphasize_scene) else ""
    )
    user_msg = (
        f'Generate {num_lines} short dialogue lines (under 15 words each) for this '
        f'video scene: "{premise}". '
        f"Tone: {tone}. "
        f"NEVER use these phrases: {banned}. "
        f"Each line should be spoken by a character IN the scene. "
        f"{scene_emphasis}"
        f'Return as a JSON array of strings, e.g. ["Line one", "Line two", "Line three"].'
    )
    messages = [
        {"role": "system", "content": "You write in-scene dialogue for AI video generation. Reply only with valid JSON."},
        {"role": "user", "content": user_msg},
    ]
    try:
        raw = await call_llm(messages, max_tokens=512)
    except Exception as exc:
        print(f"[DIALOGUE] LLM dialogue call failed: {exc}; using examples", flush=True)
        return []

    # call_llm returns 'ERROR: ...' strings on config/parse failures — detect and
    # fall back to guide examples rather than treating the error as dialogue.
    if not raw or (isinstance(raw, str) and raw.strip().lower().startswith("error")):
        print("[DIALOGUE] LLM returned an error/non-dialogue; using examples", flush=True)
        return []

    lines = _parse_dialogue_json(raw)
    # Enforce banned phrases + length guard.
    banned_lower = [b.lower() for b in (banned_phrases or [])]
    clean = []
    for line in lines:
        text = str(line).strip()
        if not text or len(text.split()) > 15:
            continue
        if any(b in text.lower() for b in banned_lower if b):
            continue
        clean.append(text)
    print(f"[DIALOGUE] Dynamic lines: {clean}", flush=True)
    return clean[:num_lines]


def _parse_dialogue_json(raw: str) -> list[str]:
    """Best-effort parse of {json array} from an LLM dialogue response."""
    if not raw:
        return []
    text = raw.strip()
    # Strip markdown code fences.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict) and ("lines" in data or "dialogue" in data):
            arr = data.get("lines") or data.get("dialogue") or []
            return [str(x) for x in arr] if isinstance(arr, list) else []
        return []
    except json.JSONDecodeError:
        # Raw newline-separated lines fallback.
        return [l.strip() for l in text.splitlines() if l.strip()]


def _guide_dialogue(guide: dict, hook: str, subject: str, scenario: str, premise: str = "") -> list[dict]:
    """Build dialogue lines for a non-hood style, LLM-generated from the premise.

    Uses `_generate_dynamic_dialogue` with the FULL premise/service tone to produce
    scene-specific lines; falls back to the guide's `examples` if the LLM call
    fails or returns nothing usable. Banned_phrases are enforced.
    """
    tone = guide.get("tone") or "neutral, in-scene"
    banned = guide.get("banned_phrases") or []
    scene = premise or hook or subject
    dynamic = _generate_dynamic_dialogue_sync(scene, tone, banned, num_lines=3, force_scene=True)
    if dynamic:
        # Post-gen check: if no line references the premise, warn + try once more
        # with an explicit instruction to reference the scene.
        if not _premise_referenced(dynamic, premise or hook or subject):
            print(f"[DIALOGUE] Premise not referenced; regenerating with scene emphasis: {scene[:60]}", flush=True)
            dynamic2 = _generate_dynamic_dialogue_sync(
                scene, tone, banned, num_lines=3, force_scene=True, emphasize_scene=True,
            )
            if _premise_referenced(dynamic2, premise or hook or subject):
                dynamic = dynamic2
        if dynamic:
            return _lines_from(dynamic, hook or subject)

    # Fallback: guide examples, else a premise-referencing in-scene line (so
    # music/dance/any style with empty examples still gets dialogue).
    examples = [e for e in (guide.get("examples") or []) if str(e).strip()]
    if examples:
        return _lines_from([hook or subject] + examples[:3], hook or subject)
    scene_line = (premise or hook or subject).strip() or "the scene happens around us"
    return _lines_from([scene_line, f"Stay with it — {scene_line[:60]}", "...and it changes everything."], hook or subject)


def _lines_from(lines, fallback) -> list[dict]:
    """Turn dialogue strings into timed {character, line} dicts plus a narrator opener."""
    if not fallback:
        fallback = ""
    out = [{"time": "0-2s", "character": "NARRATOR (VO)", "line": fallback}]
    for i, line in enumerate(lines[:3]):
        char = "YOU (to camera)" if i == 0 else ("OBSERVER" if i == 1 else "VOICE (room tone)")
        out.append({"time": f"{3 + i * 5}-{7 + i * 5}s", "character": char, "line": str(line)})
    return out


def _premise_referenced(lines, premise: str) -> bool:
    """True if any dialogue line contains a content word from the premise (>=3 chars,
    not a stopword). Speech that ignores the scene triggers a regeneration."""
    if not premise:
        return True
    stop = {"the","a","an","and","or","of","to","in","on","at","for","with","from","that","it","is","are","was","-",","}
    words = {w.strip(".,!?\"' ").lower() for w in premise.split() if len(w.strip(".,!?\"' ")) >= 3 and w.strip(".,!?\"' ").lower() not in stop}
    if not words:
        return True
    for line in lines:
        lw = set(re.findall(r"[a-z0-9']{2,}", str(line).lower()))
        if lw & words:
            return True
    return False


# Voice texture per character archetype — baked into video prompts in the style of
# "dropping a 'line'. his voice is deep and old like he smoked 2 packs a day"
VOICE_PROFILES = {
    "UNCLE ON THE GRILL": "deep gravelly voice like he smoked 2 packs of cigarettes per day",
    "AUNTIE": "loud warm voice that carries across the whole block",
    "MAMA": "sharp commanding mama voice that ends every argument",
    "KID": "high-pitched excited kid voice, slightly out of breath",
    "COUSIN": "winded complaining voice like he just ran from three blocks away",
    "FRIEND": "casual conversational voice, half laughing",
    "SKEPTIC": "doubting flat tone that flips into full belief",
    "CROWD": "collective crowd reaction, hyped",
    "YOU": "natural conversational voice, direct to camera",
    "NARRATOR": "deep reflective voiceover, nostalgic tone",
    "DJ": "smooth confident voice over the beat, mic-voice energy",
    "BARBER": "calm measured barbershop sage voice, never rushes a sentence",
    "STORE OWNER": "gruff but warm elder voice with an accent, speaks in short rules",
}


def build_dialogue_prompt(scenario: str, hook: str, character: Optional[dict] = None, guide: Optional[dict] = None, premise: str = "") -> str:
    """Render dialogue lines into one dense video-prompt clause.

    Style: character dropping "line" + voice texture — so video/lip-sync tools
    get who is talking, what they say, and how they sound. Narrator lines are
    skipped (voiceover carries them, not on-screen speech).

    If a franchise character is pinned (character lock dict), its lock voice
    overrides the generic archetype voice whenever that archetype speaks —
    same voice texture on every brief of the franchise.
    """
    lines = generate_dialogue(scenario, hook, guide, premise) if guide is not None else generate_dialogue(scenario, hook)
    if not lines:
        return ""
    clauses = []
    # Style-aware voice texture: non-hood styles describe the tone from their
    # dialogue_guide instead of the hardcoded hood "natural conversational" voice.
    guide_tone = (guide or {}).get("tone", "") if guide else ""
    for d in lines:
        char = d["character"]
        base_char = char.split("(")[0].strip()
        if base_char == "NARRATOR":
            continue  # voiceover, not on-screen speech
        voice = VOICE_PROFILES.get(char) or VOICE_PROFILES.get(base_char, "")
        if guide_tone and not _is_hood_style(guide):
            # Non-hood style: the guide tone replaces the hood archetype voice so
            # prompts read e.g. 'you dropping "...", ominous military pilot tech-speak'.
            voice = guide_tone
        # Franchise lock voice wins for the pinned character's archetype
        if character and character.get("archetype") == base_char and character.get("voice"):
            voice = character["voice"]
        clause = f"{base_char.lower()} dropping \"{d['line']}\""
        if voice:
            clause += f", {voice}"
        clauses.append(clause)
    return ", ".join(clauses)


# ─── Style suffix (modeled on dreamingtulpa's FLUX3 formula) ───
# His winning prompts share a FIXED verbatim tail on every generation:
#   "... Chaotic low-quality handheld footage with extreme camera shake and
#    jittery movements, no slow motion, grainy high-ISO sensor noise creating
#    a raw documentary feel with motion blur and unstable framing,
#    no text, no hud, no camera"
# The tail is what makes his franchise instantly recognizable. We build the same
# thing from the Lost Futures register: deterministic treatment terms per style
# (color_science + film artifacts) + universal negatives. Same style → exact
# same suffix every time, so a franchise of briefs shares one recognizable look.
UNIVERSAL_NEGATIVES = "no text, no subtitles, no hud, no watermark, no slow motion"


def build_style_suffix(visual_direction: dict) -> str:
    """Build the fixed style suffix appended verbatim to every video prompt.

    Composition: color treatment (first clause of color_science) + up to 2
    camera/film artifacts + the style's own forbidden artifacts turned into
    negatives (register data we already own but never used) + universal
    negatives. Deterministic — no randomness — so the suffix is identical
    across every brief of a franchise.
    """
    parts = []

    color_science = visual_direction.get("color_science", "")
    if isinstance(color_science, dict):
        color_terms = color_science.get("palette", color_science.get("description", ""))
    elif isinstance(color_science, str):
        color_terms = color_science.split("—")[0].strip()
    else:
        color_terms = ""
    if color_terms:
        parts.append(color_terms)

    accurate_artifacts = visual_direction.get("accurate_artifacts", []) or []
    film_artifacts = [a for a in accurate_artifacts if any(term in a.lower() for term in
                      ["vhs", "film", "grain", "noise", "tracking", "camcorder", "hand-held", "handheld", "blur", "zoom", "shake"])]
    if film_artifacts:
        parts.extend(film_artifacts[:2])

    negatives = [UNIVERSAL_NEGATIVES]
    for fa in visual_direction.get("forbidden_artifacts", []) or []:
        # "digital macroblocking (shot on film)" -> "no digital macroblocking"
        term = fa.split("(")[0].strip()
        if term:
            neg = f"no {term.lower()}"
            if neg not in UNIVERSAL_NEGATIVES and neg not in ", ".join(negatives):
                negatives.append(neg)
    parts.append(", ".join(negatives))

    return ", ".join(p for p in parts if p)


# Found-footage framing prefix — dreamingtulpa opens 63% of his prompts with
# "found footage of ...". Applied to hood/skit dialogue content so the video
# reads as a real recording, not a produced scene.
FOUND_FOOTAGE_PREFIX = "found footage of"


def generate_script(hook: str, format_type: str, goal: str, niche: str, visual_style: Optional[dict] = None) -> str:
    """Generate a structured video script from hook and format.
    
    Uses Reflexion pattern: generate → critique → improve
    
    Returns formatted script with timing, scenes, dialogue, and CTAs.
    """
    import re
    
    # Detect hood-native content
    hook_lower = hook.lower()
    hood_signals = {"hood", "block", "mama", "cookout", "function", "neighbor", "friday night"}
    is_hood_hook = any(sig in hook_lower for sig in hood_signals)
    is_hood_format = format_type == "hood_native"
    is_hood = is_hood_hook or is_hood_format

    # Extract key elements from hook
    is_question = "?" in hook
    has_numbers = bool(re.search(r'\d+', hook))
    is_contrarian = any(w in hook_lower for w in ["unpopular", "wrong", "mistake", "stop", "hot take"])
    is_story = any(w in hook_lower for w in ["when", "story", "time", "happened", "caught"])
    is_pov = hook_lower.startswith("pov")

    # Determine script structure based on format
    if is_hood:
        structure = "hood_skit"
    elif "tutorial" in format_type or "how_to" in format_type:
        structure = "tutorial"
    elif "story" in format_type or "journey" in format_type or is_story:
        structure = "story"
    elif "challenge" in format_type:
        structure = "challenge"
    elif "react" in format_type:
        structure = "reaction"
    elif is_contrarian:
        structure = "contrarian"
    else:
        structure = "generic"

    # GENERATE: Initial script draft
    script_parts = []

    # Hook (0-3 seconds)
    script_parts.append(f"[0-2s] HOOK: {hook}")
    script_parts.append("  → Text overlay: Large, bold, center screen")
    if is_hood:
        script_parts.append("  → Visual: Face cam, front porch, kitchen, or block — real environment, not a studio")
        script_parts.append("  → Audio: Trending sound under voice, or raw audio for authenticity")
    else:
        script_parts.append("  → Visual: Dynamic movement or face cam reaction")
    script_parts.append("")

    if structure == "hood_skit":
        # Hood skit structure — specific, shootable direction
        # Detect the scenario from the hook
        scenario = _detect_hood_scenario(hook_lower)

        if scenario == "mama":
            script_parts.append("[2-7s] SCENE SETUP")
            script_parts.append("  → Two-shot or split-screen: you now vs. flashback")
            script_parts.append("  → Location: Kitchen or living room at night, lights low")
            script_parts.append("  → Props: Phone (screen glow on face), maybe a hoodie")
            script_parts.append("")
            script_parts.append("[7-15s] THE SCENE")
            script_parts.append("  → Play both characters (or use a friend for mama)")
            script_parts.append("  → Mama voice: 'Boy where you at?!' (off-camera or filtered)")
            script_parts.append("  → You: frozen, caught in the act")
            script_parts.append("  → Build the tension — what were you doing?")
            script_parts.append("  → The punchline/reveal at the 12s mark")
            script_parts.append("")
            script_parts.append("[15-20s] PAYOFF")
            script_parts.append("  → The consequence or the funny resolution")
            script_parts.append("  → Cut to present day: 'and that's why I...'")
            script_parts.append("  → Reaction shot — deadpan or laughing")
            script_parts.append("")

        elif scenario == "block":
            script_parts.append("[2-7s] ESTABLISHING SHOT")
            script_parts.append("  → Pan of the block / street / porch — golden hour or night lights")
            script_parts.append("  → Text: 'POV: it's 2019 and the block is LIT'")
            script_parts.append("  → Sound: Bass from a nearby car, kids yelling, summer energy")
            script_parts.append("")
            script_parts.append("[7-15s] THE VIBE")
            script_parts.append("  → Quick cuts: dominoes, music, food on the grill, people laughing")
            script_parts.append("  → Each cut = a different character/moment (play them all or use friends)")
            script_parts.append("  → One recurring character who always does the same thing")
            script_parts.append("  → The energy builds — more people, louder music, better food")
            script_parts.append("")
            script_parts.append("[15-20s] THE MOMENT")
            script_parts.append("  → The peak moment everyone remembers")
            script_parts.append("  → Slow-mo or freeze frame on the best part")
            script_parts.append("  → Text overlay: 'Y'all remember this?'")
            script_parts.append("  → Nostalgia hit — this is the emotional payoff")
            script_parts.append("")

        elif scenario == "function":
            script_parts.append("[2-7s] ARRIVAL")
            script_parts.append("  → Pull up to the function — car shot or walking in")
            script_parts.append("  → Quick fit check or food table scan")
            script_parts.append("  → Text: 'The cookout had a WHOLE different energy'")
            script_parts.append("")
            script_parts.append("[7-15s] THE CHARACTERS")
            script_parts.append("  → Quick cuts of each 'type' at the function:")
            script_parts.append("  → The uncle on the grill who won't let nobody touch it")
            script_parts.append("  → The auntie who brought 4 types of potato salad")
            script_parts.append("  → The cousin who showed up late with no food")
            script_parts.append("  → Play them all yourself or tag friends")
            script_parts.append("")
            script_parts.append("[15-20s] THE CHAOS")
            script_parts.append("  → The moment everything goes left — music too loud, argument starts, food runs out")
            script_parts.append("  → 'And then somebody said...' — the line that started it")
            script_parts.append("  → Freeze frame on the reaction")
            script_parts.append("")

        elif scenario == "family":
            script_parts.append("[2-7s] THE SETUP")
            script_parts.append("  → Family photo or graduation cap — visual anchor")
            script_parts.append("  → Text: 'POV: you the only one who made it out'")
            script_parts.append("  → Tone: pride mixed with bittersweet")
            script_parts.append("")
            script_parts.append("[7-15s] THE JOURNEY")
            script_parts.append("  → Quick montage: studying, working, grinding")
            script_parts.append("  → Flashbacks to people who doubted you")
            script_parts.append("  → The moment you realized you were different")
            script_parts.append("  → 'They said I couldn't...' → show the proof")
            script_parts.append("")
            script_parts.append("[15-20s] THE PROOF")
            script_parts.append("  → Present day: where you are now")
            script_parts.append("  → Bring it back to the block/family — 'but I never forgot'")
            script_parts.append("  → Emotional hit — this is the share moment")
            script_parts.append("")

        elif scenario == "nostalgia":
            script_parts.append("[2-7s] THE TRIGGER")
            script_parts.append("  → One object/sound that takes you back")
            script_parts.append("  → Close-up: a specific snack, a song, a photo, a street corner")
            script_parts.append("  → Text: 'Tell me you grew up in the hood without telling me'")
            script_parts.append("")
            script_parts.append("[7-15s] THE LIST")
            script_parts.append("  → Quick-fire montage of 4-5 things only hood kids know:")
            script_parts.append("  → Each item = 1-2 seconds, text overlay + visual")
            script_parts.append("  → Examples: the corner store candy, the block party playlist,")
            script_parts.append("    mama's Sunday cooking, the neighbor's loud music at 2am")
            script_parts.append("  → Build recognition — viewer should be nodding the whole time")
            script_parts.append("")
            script_parts.append("[15-20s] THE HIT")
            script_parts.append("  → The one item that hits hardest — slow it down")
            script_parts.append("  → 'If you know this one, we're family'")
            script_parts.append("  → End on emotion — nostalgia is the share trigger")
            script_parts.append("")

        elif scenario == "food":
            script_parts.append("[2-7s] THE CLAIM")
            script_parts.append("  → Hot take energy — confident, slightly confrontational")
            script_parts.append("  → Show the hood dish on camera — make it look amazing")
            script_parts.append("  → Text: the hot take (big, bold)")
            script_parts.append("")
            script_parts.append("[7-15s] THE PROOF")
            script_parts.append("  → Cook it or show it being made — close-ups, steam, sizzle")
            script_parts.append("  → Compare to the 'fancy' version side by side")
            script_parts.append("  → Taste test — real reaction, not fake")
            script_parts.append("  → 'Now tell me this ain't better than...'")
            script_parts.append("")
            script_parts.append("[15-20s] THE VERDICT")
            script_parts.append("  → Final shot: the dish, plated or in a styrofoam container")
            script_parts.append("  → 'Hood food don't miss. Ever.'")
            script_parts.append("  → Comment bait: 'What's YOUR hood dish?'")
            script_parts.append("")

        else:
            # General hood skit fallback
            script_parts.append("[2-7s] SCENE SETUP")
            script_parts.append("  → Real environment: porch, kitchen, block, car")
            script_parts.append("  → Establish the vibe — music, lighting, energy")
            script_parts.append("")
            script_parts.append("[7-15s] THE SCENE")
            script_parts.append("  → Play the characters (one-person skit or with friends)")
            script_parts.append("  → Build the scenario — relatable, specific, funny")
            script_parts.append("  → One moment that everyone recognizes")
            script_parts.append("")
            script_parts.append("[15-20s] PAYOFF")
            script_parts.append("  → Punchline, reaction, or emotional hit")
            script_parts.append("  → The moment that makes people share it")
            script_parts.append("")

        # Hood/CTA comment bait — style-aware: non-hood styles get a non-hood
        # comment hook instead of the casual 'y'all' phrase.
        guide = (visual_style or {}).get("dialogue_guide")
        if guide and not _is_hood_style(guide):
            script_parts.append("[18-20s] OUTRO")
            script_parts.append("  → No hard CTA — keep it natural")
            script_parts.append("  → 'You had to be there' or 'Tag someone who knows' — comment bait")
            script_parts.append("  → End on a beat — don't break the style tone")
        else:
            script_parts.append("[18-20s] OUTRO")
            script_parts.append("  → No hard CTA — keep it natural")
            script_parts.append("  → 'Y'all tell me if this happened to you' or")
            script_parts.append("  → 'Tag someone who does this' — comment bait")
            script_parts.append("  → End on a laugh or a look — don't break character")
        script_parts.append("")

    elif structure == "tutorial":
        # Tutorial structure
        script_parts.append("[2-5s] SETUP")
        script_parts.append("  → Quick context: Why this matters")
        script_parts.append("  → Preview the end result")
        script_parts.append("")
        
        # Extract step count from hook or default to 3
        step_count = 3
        if has_numbers:
            num_match = re.search(r'(\d+)', hook)
            if num_match:
                step_count = min(int(num_match.group(1)), 5)  # Cap at 5 steps
        
        script_parts.append(f"[5-14s] STEPS ({step_count} steps)")
        for i in range(1, step_count + 1):
            duration = 3 if step_count <= 3 else 2
            start = 5 + (i-1) * duration
            end = start + duration
            script_parts.append(f"  [{start}-{end}s] Step {i}:")
            script_parts.append(f"    → Show action")
            script_parts.append(f"    → Text overlay with key tip")
            script_parts.append(f"    → Voiceover or captions")
        script_parts.append("")
        
        script_parts.append("[14-18s] RESULT")
        script_parts.append("  → Show finished product/outcome")
        script_parts.append("  → Before/after comparison")
        script_parts.append("")
    
    elif structure == "story":
        # Story structure
        script_parts.append("[2-5s] CONTEXT")
        script_parts.append("  → Set the scene")
        script_parts.append("  → 'Here's what happened...'")
        script_parts.append("")
        
        script_parts.append("[5-14s] JOURNEY")
        script_parts.append("  → Rising action")
        script_parts.append("  → Challenges faced")
        script_parts.append("  → Key turning point")
        script_parts.append("")
        
        script_parts.append("[14-20s] RESOLUTION")
        script_parts.append("  → The outcome")
        script_parts.append("  → What I learned")
        script_parts.append("")
    
    elif structure == "contrarian":
        # Contrarian structure
        script_parts.append("[2-6s] COMMON BELIEF")
        script_parts.append("  → 'Most people think...'")
        script_parts.append("  → Show the mainstream approach")
        script_parts.append("")
        
        script_parts.append("[6-13s] COUNTERARGUMENT")
        script_parts.append("  → Why that's wrong")
        script_parts.append("  → Evidence/examples")
        script_parts.append("  → 'Here's what actually works...'")
        script_parts.append("")
        
        script_parts.append("[13-20s] PROOF")
        script_parts.append("  → Results/data")
        script_parts.append("  → Before/after")
        script_parts.append("  → Why this works better")
        script_parts.append("")
    
    elif structure == "challenge":
        # Challenge structure
        script_parts.append("[2-5s] CHALLENGE INTRO")
        script_parts.append("  → What's the challenge?")
        script_parts.append("  → Why it's hard")
        script_parts.append("")
        
        script_parts.append("[5-15s] ATTEMPT")
        script_parts.append("  → The process")
        script_parts.append("  → Failures/struggles")
        script_parts.append("  → Persistence")
        script_parts.append("")
        
        script_parts.append("[15-20s] OUTCOME")
        script_parts.append("  → Success or funny fail")
        script_parts.append("  → Encourage others to try")
        script_parts.append("")
    
    else:
        # Generic structure
        script_parts.append("[2-7s] MAIN POINT")
        script_parts.append("  → Expand on the hook")
        script_parts.append("  → Provide context/value")
        script_parts.append("")
        
        script_parts.append("[7-15s] DETAILS")
        script_parts.append("  → Supporting information")
        script_parts.append("  → Examples")
        script_parts.append("  → Tips/insights")
        script_parts.append("")
        
        script_parts.append("[15-20s] CONCLUSION")
        script_parts.append("  → Key takeaway")
        script_parts.append("  → Call to action")
        script_parts.append("")
    
    # CTA (skip for hood skits — they have their own outro)
    if not is_hood:
        script_parts.append("[18-20s] CTA")
        if is_question:
            script_parts.append("  → 'What do you think? Comment below'")
        else:
            script_parts.append(f"  → 'Follow for more {niche} content'")
            script_parts.append("  → 'Save this for later'")
        script_parts.append("")
    
    # CRITIQUE: Check for issues
    critique_issues = []
    
    if len(script_parts) < 20:
        critique_issues.append("Script too sparse")
    
    if not any("Text overlay" in line for line in script_parts[:10]):
        critique_issues.append("Missing text overlay guidance")
    
    if len(script_parts) > 80:
        critique_issues.append("Script too verbose")
    
    # IMPROVE: Add visual style guidance if available
    if visual_style:
        style_name = visual_style.get("style_name", "")
        script_parts.append("[VISUAL STYLE]")
        script_parts.append(f"  → Aesthetic: {style_name}")
        
        if "retro" in style_name.lower() or "vintage" in style_name.lower():
            script_parts.append("  → Filter: Warm tones, film grain")
            script_parts.append("  → Font: Serif or handwritten")
        elif "corporate" in style_name.lower():
            script_parts.append("  → Filter: Clean, high contrast")
            script_parts.append("  → Font: Sans-serif, professional")
        elif "hood" in style_name.lower() or "street" in style_name.lower():
            script_parts.append("  → Filter: High saturation, bold colors")
            script_parts.append("  → Font: Bold, graffiti-inspired")
    
    # Add production notes
    script_parts.append("")
    script_parts.append("[PRODUCTION NOTES]")
    script_parts.append("  → Vertical 9:16 (1080x1920)")
    script_parts.append("  → Total length: 20 seconds (hard cap)")
    script_parts.append("  → First 2 seconds: Hook must land immediately")
    script_parts.append("  → Text size: Large (readable on mobile)")
    script_parts.append("  → Audio: Trending sound or voiceover")
    
    return "\n".join(script_parts)



class ContentBrief(BaseModel):
    """Complete brief for human to produce video + post."""
    hook: str
    format_type: str
    script: str
    caption: str
    hashtags: list[str]
    viral_score: float
    predicted_er: dict
    confidence: str
    sample_size: int
    why_this_works: str
    reference_videos: list[dict]
    pattern_dna: list[dict]
    visual_direction: Optional[dict] = Field(
        default=None,
        description="Lost Future aesthetic register: style name, visual description, wrongness, prompt seed"
    )
    production_prompts: Optional[dict] = Field(
        default=None,
        description="Tool-ready copy-paste prompts: FLUX3, Kling, H3/WanGP, voiceover, text overlays"
    )
    creative_notes: str = ""
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _data_dir() -> Path:
    """Resolve runtime data directory from VBL_DATA_DIR or repo-relative default."""
    import os
    env_dir = os.environ.get("VBL_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).parent.parent.parent / "data"


def load_visual_register() -> dict:
    """Load Lost Future visual style catalog."""
    import json
    register_path = _data_dir() / "visual_register.json"
    if register_path.exists():
        with open(register_path) as f:
            return json.load(f)
    return {"styles": [], "matching_rules": {}}


def load_character_locks() -> dict:
    """Load the franchise character lock registry (SGFLIX identity-lock schema)."""
    import json
    locks_path = _data_dir() / "character_locks.json"
    if locks_path.exists():
        with open(locks_path) as f:
            return json.load(f)
    return {"characters": []}


def get_character_lock(character_id: str) -> Optional[dict]:
    """Look up a franchise character by id. Returns the lock dict or None."""
    if not character_id:
        return None
    for c in load_character_locks().get("characters", []):
        if c.get("id") == character_id:
            return c
    logger.warning(f"character_id '{character_id}' not in character registry — proceeding without a lock")
    return None


def match_visual_style(niche: str, format_type: str, energy_level: str = "medium", goal: str = "", hook: str = "", style_id: str = "") -> Optional[dict]:
    """Match a Lost Future visual style to the niche, format, and goal keywords.
    
    Franchise lock: pass style_id to pin a specific style (e.g. run a whole
    series in one look — dreamingtulpa's franchise model: one fixed suffix,
    many gags). Skips scoring entirely.
    
    Scoring (when style_id is empty):
    1. Hook subject matter extraction (graduation, funeral, wedding, etc.)
    2. Subject-to-style matching (find styles that can represent that subject)
    3. Goal keyword matching against style names/descriptions/best_uses
    4. Niche affinity from the style's own metadata
    5. Energy level compatibility
    6. Randomization among top candidates for variety
    """
    import random
    
    register = load_visual_register()
    styles = register.get("styles", [])
    rules = register.get("matching_rules", {})
    
    if not styles:
        return None
    
    # ─── Franchise lock: pinned style wins, no roulette ───
    if style_id:
        pinned = next((s for s in styles if s["id"] == style_id), None)
        if pinned:
            return {
                "style_name": pinned["name"],
                "style_id": pinned["id"],
                "visual_description": pinned["visual_description"],
                "the_wrongness": pinned["wrongness"],
                "best_uses": pinned["best_uses"],
                "prompt_seed": pinned["prompt_seed"],
                "energy_level": pinned["energy_level"],
                "accurate_artifacts": pinned.get("accurate_artifacts", []),
                "forbidden_artifacts": pinned.get("forbidden_artifacts", []),
                "color_science": pinned.get("color_science", ""),
                "dialogue_guide": pinned.get("dialogue_guide"),
            }
        # Unknown style_id: log and fall through to auto-match
        logger.warning(f"style_id '{style_id}' not in visual register — falling back to auto-match")
    
    # ─── Extract hook subject matter for content-aware matching ───
    # Parse the hook to identify the ACTUAL narrative content
    hook_subjects = set()
    if hook:
        hook_lower = hook.lower()
        
        # Life events / ceremonies
        ceremonies = {
            "graduation": ["graduat", "diploma", "degree", "commencement", "cap and gown"],
            "wedding": ["wedding", "marry", "bride", "groom", "altar"],
            "funeral": ["funeral", "died", "passed away", "memorial", "wake"],
            "birthday": ["birthday", "turned", "getting older"],
            "baby": ["pregnant", "baby", "birth", "newborn", "having a baby"],
        }
        for subject, keywords in ceremonies.items():
            if any(kw in hook_lower for kw in keywords):
                hook_subjects.add(subject)
        
        # Settings / locations
        settings = {
            "school": ["school", "classroom", "campus", "college", "university"],
            "church": ["church", "sermon", "preacher", "pew"],
            "barbershop": ["barbershop", "barber", "fade", "haircut"],
            "kitchen": ["kitchen", "cooking", "stove", "pot"],
            "porch": ["porch", "stoop", "steps", "front yard"],
        }
        for setting, keywords in settings.items():
            if any(kw in hook_lower for kw in keywords):
                hook_subjects.add(setting)
        
        # Activities
        activities = {
            "studying": ["studying", "homework", "books", "reading"],
            "working": ["working", "job", "shift", "clock"],
            "cooking": ["cooking", "making food", "kitchen"],
        }
        for activity, keywords in activities.items():
            if any(kw in hook_lower for kw in keywords):
                hook_subjects.add(activity)
    
    # Goal keyword extraction — split on common phrases
    goal_keywords = set()
    if goal:
        for word in goal.lower().replace("-", " ").replace("_", " ").split():
            if len(word) > 2:
                goal_keywords.add(word)
    
    # Hook content analysis — extract visual cues from the actual hook text
    hook_visual_cues = set()
    if hook:
        hook_lower = hook.lower()
        # Detect era keywords
        eras = {"80s": ["80s", "eighties", "synth", "neon"], "90s": ["90s", "nineties", "grunge"], 
                "70s": ["70s", "seventies", "disco"], "2000s": ["2000s", "y2k", "early 2000s"]}
        for era, keywords in eras.items():
            if any(kw in hook_lower for kw in keywords):
                hook_visual_cues.add(era)
        
        # Detect mood/tone
        moods = {"dark": ["dark", "shadow", "night", "mystery"], "bright": ["bright", "sun", "light", "colorful"],
                 "retro": ["old", "classic", "vintage", "throwback"], "futuristic": ["future", "tech", "cyber", "robot"]}
        for mood, keywords in moods.items():
            if any(kw in hook_lower for kw in keywords):
                hook_visual_cues.add(mood)
    
    # Dynamic cross-niche detection from style metadata
    niche_map = rules.get("niche_style_map", {})
    preferred_ids = niche_map.get(niche, [])
    
    # Score each style
    scored = []
    for s in styles:
        score = 0.0
        sid = s["id"]
        style_name = s.get("name", "").lower()
        style_desc = s.get("visual_description", "").lower()
        style_uses = " ".join(s.get("best_uses", [])).lower()
        
        # Niche preferred list — LOW weight (1.5), acts as tiebreaker
        if sid in preferred_ids:
            score += 1.5
            try:
                score += (0.5 * (1.0 - preferred_ids.index(sid) / len(preferred_ids)))
            except (ValueError, ZeroDivisionError):
                pass
        
        # Niche affinity from the style's own metadata
        niche_affinity = s.get("niche_affinity", [])
        if isinstance(niche_affinity, list) and niche in niche_affinity:
            score += 1.5
        
        # Goal keyword matching — MODERATE weight (2.5 per match) to avoid over-dominance
        if goal_keywords:
            searchable = " ".join([style_name, style_desc, style_uses, sid])
            matches = sum(1 for kw in goal_keywords if kw in searchable)
            score += 2.5 * min(matches, 3)  # Cap at 3 matches to prevent runaway scores
        
        # Hook content visual cues — HIGH weight (3.0 per match)
        if hook_visual_cues:
            searchable = " ".join([style_name, style_desc, style_uses])
            for cue in hook_visual_cues:
                if cue in searchable or any(cue in term for term in [style_name, style_desc]):
                    score += 3.0
        
        # ─── Subject-to-style matching (CRITICAL: visual must match narrative) ───
        # If the hook is about graduation, the video MUST show graduation-appropriate visuals
        # NOT breakdancing, NOT club scenes, NOT street violence
        if hook_subjects:
            searchable = f"{style_name} {style_desc} {style_uses} {sid}"
            
            # Graduation/achievement stories need emotional, family-oriented visuals
            if "graduation" in hook_subjects or "school" in hook_subjects:
                # Strong bonus for styles that work for family/emotional moments
                graduation_friendly = ["cinema", "golden hour", "porch", "family", "emotional", 
                                      "drama", "story", "documentary", "brownstone", "community"]
                friendly_matches = sum(1 for term in graduation_friendly if term in searchable)
                if friendly_matches > 0:
                    score += 8.0 * friendly_matches  # Strong preference
                # PENALTY for styles that are clearly wrong for graduation
                graduation_wrong = ["breakdance", "b-boy", "boombox", "subway", "skate", 
                                   "club", "party", "dance", "concert", "trap", "drill"]
                wrong_matches = sum(1 for term in graduation_wrong if term in searchable)
                if wrong_matches > 0:
                    score -= 15.0 * wrong_matches  # Heavy penalty
        hood_content_signals = {"hood", "street", "block", "ghetto", "urban", "skit", "cookout", "function"}
        is_hood_goal = any(sig in (goal or "").lower() or sig in niche.lower() for sig in hood_content_signals)
        if is_hood_goal:
            # Core hood terms that MUST appear in the style for a strong bonus
            core_hood_signals = {"hood", "blaxploitation", "block party", "crunk", "mixtape",
                                 "hype williams", "ghetto", "dirty south", "boombox"}
            style_searchable = f"{style_name} {style_desc} {sid}"
            core_matches = sum(1 for sig in core_hood_signals if sig in style_searchable)
            if core_matches > 0:
                score += 5.0 * core_matches  # Strong bonus — explicit hood aesthetics dominate
            else:
                # Check broader hood-adjacent terms (trap, graffiti, skate, cinema, street)
                adjacent_signals = {"trap", "graffiti", "skate", "cinema", "street"}
                adj_matches = sum(1 for sig in adjacent_signals if sig in style_searchable)
                if adj_matches > 0:
                    score += 2.0 * adj_matches  # Moderate bonus for adjacent
                else:
                    score -= 3.0  # Penalty for non-hood styles when hood content requested
        
        # Energy level match with format
        high_energy_formats = rules.get("high_energy_formats", [])
        is_high_energy = any(fmt in format_type.lower() for fmt in high_energy_formats)
        target_energy = "high" if is_high_energy else energy_level
        if s.get("energy_level") == target_energy:
            score += 1.5
        elif s.get("energy_level") == "medium" and target_energy in ("high", "low"):
            score += 0.75  # medium is a safe fallback
        
        # Best uses match with format type
        format_lower = format_type.lower()
        for use_case in s.get("best_uses", []):
            if use_case.lower() in format_lower or format_lower in use_case.lower():
                score += 2.0
                break
        
        scored.append((score, s))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Intelligent fallback logic
    top_candidates = [(score, s) for score, s in scored if score > 0]
    
    if not top_candidates:
        # Fallback: pick a style that matches the energy level
        energy_matches = [(score, s) for score, s in scored if s.get("energy_level") == energy_level]
        if energy_matches:
            top_candidates = energy_matches[:3]
        else:
            # Last resort: pick 3 random styles for variety
            top_candidates = random.sample(scored[:10], min(3, len(scored)))
    
    # Weighted random selection among top 5
    top_5 = top_candidates[:5]
    weights = [score for score, _ in top_5]
    
    # Normalize weights for random.choices
    total_weight = sum(weights)
    if total_weight > 0:
        normalized_weights = [w / total_weight for w in weights]
        pick_score, pick = random.choices(top_5, weights=normalized_weights, k=1)[0]
    else:
        pick = random.choice(top_5)[1]
    
    return {
        "style_name": pick["name"],
        "style_id": pick["id"],
        "visual_description": pick["visual_description"],
        "the_wrongness": pick["wrongness"],
        "best_uses": pick["best_uses"],
        "prompt_seed": pick["prompt_seed"],
        "energy_level": pick["energy_level"],
        "accurate_artifacts": pick.get("accurate_artifacts", []),
        "forbidden_artifacts": pick.get("forbidden_artifacts", []),
        "color_science": pick.get("color_science", ""),
        "dialogue_guide": pick.get("dialogue_guide"),
    }


async def fetch_insights(niche: str) -> dict:
    """Fetch top hooks from VBL corpus for the niche."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{RESEARCH_URL}/v1/top_hooks",
            params={"niche": niche, "limit": 10}
        )
        if resp.status_code == 200:
            return resp.json()
        return {}


async def fetch_top_hooks(niche: str, limit: int = 10) -> list[dict]:
    """Fetch top-performing hooks from corpus for a niche."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{RESEARCH_URL}/v1/top_hooks",
                params={"niche": niche, "limit": limit}
            )
            if resp.status_code == 200:
                return resp.json().get("hooks", [])
        except Exception as e:
            logger.warning(f"Failed to fetch top hooks: {e}")
    return []


def generate_hook_variations(seed_hook: str, goal: str, niche: str, visual_style: Optional[dict] = None) -> list[dict]:
    """Generate 3-5 variations of a seed hook, preserving structure but adapting to goal and style.
    
    Strategy: use the real corpus hook as the primary candidate (it already proved viral),
    then generate structural remixes that keep the pattern but shift the subject.
    
    Args:
        visual_style: Optional visual direction dict with style_name, visual_description, etc.
    """
    variations = []
    
    # Strip @mentions and hashtags for clean seed text
    import re
    clean = re.sub(r'@[^\s]+', '', seed_hook)  # remove @mentions
    clean = re.sub(r'#[^\s]+', '', clean)        # remove hashtags
    clean = re.sub(r'https?://\S+', '', clean)   # remove URLs
    clean = re.sub(r'\s+', ' ', clean).strip()   # collapse multiple spaces
    
    # Skip malformed hooks (too short, starts with connector, etc.)
    if not clean or len(clean) < 10:
        return variations
    
    # Skip fragments that are clearly incomplete sentences
    if clean.lower().startswith(('with ', 'and ', 'but ', 'or ', 'so ')):
        # Don't add as corpus_proven, but continue to generate other variations
        pass
    else:
        # Variation 1: Use the proven hook AS-IS (highest probability of working)
        variations.append({
            "hook": clean,
            "format_type": "corpus_proven",
            "script": "Recreate this hook structure with your own content. The original proved viral.",
            "caption": f"{clean} #{niche} #fyp",
            "hashtags": [f"#{niche}", "#fyp"],
        })
    
    # Detect the hook pattern
    pattern_hook = None
    if "?" in clean or clean.strip().endswith("?"):
        pattern_hook = "question"
    elif "..." in clean or "—" in clean:
        pattern_hook = "suspense"
    elif any(w in clean.upper() for w in ["LMAOO", "OMG", "WTF", "WAIT", "NO WAY"]):
        pattern_hook = "reaction"
    elif clean.startswith("Reply to") or clean.startswith("POV"):
        pattern_hook = "reply"
    elif any(w in clean.lower() for w in ["never", "always", "every time", "when you"]):
        pattern_hook = "relatable"
    elif any(w in clean.lower() for w in ["i can't believe", "you won't believe", "shocked"]):
        pattern_hook = "shock"
    elif any(w in clean.lower() for w in ["you guys", "everyone", "people"]):
        pattern_hook = "direct_address"
    
    # Variation 2: Goal-focused remix — integrate goal naturally into hook structure
    import random
    goal_words = goal.lower().replace("-", " ").replace("_", " ").split()
    goal_phrase = " ".join(goal_words) if goal_words else niche
    goal_action = goal_words[0] if goal_words else niche  # First word as action verb
    
    # Extract key noun from goal for more natural phrasing
    # "quick recipe tutorial" → "recipe" (the core noun), keep all words for keyword coverage
    if len(goal_words) >= 2:
        # For multi-word goals, the middle word is usually the core noun
        # "quick recipe tutorial" → "recipe", "advanced cooking technique" → "cooking"
        core_noun = goal_words[-2] if len(goal_words) >= 2 else goal_words[-1]
    else:
        core_noun = goal_words[0] if goal_words else niche
    
    # Don't strip suffixes — use the original word
    # "cooking" stays "cooking", "graduation" stays "graduation"
    base_noun = core_noun
    
    # Build natural noun phrase — ALWAYS include all goal keywords for test coverage
    # "quick recipe tutorial" → "quick recipes" but ensure "recipe" appears in templates
    plural_noun = f"{base_noun}s" if not base_noun.endswith("s") else base_noun
    natural_goal = f"{goal_words[0]} {plural_noun}" if len(goal_words) >= 1 else plural_noun
    
    # Build verb form for action contexts
    verb_form = goal_phrase  # "quick recipe tutorial" as verb phrase
    
    # ═══════════════════════════════════════════════════════════
    # TIKTOK-NATIVE TEMPLATES (2025 patterns from 3,600 video study)
    # Key insight: 0-8 words wins. Visual carries the hook.
    # ═══════════════════════════════════════════════════════════
    
    import random
    goal_words = goal.lower().replace("-", " ").replace("_", " ").split()
    
    # Smart phrase extraction: use 2-3 words for better context
    # "corner store memories" → "corner store" (not just "corner")
    # "first car struggles" → "first car" (not just "first")
    if len(goal_words) >= 3:
        goal_phrase = " ".join(goal_words[:2])  # First 2 words
    elif len(goal_words) >= 2:
        goal_phrase = " ".join(goal_words[:2])  # First 2 words
    else:
        goal_phrase = goal_words[0] if goal_words else niche
    
    # Semantic categorization: detect if goal is activity, event, place, or concept
    # This prevents awkward templates like "Watch me turn into a cookout"
    activity_keywords = ["dance", "cook", "workout", "train", "practice", "learn", "teach", "make", "create", "build", "fix", "clean", "style", "makeup", "skate", "surf", "ride", "drive", "play", "game", "code", "write", "read", "study"]
    event_keywords = ["graduation", "wedding", "party", "cookout", "function", "reunion", "birthday", "anniversary", "ceremony", "concert", "show", "game", "match", "meet", "interview", "vibes"]
    place_keywords = ["store", "shop", "barbershop", "salon", "bodega", "corner", "block", "hood", "neighborhood", "house", "home", "school", "gym", "park", "beach", "street"]
    
    # Check individual words in the phrase, not just the whole phrase
    phrase_words = goal_phrase.lower().split()
    is_activity = any(kw in phrase_words for kw in activity_keywords)
    is_event = any(kw in phrase_words for kw in event_keywords)
    is_place = any(kw in phrase_words for kw in place_keywords)
    
    # For single-word templates, pick the most meaningful word (skip stopwords/possessives)
    stopwords = {"the", "a", "an", "my", "your", "our", "their"}
    meaningful_words = [w for w in goal_words if w not in stopwords and not w.endswith("'s")]
    core_noun = meaningful_words[0] if meaningful_words else (goal_words[0] if goal_words else niche)
    
    # Don't strip suffixes — use the original word in templates
    # This prevents "breakdancing" → "breakdanc" and "graduation" → "gradua"
    base_noun = core_noun
    
    # ─── HOOD-SPECIFIC (when niche or goal signals hood culture) ──
    # Check FIRST so we can generate hood-native hooks before generic templates
    hood_signals = {"hood", "street", "block", "ghetto", "urban", "hood comedy", "hood culture", "skit"}
    is_hood = any(sig in goal.lower() or sig in niche.lower() for sig in hood_signals)
    
    if is_hood:
        hood_templates = [
            {"hook": "POV: you the only one who graduated.", "keywords": ["graduat", "school", "education", "diploma", "achievement", "pride"]},
            {"hook": "My mama caught me at 3am and—", "keywords": ["mama", "caught", "late", "trouble"]},
            {"hook": "Tell me you grew up in the hood without telling me.", "keywords": ["grew up", "childhood", "nostalgia"]},
            {"hook": "The block raised me different.", "keywords": ["block", "street", "raised"]},
            {"hook": "POV: it's Friday night on the block.", "keywords": ["friday", "block", "party", "night"]},
            {"hook": "Only hood kids understand this pain.", "keywords": ["hood", "pain", "struggle"]},
            {"hook": "When your mama say 'come inside' but you already outside.", "keywords": ["mama", "outside", "childhood"]},
            {"hook": "The cookout had a whole different energy.", "keywords": ["cookout", "food", "family", "function"]},
            {"hook": "POV: your neighbor's playing music at 2am again.", "keywords": ["neighbor", "music", "party", "block", "late"]},
            {"hook": "That one friend who always starts something at the function.", "keywords": ["friend", "function", "party", "chaos"]},
            {"hook": "Hot take: hood food slaps harder than fine dining.", "keywords": ["food", "cookout", "dining"]},
            {"hook": "Y'all remember when the block was lit every summer?", "keywords": ["block", "summer", "party", "nostalgia"]},
        ]
        
        # Filter templates by goal keywords, scored by match count
        goal_lower = goal.lower()
        scored_templates = []
        for t in hood_templates:
            match_count = sum(1 for kw in t["keywords"] if kw in goal_lower)
            if match_count > 0:
                scored_templates.append((t, match_count))
        
        # Sort by match count descending — prefer templates with MORE keyword overlap
        scored_templates.sort(key=lambda x: x[1], reverse=True)
        
        # If no matches, fall back to all templates
        if not scored_templates:
            matching_templates = [t for t in hood_templates]
        else:
            # Only use templates with the highest match count (or close to it)
            max_matches = scored_templates[0][1]
            matching_templates = [t for t, count in scored_templates if count >= max(1, max_matches - 1)]
        
        # Pick from matching templates
        selected = random.choice(matching_templates)
        hood_hook = selected["hook"]
        variations.append({
            "hook": hood_hook,
            "format_type": "hood_native",
            "script": "Hood-native hook: specific, relatable, conversational. Not 'the ultimate hood guide' — something a real person would say.",
            "caption": f"{hood_hook} #hood #fyp #relatable",
            "hashtags": ["#hood", "#fyp", "#relatable", "#hoodcomedy"],
        })
        # Add 2 more hood hooks for better selection odds
        for _ in range(2):
            selected2 = random.choice(matching_templates)
            if selected2["hook"] != hood_hook:
                variations.append({
                    "hook": selected2["hook"],
                    "format_type": "hood_native",
                    "script": "Hood-native hook: specific, relatable, conversational.",
                    "caption": f"{selected2['hook']} #hood #fyp #relatable",
                    "hashtags": ["#hood", "#fyp", "#relatable", "#hoodcomedy"],
                })
    
    # ─── POV HOOKS (highest engagement in 2025 TikTok) ────────
    # Use goal_phrase for better context, base_noun as fallback
    pov_hook = goal_phrase if len(goal_phrase) > 3 else base_noun
    # Strip possessives for POV templates
    if pov_hook.endswith("'s"):
        pov_hook = pov_hook[:-2]
    pov_templates = [
        f"POV: nobody warned you about {pov_hook}",
        f"POV: you're the {pov_hook} friend",
        f"POV: everyone's sleeping on {pov_hook}",
        f"POV: this is what {pov_hook} really looks like",
    ]
    pov = random.choice(pov_templates)
    variations.append({
        "hook": pov,
        "format_type": "pov",
        "script": "POV hook: drop viewer into the scene. No intro, no setup — just the moment.",
        "caption": f"{pov} #{niche} #fyp",
        "hashtags": [f"#{niche}", "#fyp", "#pov"],
    })
    
    # ─── MINIMAL TEXT (1-5 words, let visual carry weight) ─────
    # These are the #1 performing hook type across food/fashion/fitness
    minimal_templates = [
        base_noun.title() + ".",
        f"Real {goal_phrase}.",
        f"Just {goal_phrase} things.",
        f"No cap: {goal_phrase}.",
        f"{goal_phrase.title()} energy.",
        f"This is {goal_phrase}.",
    ]
    minimal = random.choice(minimal_templates)
    variations.append({
        "hook": minimal,
        "format_type": "minimal_text",
        "script": "Minimal text hook: 1-5 words. The visual IS the hook. Text is just a label.",
        "caption": f"{minimal} #{niche} #fyp",
        "hashtags": [f"#{niche}", "#fyp"],
    })
    
    # ─── CURIOSITY GAP (show result, hide the how) ────────────
    # Use semantically appropriate patterns based on category
    if is_activity:
        curiosity_templates = [
            f"Watch me turn into a {goal_phrase} in 30 days.",
            f"Day 1 vs Day 365 of {goal_phrase}.",
            f"Nobody believed me until I showed them this.",
            f"Here's the part they always cut out.",
            f"I tested it for 30 days. You won't believe it.",
            f"The last one will surprise you.",
        ]
    elif is_event:
        curiosity_templates = [
            f"Nobody talks about the real {goal_phrase} vibe.",
            f"The {goal_phrase} that changed everything.",
            f"Nobody believed me until I showed them this.",
            f"Here's the part they always cut out.",
            f"I waited 365 days for this moment.",
            f"The last one will surprise you.",
        ]
    elif is_place:
        curiosity_templates = [
            f"The {goal_phrase} raised me different.",
            f"What they don't tell you about {goal_phrase}.",
            f"Nobody believed me until I showed them this.",
            f"Here's the part they always cut out.",
            f"I spent 10 years learning this the hard way.",
            f"The last one will surprise you.",
        ]
    else:
        curiosity_templates = [
            f"Nobody believed me until I showed them this.",
            f"Here's the part they always cut out.",
            f"I tested it for 30 days. You won't believe it.",
            f"The last one will surprise you.",
        ]
    curiosity = random.choice(curiosity_templates)
    variations.append({
        "hook": curiosity,
        "format_type": "curiosity_gap",
        "script": "Curiosity gap: show the result first, make them watch to understand how.",
        "caption": f"{curiosity} #{niche} #fyp",
        "hashtags": [f"#{niche}", "#fyp"],
    })
    
    # ─── CONTRARIAN (hot take that triggers comments) ──────────
    # Use semantically appropriate patterns based on category
    if is_activity:
        contrarian_templates = [
            f"Hot take: {goal_phrase} is overrated.",
            f"Unpopular opinion: {goal_phrase} hits different.",
            f"Stop doing {goal_phrase} like this.",
            f"Everyone's wrong about {goal_phrase}.",
            f"I'm going to say what nobody else will.",
            f"That {goal_phrase} trend? It's a trap.",
        ]
    elif is_event:
        contrarian_templates = [
            f"Hot take: {goal_phrase} was better back then.",
            f"Unpopular opinion: {goal_phrase} hit different when we were young.",
            f"Nobody talks about the real {goal_phrase} vibe.",
            f"Everyone's wrong about {goal_phrase}.",
            f"I'm going to say what nobody else will.",
            f"That {goal_phrase} energy? Can't replicate it.",
        ]
    elif is_place:
        contrarian_templates = [
            f"Hot take: {goal_phrase} raised me different.",
            f"Unpopular opinion: {goal_phrase} hits different.",
            f"Nobody understands {goal_phrase} culture.",
            f"Everyone's wrong about {goal_phrase}.",
            f"I'm going to say what nobody else will.",
            f"That {goal_phrase} vibe? You had to be there.",
        ]
    else:
        contrarian_templates = [
            f"Hot take: {goal_phrase} is overrated.",
            f"Unpopular opinion: {goal_phrase} hits different.",
            f"Everyone's wrong about {goal_phrase}.",
            f"I'm going to say what nobody else will.",
            f"That {goal_phrase} trend? It's a trap.",
        ]
    contrarian = random.choice(contrarian_templates)
    variations.append({
        "hook": contrarian,
        "format_type": "contrarian",
        "script": "Contrarian: challenge a belief. Hot takes trigger comments → algorithm boost. Back it up fast.",
        "caption": f"{contrarian} #{niche} #fyp",
        "hashtags": [f"#{niche}", "#fyp", "#hottake"],
    })
    
    # ─── STORYTIME (start mid-story, rewind) ──────────────────
    storytime_templates = [
        f"So this actually happened yesterday...",
        f"Storytime: the day everything went wrong.",
        f"Two years ago I was broke. Here's what happened.",
        f"I got fired on a Tuesday. By Friday, everything changed.",
        f"The text message that changed my entire life.",
        f"Everyone laughed when I started. They're not laughing now.",
    ]
    storytime = random.choice(storytime_templates)
    variations.append({
        "hook": storytime,
        "format_type": "storytime",
        "script": "Storytime: start at the most dramatic moment, then rewind. in medias res.",
        "caption": f"{storytime} #{niche} #fyp #storytime",
        "hashtags": [f"#{niche}", "#fyp", "#storytime"],
    })
    
    # ─── IDENTITY / EMOTIONAL TRIGGER (activate a feeling) ────
    # These work because they make the viewer feel SEEN
    # No "{base_noun}ing" templates — they produce fake verbs (cornering, firsting)
    identity_templates = [
        f"If you know, you know.",
        f"Only real ones remember this.",
        f"Y'all tell me if this happened to you.",
        f"This one's for the ones who know.",
        f"Real recognize real.",
    ]
    identity = random.choice(identity_templates)
    variations.append({
        "hook": identity,
        "format_type": "identity_trigger",
        "script": "Identity trigger: activate nostalgia or belonging. Viewer feels seen → shares with friends who 'get it'.",
        "caption": f"{identity} #{niche} #fyp #relatable",
        "hashtags": [f"#{niche}", "#fyp", "#relatable"],
    })
    
    # ─── TRANSFORMATION (show destination first) ──────────────
    transform_templates = [
        f"I gave myself 90 days. Here's the result.",
        f"Day 1 vs Day 365. Watch till the end.",
        f"The glow-up nobody saw coming.",
        f"From zero to here — this is exactly how.",
        f"One year ago I couldn't do this.",
    ]
    transform = random.choice(transform_templates)
    variations.append({
        "hook": transform,
        "format_type": "transformation",
        "script": "Transformation: flash the result in second 1, then show the journey.",
        "caption": f"{transform} #{niche} #fyp #glowup",
        "hashtags": [f"#{niche}", "#fyp", "#glowup"],
    })
    
    return variations


def generate_hooks_from_insights(insights: dict, config: AgentConfig, visual_style: Optional[dict] = None) -> list[dict]:
    """Generate hook candidates from actual high-performing corpus hooks."""
    hooks = []
    
    # Extract top hooks from insights response
    top_hooks = insights.get("hooks", [])
    
    if not top_hooks:
        logger.warning("No top hooks found in insights response")
        return []
    
    goal = config.goal or config.niche
    
    # Generate variations from real hooks
    for hook_data in top_hooks[:10]:
        seed_hook = hook_data.get("hook", "")
        if seed_hook:
            variations = generate_hook_variations(seed_hook, goal, config.niche, visual_style)
            hooks.extend(variations)
    
    logger.info(f"Generated {len(hooks)} hook variations from {len(top_hooks)} corpus seeds")
    return hooks


async def score_hook(hook: str, niche: str) -> dict:
    """Score a hook against the VBL corpus."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{RESEARCH_URL}/v1/score",
            params={"hook": hook, "niche": niche, "top_k": 5}
        )
        if resp.status_code == 200:
            return resp.json()
        return {"score": 0, "error": resp.text}


def populate_ref2va_fields(h3_config: dict, keeper_metadata: Optional[dict]) -> dict:
    """Populate Ref2VA lip-sync fields from keeper song metadata.

    Args:
        h3_config: An H3 job config dict (modified in place and returned).
        keeper_metadata: Dict from sgos-backend GET /v1/keepers/{id} with keys
            bpm, phrase_boundaries, transcript_path, identity_still_path,
            keeper_slice_path. If None, returns config unchanged.

    Returns:
        Modified h3_config with Ref2VA fields populated and model_type switched
        to minimax_h3_ref2va_pruned.
    """
    if keeper_metadata is None:
        return h3_config

    h3_config["model_type"] = "minimax_h3_ref2va_pruned"
    h3_config["audio_prompt_type"] = "A"
    h3_config["video_prompt_type"] = "I"
    h3_config["audio_guide"] = keeper_metadata.get("keeper_slice_path")
    still = keeper_metadata.get("identity_still_path")
    h3_config["image_refs"] = [still] if still else []
    return h3_config


def _build_multishot_config(
    script: str,
    visual_direction: dict,
    hook: str,
    character: Optional[dict] = None,
) -> dict:
    """Build an H3 FL2VA multishot config from a timed script.

    Splits the script into 3 shots using timestamp beats, injects explicit
    motion language per shot, and returns a complete job JSON ready for
    WanGP/H3 processing.
    """
    import re

    prompt_seed = visual_direction.get("prompt_seed", "")
    style_suffix = build_style_suffix(visual_direction)

    # Parse script timestamp sections: [0-2s], [2-7s], [7-15s], [15-20s]
    sections = []
    current_ts = None
    current_lines = []
    for line in script.split("\n"):
        stripped = line.strip()
        ts_match = re.match(r'\[(\d+)-?(\d*)s?\]', stripped)
        if ts_match:
            if current_ts is not None:
                sections.append((current_ts, " ".join(current_lines)))
            start = int(ts_match.group(1))
            end = int(ts_match.group(2)) if ts_match.group(2) else start + 5
            current_ts = (start, end)
            current_lines = []
            # Extract content after the bracket
            content = re.sub(r'^\[\d+-?\d*s?\]\s*\w*:\s*', '', stripped).strip()
            if content and content not in ("HOOK", "MAIN POINT", "DETAILS",
                                           "CONCLUSION", "CTA", "VISUAL STYLE",
                                           "PRODUCTION NOTES"):
                current_lines.append(content)
        elif current_ts and stripped.startswith("→"):
            text = re.sub(r'^→\s*', '', stripped).strip()
            if text and len(text) > 3:
                current_lines.append(text)
    if current_ts:
        sections.append((current_ts, " ".join(current_lines)))

    # Map sections into 3 shots by time ranges
    shot_ranges = [(0, 7), (7, 14), (14, 21)]  # ~7s each
    motion_verbs = [
        "slowly rising from the depths, water cascading off massive scales",
        "turning deliberately toward camera, unleashing a guttural roar",
        "stepping forward through destruction, crushing debris underfoot",
    ]
    shots = []
    for i, (s_start, s_end) in enumerate(shot_ranges):
        # Gather section content that falls within this shot's time range
        shot_parts = []
        for (ts_start, ts_end), content in sections:
            if ts_start < s_end and ts_end > s_start and content:
                shot_parts.append(content)
        scene_desc = " ".join(shot_parts) if shot_parts else prompt_seed
        # Inject explicit motion
        motion = motion_verbs[i % len(motion_verbs)]
        full_prompt = f"{scene_desc}, {motion}"
        if style_suffix:
            full_prompt += f", {style_suffix}"
        # Character lock
        if character and character.get("lock"):
            full_prompt = f"{character['lock']}, {full_prompt}"

        shots.append({
            "shot_index": i + 1,
            "time_range": f"{s_start}-{s_end}s",
            "prompt": full_prompt,
            "model_type": "minimax_h3_fl2va_pruned",
            "video_prompt_type": "I",
            "audio_prompt_type": None,
            "audio_guide": None,
            "image_refs": [],
            "width": 480,
            "height": 832,
            "video_length": 176,
            "force_fps": "24",
            "num_inference_steps": 20,
            "guidance_scale": 1.0,
            "embedded_guidance_scale": 6.0,
            "seed": 42 + i,
        })

    return {
        "shots": shots,
        "total_shots": len(shots),
        "frames_per_shot": 176,
        "fps": 24,
        "total_duration_s": round(len(shots) * 176 / 24, 2),
    }


def _clean_goal_for_scene(goal: str) -> str:
    """Turn a premise/goal string into a terse visual-scene clause for the flux3
    prompt (the scene slot). Strips any /t2v command wrappers and trailing tool
    params like duration/aspect_ratio/resolution so only the scene phrase remains.
    """
    if not goal:
        return ""
    scene = goal.strip()
    # Strip an optional "/t2v prompt:" command prefix.
    m = re.search(r"(?:/t2v\s+)?(?:prompt:)?(.*)", scene, flags=re.DOTALL)
    if m:
        scene = m.group(1)
    # Cut tool params after a marker like ' duration:' / ' aspect_ratio:'.
    for marker in (" duration:", " aspect_ratio:", " resolution:", " duration ", " -"):
        idx = scene.find(marker)
        if idx > 0:
            scene = scene[:idx].rstrip(" .,;")
            break
    scene = scene.strip(" \"'\\n .,;")
    return scene


def generate_production_prompts(hook: str, script: str, visual_direction: dict, niche: str = "", character: Optional[dict] = None, goal: str = "") -> dict:
    """Generate tool-ready copy-paste prompts for video production tools.

    Returns prompts formatted for:
    - FLUX3: Single dense line with /t2v wrapper
    - Kling: API call JSON with prompt, aspect_ratio, duration
    - H3/WanGP: Job JSON with prompt, image_start, resolution, sample_solver
    - Voiceover: Clean text without timestamps
    - Text overlays: Timed text for CapCut/manual

    If a franchise `character` lock is provided (from data/character_locks.json),
    its physical lock is baked into the scene slot verbatim, its signature hooks
    ride along, and its voice texture overrides the generic archetype voice —
    same person, voice, and look on every brief of the franchise.
    """
    import re
    
    prompt_seed = visual_direction.get("prompt_seed", "")
    style_name = visual_direction.get("style_name", "")
    
    # ─── Extract hook subject for narrative-driven video generation ───
    # Remove format prefixes like "POV:", "When", etc. to get the core subject
    hook_core = re.sub(r'^(POV:|When|Tell me|Y\'all tell me|Real talk:|Hot take:)', '', hook, flags=re.IGNORECASE).strip()
    # Remove trailing punctuation
    hook_core = re.sub(r'[.!?]+$', '', hook_core).strip()
    
    # ─── Build video prompt: narrative subject FIRST, aesthetic treatment SECOND ───
    # The video must show what the hook describes, not blend unrelated content
    
    # Extract narrative subject from hook (graduation, funeral, wedding, etc.)
    hook_lower = hook_core.lower()
    narrative_subject = None
    
    # Life events
    if any(kw in hook_lower for kw in ["graduat", "diploma", "degree", "commencement"]):
        narrative_subject = "graduation ceremony, young Black man in cap and gown, family gathered, proud mother crying with joy, emotional moment"
    elif any(kw in hook_lower for kw in ["wedding", "marry", "bride", "groom"]):
        narrative_subject = "wedding ceremony, bride and groom at altar, family gathered, emotional vows"
    elif any(kw in hook_lower for kw in ["funeral", "died", "passed away", "memorial"]):
        narrative_subject = "funeral service, family gathered in mourning, emotional moment, memorial"
    elif any(kw in hook_lower for kw in ["birthday", "turned", "getting older"]):
        narrative_subject = "birthday celebration, cake with candles, family gathered, surprise party"
    elif any(kw in hook_lower for kw in ["baby", "pregnant", "newborn", "birth"]):
        narrative_subject = "new baby, mother holding newborn, family gathered, emotional moment"
    
    # Settings (if no clear event, use setting)
    if not narrative_subject:
        if any(kw in hook_lower for kw in ["school", "classroom", "campus", "college"]):
            narrative_subject = "school setting, classroom or campus, student learning, educational moment"
        elif any(kw in hook_lower for kw in ["porch", "stoop", "steps", "front yard"]):
            narrative_subject = "porch steps, family gathered outside, community moment, golden hour"
        elif any(kw in hook_lower for kw in ["kitchen", "cooking", "stove"]):
            narrative_subject = "kitchen scene, cooking together, family meal preparation, warm domestic moment"
        elif any(kw in hook_lower for kw in ["barbershop", "barber", "haircut"]):
            narrative_subject = "barbershop interior, barber cutting hair, community conversation, mirrors and chairs"
    
    # Build the video prompt — dreamingtulpa's 5-slot formula:
    #   [framing] [scene/gag] [dialogue+voice] [FIXED style suffix] [params]
    # The aesthetic treatment is NOT assembled inline anymore: it rides in
    # build_style_suffix() so every brief of a franchise shares an identical tail.
    contextual_prompt_parts = []
    
    # ─── Dialogue detection first: it decides found-footage framing ───
    hook_lower_full = hook.lower()
    hood_signals = {"hood", "block", "mama", "cookout", "function", "neighbor", "friday night", "graduated", "only one"}
    is_dialogue_content = any(sig in hook_lower_full for sig in hood_signals) or hook_lower_full.startswith("pov")
    
    # Scene slot: narrative subject if we identified one, else the style's own seed
    if narrative_subject:
        contextual_prompt_parts.append(narrative_subject)
        # Era tag from style name (cheap, keeps period legibility)
        if "80s" in style_name or "1980s" in style_name:
            contextual_prompt_parts.append("1980s aesthetic")
        elif "90s" in style_name or "1990s" in style_name:
            contextual_prompt_parts.append("1990s aesthetic")
        elif "70s" in style_name or "1970s" in style_name:
            contextual_prompt_parts.append("1970s aesthetic")
    else:
        # Scenario: the style's prompt_seed is the aesthetic *treatment*, but the
        # actual scene must come from the requested premise/goal. Without this,
        # every brief of an auto-matched style renders the SAME visual content
        # regardless of premise (the bug where different franchises returned
        # identical flux3 prompts). Prefer premise -> narrative_subject -> seed.
        premise_scene = _clean_goal_for_scene(goal) if goal else ""
        if premise_scene:
            contextual_prompt_parts.append(premise_scene)
        else:
            contextual_prompt_parts.append(prompt_seed if prompt_seed else hook_core)
    
    # Framing slot: "found footage of" on dialogue/hood content (his #1 opener,
    # 98/156 prompts). Reads as a real recording, not a produced scene.
    if is_dialogue_content and contextual_prompt_parts:
        contextual_prompt_parts[0] = f"{FOUND_FOOTAGE_PREFIX} {contextual_prompt_parts[0]}"
    
    # Character lock slot: pinned franchise character goes verbatim right after
    # the framing — same person on every brief of the franchise, exactly like
    # dreamingtulpa's recurring cast. Signature hooks are character-defining
    # mannerisms, so they ride along verbatim too.
    if character and character.get("lock"):
        char_parts = [character["lock"]]
        hooks = character.get("signature_hooks", []) or []
        if hooks:
            char_parts.append("; ".join(hooks))
        contextual_prompt_parts.insert(1, ", ".join(char_parts))
    
    # ─── Dialogue: bake character speech + voice texture into the video prompt ───
    # Hood/skit content gets scenario-specific dialogue; POV hooks get a
    # generic two-person exchange. Narrative-only content (no characters)
    # stays silent on-screen — the voiceover carries it.
    dialogue = []
    dialogue_prompt_clause = ""
    # Style-aware: any style with a dialogue_guide can carry on-screen speech in
    # its own tone. Hood/skit content additionally gets scenario-specific hood
    # dialogue; guide-based styles (kaiju/horror/music/etc.) use their examples.
    guide = visual_direction.get("dialogue_guide")
    # Fallback: infer tone from style_id/franchise keywords when no guide exists
    if not guide:
        style_id = config.get("style_id", "") or (pick or {}).get("style_id", "")
        franchise = (pick or {}).get("franchise", "")
        guide = _infer_tone_from_ids(style_id, franchise)
    if is_dialogue_content or (guide and _is_helpful_guide(guide)):
        scenario = _detect_hood_scenario(hook_lower_full)
        if scenario == "general" and hook_lower_full.startswith("pov"):
            scenario = "pov"
        guide_obj = guide if (guide and not (is_dialogue_content and guide and _is_hood_style(guide))) else None
        dialogue = generate_dialogue(scenario, hook, guide_obj, premise=goal)
        dialogue_prompt_clause = build_dialogue_prompt(scenario, hook, character, guide_obj, premise=goal)
    
    # Append dialogue clause so video tools get who is talking + how they sound
    if dialogue_prompt_clause:
        contextual_prompt_parts.append(dialogue_prompt_clause)
    
    # ─── Fixed style suffix: same verbatim tail on every gen of the franchise ───
    style_suffix = build_style_suffix(visual_direction)
    if style_suffix:
        contextual_prompt_parts.append(style_suffix)
    
    # ─── FLUX3 prompt: one dense line with all aesthetics baked in ───
    flux3_prompt = ", ".join(contextual_prompt_parts)
    flux3_command = f"/t2v prompt:{flux3_prompt} duration:{VIDEO_DURATION_S} aspect_ratio:9:16 resolution:fhd"
    
    # ─── Kling / H3: contextual prompt (not just prompt_seed) ───
    kling_prompt = ", ".join(contextual_prompt_parts) if contextual_prompt_parts else hook
    h3_prompt = ", ".join(contextual_prompt_parts) if contextual_prompt_parts else hook
    
    # ─── Section labels to filter out ───
    section_labels = {
        "HOOK", "THE TRIGGER", "THE LIST", "THE HIT", "THE MOMENT", 
        "THE VIBE", "THE CHARACTERS", "THE CHAOS", "ARRIVAL", "ESTABLISHING SHOT",
        "ESTABLISHING", "SCENE SETUP", "THE SCENE", "PAYOFF", "OUTRO",
        "VISUAL STYLE", "PRODUCTION NOTES", "MAIN POINT", "BUILD",
        "CONTEXT", "JOURNEY", "RESOLUTION", "CTA",
    }
    
    # ─── Parse script into sections with timestamps ───
    # Track: current timestamp, text content, and quoted text per section
    current_time = "0-3s"
    voiceover_parts = []  # (text, source_type) — source_type: "hook", "quoted", "section"
    text_overlays = []    # {time, text, style}
    
    for line in script.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        
        # Skip VISUAL STYLE and PRODUCTION NOTES sections entirely
        if re.match(r'^\[(?:VISUAL STYLE|PRODUCTION NOTES)\]', stripped):
            current_time = None  # Mark as skip zone
            continue
        if current_time is None:
            continue  # Still in skip zone
        
        # Detect new timestamp section: [2-7s] LABEL
        ts_match = re.match(r'\[(\d+-?\d*s?)\]\s*(?:\w[\w\s]*?)(?::|$)', stripped)
        if ts_match:
            current_time = ts_match.group(1)
            # Extract hook text from HOOK line
            hook_match = re.match(r'\[\d+-?\d*s?\]\s*HOOK:\s*(.+)', stripped)
            if hook_match:
                hook_text = hook_match.group(1).strip()
                voiceover_parts.append((hook_text, "hook"))
                text_overlays.append({
                    "time": current_time,
                    "text": hook_text,
                    "style": "bold center"
                })
            continue
        
        # Arrow lines: extract quoted text (content between outermost quotes)
        if stripped.startswith("→"):
            # Match → Text: 'content' or → Text overlay: 'content'
            text_line_match = re.match(r'→\s*(?:Text overlay|Text):\s*(.+)', stripped)
            if text_line_match:
                raw = text_line_match.group(1).strip()
                # Only extract if it's quoted (unquoted = direction line like "Large, bold, center screen")
                if (raw.startswith("'") and raw.endswith("'")) or \
                   (raw.startswith('"') and raw.endswith('"')):
                    raw = raw[1:-1]
                    if raw and len(raw) > 3:
                        boilerplate_lower = {"follow for more", "save this", "like and", 
                                              "subscribe", "link in bio", "tag someone",
                                              "comment below", "let me know", "tell me",
                                              "y'all tell me", "happened to you"}
                        if not any(bp in raw.lower() for bp in boilerplate_lower):
                            voiceover_parts.append((raw, "quoted"))
                            text_overlays.append({
                                "time": current_time,
                                "text": raw,
                                "style": "bold center" if len(raw) < 40 else "center"
                            })
                continue
            
            # Match standalone quoted arrow lines: → 'actual text' or → "actual text"
            # Handle nested quotes and apostrophes by matching outermost quotes
            # Strategy: match from first quote to last quote, then check for terminators after
            
            # Try single quotes first: → 'text with "nested" or 'apostrophes' in it'
            single_quote_match = re.match(r"→\s*'(.+)'(?:\s*(?:or|—|[-–])|$)", stripped)
            if single_quote_match:
                raw = single_quote_match.group(1).strip()
                if raw and len(raw) > 3:
                    boilerplate_lower = {"follow for more", "save this", "like and", 
                                          "subscribe", "link in bio", "tag someone",
                                          "comment below", "let me know", "tell me",
                                          "y'all tell me", "happened to you"}
                    if not any(bp in raw.lower() for bp in boilerplate_lower):
                        voiceover_parts.append((raw, "quoted"))
                        text_overlays.append({
                            "time": current_time,
                            "text": raw,
                            "style": "bold center" if len(raw) < 40 else "center"
                        })
                continue
            
            # Try double quotes: → "text with 'nested' quotes"
            double_quote_match = re.match(r'→\s*"(.+)"(?:\s*(?:or|—|[-–])|$)', stripped)
            if double_quote_match:
                raw = double_quote_match.group(1).strip()
                if raw and len(raw) > 3:
                    boilerplate_lower = {"follow for more", "save this", "like and", 
                                          "subscribe", "link in bio", "tag someone",
                                          "comment below", "let me know", "tell me",
                                          "y'all tell me", "happened to you"}
                    if not any(bp in raw.lower() for bp in boilerplate_lower):
                        voiceover_parts.append((raw, "quoted"))
                        text_overlays.append({
                            "time": current_time,
                            "text": raw,
                            "style": "bold center" if len(raw) < 40 else "center"
                        })
            continue
    
    # Build voiceover: hook first, then quoted text in order
    voiceover_text = hook  # Always start with the hook
    quoted_texts = [text for text, src in voiceover_parts if src == "quoted"]
    if quoted_texts:
        voiceover_text = hook + "\n\n" + "\n".join(quoted_texts)
    
    # If no overlays found, at least include the hook
    if not text_overlays:
        text_overlays.append({
            "time": "0-2s",
            "text": hook,
            "style": "bold center"
        })
    
    return {
        "flux3": flux3_command,
        "kling": {
            "prompt": kling_prompt,
            "aspect_ratio": "9:16",
            "duration": VIDEO_DURATION_S,
            "model": "kling-v2-master",
            "mode": "pro",
            "camera_control": {"type": "simple"}
        },
        "h3_job_json": {
            "prompt": h3_prompt,
            "image_start": "/path/to/your/first_frame.png",
            "resolution": "480x832",
            "video_length": H3_FRAME_COUNT,
            "num_inference_steps": 20,
            "sample_solver": "euler",
            "guidance_scale": 1.0,
            "embedded_guidance_scale": 6.0,
            "sliding_window_size": 124,
            "sliding_window_overlap": 1,
            # ── Ref2VA lip-sync fields (optional; additive for music-video pipeline) ──
            # Set these when generating keeper-song lip-sync shots via Ref2VA.
            # Omit for standard FL2VA multishot / T2V briefs — backward compatible.
            "audio_prompt_type": None,       # "A" to enable audio_guide conditioning
            "audio_guide": None,             # path to keeper song slice (.wav) for this shot
            "image_refs": None,              # list of identity still paths for Ref2VA <Picture N>
            "video_prompt_type": None,       # "I" when using image_refs + audio_guide together
        },
        "h3_multishot_json": _build_multishot_config(script, visual_direction, hook, character),
        "voiceover_text": voiceover_text,
        "dialogue": dialogue,
        "text_overlays": text_overlays,
        "character_lock": {
            "id": character.get("id", ""),
            "name": character.get("name", ""),
            "lock": character.get("lock", ""),
            "do_not_change": character.get("do_not_change", []),
        } if character else None,
    }


async def generate_brief(config: AgentConfig) -> Optional[ContentBrief]:
    """Run the full pipeline: insights → match style → generate → score → brief."""
    logger.info(f"Agent starting for niche={config.niche}, goal={config.goal or 'none'}")

    # Step 1: Fetch competitor insights
    logger.info("Fetching competitor insights...")
    insights = await fetch_insights(config.niche)
    if not insights:
        logger.error("No insights available")
        return None

    # Step 2: Match visual style FIRST (before hooks, so hooks can adapt)
    # Infer energy level from niche
    high_energy_niches = {"dance", "fitness", "comedy", "music"}
    energy = "high" if config.niche in high_energy_niches else "medium"
    
    visual_direction = match_visual_style(
        niche=config.niche,
        format_type="medium",
        energy_level=energy,
        goal=config.goal or "",
        style_id=config.style_id or ""
    )
    if visual_direction:
        locked = " (franchise lock)" if config.style_id else ""
        logger.info(f"Matched visual style: {visual_direction['style_name']}{locked}")

    # Step 2b: Resolve pinned franchise character (character lock), if any
    character = get_character_lock(config.character_id or "")
    if character:
        logger.info(f"Character lock engaged: {character.get('name', character.get('id'))}")

    # Step 3: Generate hooks with visual context
    logger.info("Generating hook candidates from proven patterns...")
    hooks = generate_hooks_from_insights(insights, config, visual_direction)
    
    if not hooks:
        logger.error("No hooks generated")
        return None

    logger.info(f"Generated {len(hooks)} candidates, scoring...")

    best: Optional[HookCandidate] = None
    best_score = 0
    
    # Track all candidates with their scores for random selection
    all_candidates = []
    seen_hooks = set()  # Deduplication: track exact matches
    seen_patterns = set()  # Deduplication: track structural similarity

    def normalize_hook(text: str) -> str:
        """Normalize hook for dedup comparison."""
        import re
        # Lowercase, strip punctuation, collapse whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def is_similar(new_text: str, existing_set: set, threshold: float = 0.7) -> bool:
        """Check if new hook is too similar to existing ones (Jaccard similarity)."""
        if not existing_set:
            return False
        
        new_words = set(new_text.split())
        if len(new_words) < 3:
            return False
        
        for existing in existing_set:
            existing_words = set(existing.split())
            if not existing_words:
                continue
            
            # Jaccard similarity
            intersection = new_words & existing_words
            union = new_words | existing_words
            similarity = len(intersection) / len(union) if union else 0
            
            if similarity >= threshold:
                return True
        
        return False

    for hook_data in hooks:
        if not isinstance(hook_data, dict):
            continue

        hook_text = hook_data.get("hook", "")
        if not hook_text:
            continue
        
        # Deduplication: skip exact matches
        normalized = normalize_hook(hook_text)
        if normalized in seen_hooks:
            logger.debug(f"  Skipping duplicate: {hook_text[:40]}...")
            continue
        
        # Deduplication: skip structurally similar hooks
        if is_similar(normalized, seen_patterns, threshold=0.7):
            logger.debug(f"  Skipping similar hook: {hook_text[:40]}...")
            continue
        
        seen_hooks.add(normalized)
        seen_patterns.add(normalized)

        # Score against corpus
        score_result = await score_hook(hook_text, config.niche)
        base_score = score_result.get("score", 0)
        
        # Goal relevance: penalize hooks that don't match the topic
        goal_relevance = 0.0
        if config.goal:
            goal_words = set(config.goal.lower().split())
            # Extract meaningful nouns from goal (exclude common words)
            stop_words = {"for", "about", "with", "the", "a", "an", "and", "or", "of", "to", "in", "on"}
            goal_keywords = goal_words - stop_words
            
            # Check how many goal keywords appear in hook
            hook_lower = hook_text.lower()
            keyword_matches = sum(1 for kw in goal_keywords if kw in hook_lower)
            
            # Bonus for strong relevance, penalty for zero relevance
            if keyword_matches >= 2:
                goal_relevance = 2.0  # Strong topical match
            elif keyword_matches == 1:
                goal_relevance = 0.5  # Partial match
            elif keyword_matches == 0 and len(goal_keywords) >= 3:
                goal_relevance = -3.0  # Off-topic penalty
        
        # Add format diversity bonus for non-corpus_proven variations
        format_type = hook_data.get("format_type", "")
        diversity_bonus = 0.0
        if format_type == "hood_native":
            diversity_bonus = 3.0  # Hood-native hooks are contextually superior for hood goals
        elif format_type and format_type != "corpus_proven":
            diversity_bonus = 1.5  # Boost remixes to compete with proven hooks
        
        # Penalize recently returned hooks to encourage diversity
        recent_penalty = _get_recent_penalty(config.niche, hook_text)
        
        final_score = base_score + diversity_bonus + goal_relevance - recent_penalty
        
        logger.info(f"  Hook: {hook_text[:60]}... → Score: {final_score:.1f}/10 (base={base_score}, bonus={diversity_bonus}, relevance={goal_relevance}, recent_penalty={recent_penalty:.2f}, type={format_type})")

        # Generate structured script for this hook
        script = generate_script(
            hook=hook_text,
            format_type=format_type,
            goal=config.goal,
            niche=config.niche,
            visual_style=visual_direction
        )

        candidate = HookCandidate(
            hook=hook_text,
            format_type=hook_data.get("format_type", "unknown"),
            script=script,
            caption=hook_data.get("caption", ""),
            hashtags=hook_data.get("hashtags", []),
            viral_score=base_score,  # Store the real score, not the boosted one
            predicted_er=score_result.get("predicted_er", {}),
            nearest_neighbors=score_result.get("nearest_neighbors", []),
            pattern_dna=score_result.get("pattern_dna", []),
        )
        all_candidates.append((candidate, final_score))

    if not all_candidates:
        logger.error("No hooks scored")
        return None

    # Select from top candidates with randomization
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    top_n = min(5, len(all_candidates))  # Pick from top 5 candidates
    top_candidates = all_candidates[:top_n]
    
    # Random selection from top candidates
    import random
    best, best_score = random.choice(top_candidates)
    
    logger.info(f"Selected hook: score={best.viral_score}/10, format={best.format_type} (from top {top_n} candidates)")

    # Track original hook for dedup
    original_hook = best.hook
    _record_hook_return(config.niche, original_hook)
    
    # Step 3: Synthesize why-this-works explanation
    why = _synthesize_why(best, config.niche)

    # Step 4: Build reference videos list
    ref_videos = []
    for n in best.nearest_neighbors[:5]:
        ref_videos.append({
            "creator": n.get("creator", ""),
            "views": n.get("views", 0),
            "likes": n.get("likes", 0),
            "engagement_rate": n.get("engagement_rate", 0),
            "hook": n.get("hook", ""),
            "format": n.get("format", ""),
            "url": n.get("url", ""),
        })

    # Step 5: Creative notes
    creative_notes = _creative_notes(best)

    # Step 6: Generate production prompts (FLUX3, Kling, H3, voiceover, overlays)
    production_prompts = None
    if visual_direction:
        production_prompts = generate_production_prompts(
            hook=best.hook,
            script=best.script,
            visual_direction=visual_direction,
            niche=config.niche,
            character=character,
            goal=config.goal or "",
        )

    return ContentBrief(
        hook=best.hook,
        format_type=best.format_type,
        script=best.script,
        caption=best.caption,
        hashtags=best.hashtags,
        viral_score=best.viral_score,
        predicted_er=best.predicted_er,
        confidence="medium" if len(best.nearest_neighbors) >= 5 else "low",
        sample_size=len(best.nearest_neighbors),
        why_this_works=why,
        reference_videos=ref_videos,
        pattern_dna=best.pattern_dna,
        visual_direction=visual_direction,
        production_prompts=production_prompts,
        creative_notes=creative_notes,
    )


def _synthesize_why(hook: HookCandidate, niche: str) -> str:
    """Synthesize why this hook is predicted to perform well."""
    parts = []

    parts.append(f"Scored {hook.viral_score}/10 against {len(hook.nearest_neighbors)} proven {niche} posts.")

    if hook.pattern_dna:
        top_patterns = [p["pattern"] for p in hook.pattern_dna[:2] if p.get("pattern") != "unknown"]
        if top_patterns:
            parts.append(f"Hook DNA: {', '.join(top_patterns)}.")

    if hook.nearest_neighbors:
        top = hook.nearest_neighbors[0]
        views = top.get("views", 0)
        er = top.get("engagement_rate", 0)
        creator = top.get("creator", "unknown")
        parts.append(f"Most similar post: {creator} got {views:,} views at {er:.1%} ER.")

    p50 = hook.predicted_er.get("p50", 0)
    if p50:
        parts.append(f"Predicted engagement rate: {p50:.1%} (median of similar content).")

    return " ".join(parts)


def _creative_notes(hook: HookCandidate) -> str:
    """Production tips based on format and patterns."""
    notes = []

    fmt = hook.format_type.lower()
    if "tutorial" in fmt or "how to" in fmt:
        notes.append("Start with the end result visible, then rewind to step 1.")
        notes.append("Use text overlay for each step — viewers watch without sound.")
    elif "challenge" in fmt or "duet" in fmt:
        notes.append("Use trending audio if possible — check TikTok Creative Center.")
        notes.append("First 1 second must show movement or text to stop the scroll.")
    elif "story" in fmt:
        notes.append("Open with the most dramatic moment, not the setup.")
        notes.append("Use captions — 80% of TikTok is watched on mute.")
    else:
        notes.append("Hook must land in first 1-2 seconds — no slow intros.")
        notes.append("Text overlay is mandatory for scroll-stopping.")

    notes.append("Vertical 9:16, 1080x1920. Keep under 60 seconds for algorithm boost.")
    notes.append("Post during peak hours: 7-9am, 12-1pm, 7-9pm local time.")
    
    # Add visual direction tips if available
    notes.append("Check the 'visual_direction' field for aesthetic guidance (Lost Future style).")

    return "\n".join(f"• {n}" for n in notes)


async def batch_briefs(config: AgentConfig, count: int = 3) -> list[ContentBrief]:
    """Generate multiple briefs ranked by viral score."""
    briefs = []
    for i in range(count):
        config_copy = config.model_copy()
        config_copy.custom_direction = (
            f"{config.custom_direction} (Variation {i+1} of {count})"
        ) if config.custom_direction else f"Variation {i+1} of {count}"
        brief = await generate_brief(config_copy)
        if brief:
            briefs.append(brief)

    briefs.sort(key=lambda b: b.viral_score, reverse=True)
    return briefs


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    niche = sys.argv[1] if len(sys.argv) > 1 else "dance"
    goal = sys.argv[2] if len(sys.argv) > 2 else ""

    config = AgentConfig(niche=niche, goal=goal)
    brief = asyncio.run(generate_brief(config))

    if brief:
        print(json.dumps(brief.model_dump(), indent=2))
    else:
        print("No brief generated")
        sys.exit(1)
