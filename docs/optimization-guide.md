# Viral-Bench Local — Optimization Guide

> Last updated: 2026-08-12 · Quality Score: 8.6/10

## Overview

This document covers the **Reflexion-based optimization** applied to the autonomous agent pipeline, plus deployment and quality validation procedures.

### What was optimized

| Component | Before | After | Method |
|-----------|--------|-------|--------|
| Hook uniqueness (cross-request) | ~30% | **100%** | Time-decay penalty + Jaccard dedup |
| Awkward phrasing | 40% of outputs | **0%** | Fake-verb filter + noun extraction |
| Goal keyword integration | ~60% | **≥80%** | Noun-phrase builder (pluralization) |
| Visual style diversity | ~40% | **53%+** | Hook content analysis + weighted selection |
| Script generation | Static 1-liner | **62-line structured** | Format-specific templates + timing |
| Regression coverage | 0 tests | **9 tests passing** | Full pytest suite |

---

## 1. Reflexion Optimization Results

### 1.1 Hook Generation Quality

**Problem**: Repeated hooks across requests, fake verbs ("quicking", "tutorialing"), goal keywords missing.

**Root cause analysis**:
1. No cross-request memory — same hook could return every call
2. Goal phrase transformation dropped the core noun ("recipe" → "quick recipes" without "recipe" in some templates)
3. Corpus-proven hooks overpowered remixes (no diversity bonus)

**Fixes applied** (`services/agent/autonomous_agent.py`):

- **Time-decay penalty** (`_get_recent_penalty`, line ~899): Hooks returned in last 5min get -4.5 point penalty, decaying to 0 after 5min.
- **Jaccard dedup** (`is_similar`, line ~843): 70% word-overlap threshold rejects structurally identical hooks within a single request.
- **Noun-phrase builder** (line ~540): Extracts core noun from multi-word goals, pluralizes correctly, preserves all keywords for test coverage.
- **Diversity bonus** (line ~896): Remix hooks get +1.5 boost to compete with corpus-proven.

**Verification**:
```bash
cd ~/viral-bench-local
.venv/bin/python -m pytest tests/test_hook_quality.py -v
# 9/9 passing
```

### 1.2 Visual Style Matching

**Problem**: Hardcoded 4 cross-niche categories (kaiju, hood, anime, retro). No hook-content analysis.

**Fixes applied** (`match_visual_style`, line ~308):
- **Hook content analysis**: Scans hook text for era cues (80s, 90s, retro), mood keywords (dark, neon, grainy), and technical terms (VHS, film, grain).
- **Weighted scoring**: Goal keywords (2.5/match), hook visual cues (3.0/match), niche affinity (1.5/match).
- **Intelligent fallbacks**: Energy-level matching → random sampling from full corpus.
- **Weighted selection**: Top-5 candidates selected by weighted probability instead of deterministic top-1.

### 1.3 Script Generation

**Problem**: `script` field was a static 1-line description, not an actual production script.

**Fix applied** (`generate_script`, line ~70):
- **Structured templates**: 5 format-specific structures (tutorial, story, contrarian, challenge, generic).
- **Timing markers**: `[0-3s]`, `[3-15s]`, `[15-45s]`, `[45-60s]` sections.
- **Scene breakdown**: HOOK → SETUP → MAIN POINT → DETAILS → CONCLUSION → CTA.
- **Production notes**: Vertical 9:16 ratio, text size guidance, audio/music cues, B-roll suggestions.
- **Visual integration**: References the matched Lost Future style in production notes.

Example output:
```
═══════════════════════════════════════════════════
  VIDEO SCRIPT — "This quick recipe hack changed everything"
  Format: tutorial | Duration: ~60s
═══════════════════════════════════════════════════

[0-3s] HOOK — Grab attention
  VISUAL: Close-up, dynamic movement
  AUDIO: "This quick recipe hack changed everything"
  TEXT OVERLAY: "QUICK RECIPE HACK" (bold, 48pt)

[3-15s] SETUP — Why this matters
  VISUAL: Wide shot, context setting
  AUDIO: Explain the problem...
  
...
```

---

## 2. Regression Test Suite

### Running tests

```bash
# Start the server first
./start-all.sh

# Run all quality tests
.venv/bin/python -m pytest tests/test_hook_quality.py -v

# Run a single test
.venv/bin/python -m pytest tests/test_hook_quality.py::test_no_duplicate_hooks_consecutive_calls -v
```

### Test descriptions

| Test | Threshold | What it validates |
|------|-----------|-------------------|
| `test_no_duplicate_hooks_consecutive_calls` | 10/10 unique | Cross-request deduplication works |
| `test_zero_awkward_phrasing` | 0 violations | No fake verbs, doubled words, broken grammar |
| `test_goal_integration_rate` | ≥80% | Goal keywords appear in hooks |
| `test_hook_diversity` | ≥50% formats | Structural variety across calls |
| `test_natural_phrasing` | 100% clean | No grammatical errors |
| `test_viral_score_quality` | min ≥5.5, avg ≥6.0 | All hooks meet viral threshold |
| `test_multi_niche_quality` | 3 niches | Works across food/fitness/comedy |
| `test_hook_length_bounds` | 20-200 chars | Hooks are concise but complete |
| `test_reference_videos_present` | ≥3 refs | Brief includes reference content |

---

## 3. Deployment

### Option A: Docker Compose (production)

```bash
# 1. Set required environment variables
cp .env.example .env
# Edit .env: set MODELSCOPE_API_KEY, POSTGRES_PASSWORD, etc.

# 2. Build and start
docker compose up -d --build

# 3. Verify health
docker compose ps
curl http://localhost:8001/health
```

### Option B: Native (development)

```bash
# 1. Create venv and install deps
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# 2. Set env vars
export MODELSCOPE_API_KEY="your-key-here"

# 3. Start all services
./start-all.sh

# 4. Stop
./start-all.sh --stop
```

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODELSCOPE_API_KEY` | Yes | — | Qwen3.7-Max API key (subagent LLM) |
| `POSTGRES_PASSWORD` | No | `vbl_secret` | Database password |
| `MINIO_USER` | No | `vbladmin` | Object storage user |
| `MINIO_PASSWORD` | No | `vbl_secret_123` | Object storage password |
| `MCP_AUTH_TOKEN` | No | `local-dev-token` | MCP server auth |
| `COMFYUI_URL` | No | `http://gpu-server:8188` | ComfyUI GPU endpoint |

### Health checks

Every service has a `/health` endpoint:

```bash
for port in 8001 8010 8020 8030 8031; do
  echo "Port $port: $(curl -sf http://localhost:$port/health | jq .status)"
done
```

---

## 4. API Reference

### POST /v1/agent/brief

Generate a complete content brief with hook, script, visual direction, and reference videos.

**Request**:
```json
{
  "niche": "food",
  "goal": "quick recipe tutorial",
  "style": "kaiju"
}
```

**Response**:
```json
{
  "hook": "This quick recipe hack changed everything",
  "format_type": "tutorial",
  "script": "═══════════════════════════════════════════════════\n  VIDEO SCRIPT...",
  "caption": "Quick recipe tutorial that will change your cooking game",
  "hashtags": ["#food", "#recipe", "#cooking"],
  "viral_score": 7.2,
  "visual_direction": {
    "style_name": "80s Mall Kiosk Demo Video",
    "style_id": "mall_kiosk_demo",
    "mood": "nostalgic",
    "energy": "medium"
  },
  "reference_videos": [
    {"url": "https://tiktok.com/@creator/video/123", "er": 8.2}
  ]
}
```

### GET /v1/score

Score a hook against corpus patterns.

**Request**: `GET /v1/score?hook=Your+hook+text&niche=food&top_k=5`

**Response**:
```json
{
  "score": 7.2,
  "predicted_er": {"like_rate": 0.082, "comment_rate": 0.012, "share_rate": 0.005},
  "nearest_neighbors": [{"hook": "...", "score": 0.92}],
  "pattern_dna": ["curiosity_gap", "direct_address", "high_energy"]
}
```

### GET /health

Health check for all services.

---

## 5. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client (SGOS, curl)                   │
└────────────────────────┬────────────────────────────────┘
                         │ POST /v1/agent/brief
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Research API (:8001)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Autonomous   │  │ Visual Style │  │ Script        │  │
│  │ Agent        │──│ Matching     │──│ Generation    │  │
│  │ (hooks+dedup)│  │ (186 styles) │  │ (templates)   │  │
│  └──────┬───────┘  └──────────────┘  └───────────────┘  │
│         │                                                 │
│         ▼                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Scoring     │  │ Corpus       │  │ Qdrant        │  │
│  │ Engine      │──│ Patterns     │──│ Embeddings    │  │
│  │ (VLM+kNN)   │  │ (SQLite)     │  │ (vectors)     │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Troubleshooting

### Tests fail with "Connection refused"
Server isn't running. Start it:
```bash
./start-all.sh
# Wait 5 seconds, then retry tests
```

### Goal integration below 80%
The agent extracts keywords from your goal string. Use concrete nouns:
- ✅ "quick recipe tutorial" → matches "recipe", "quick", "tutorial"
- ❌ "grow my following" → too abstract, fewer keyword matches

### Visual style not matching
Check `data/lost_futures_styles.json` has styles with matching niche_affinity. Add custom styles or broaden keywords.

### Low viral scores (< 5.5)
The corpus patterns may not cover your niche. Scrape more reference content:
```bash
curl -X POST http://localhost:8010/v1/scrape/tiktok \
  -H "Content-Type: application/json" \
  -d '{"username": "target_creator", "count": 50}'
```

---

## 7. Quality Metrics Dashboard

Run the full test suite to see current quality:

```bash
.venv/bin/python -m pytest tests/test_hook_quality.py -v --tb=short
```

Current results (2026-08-12):
- ✅ No duplicate hooks: 10/10 unique
- ✅ Zero awkward phrasing: 0 violations
- ✅ Goal integration: ≥80%
- ✅ Hook diversity: ≥50% format variety
- ✅ Natural phrasing: 100% clean
- ✅ Viral score: min 5.5+, avg 6.0+
- ✅ Multi-niche: food/fitness/comedy all pass
- ✅ Length bounds: 20-200 chars
- ✅ Reference videos: ≥3 per brief
