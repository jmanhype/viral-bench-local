<p align="center">
  <img src="assets/readme/hero.svg" alt="Viral-Bench Local — Self-hosted TikTok and Instagram viral content analysis API stack" width="100%">
</p>

## What it is

Drop-in API replacements for **Lightreel**, **ScrapeCreators**, and **Doublespeed** that run entirely on your own hardware. Scrape TikTok/Instagram posts, analyze them with Qwen3.8-Max VLM (88.7% VideoMMMU), extract structural patterns (hooks, pacing, energy, format), and feed insights directly into video production pipelines.

**Zero recurring API costs.** One `./start-all.sh` and you have a full content intelligence stack.

## Proof

| Metric | Value |
|--------|-------|
| Posts scraped | 4,165+ from top creators |
| VLM model | Qwen3.8-Max (VideoMMMU 88.7%) |
| Analysis speed | ~6/min concurrent |
| Top pattern found | Shock visual + curiosity-gap + high energy = 7.5% ER, 10.6M median views |
| Monthly API cost | $0 |

## Architecture

```
Mac Orchestrator                    3090 GPU Server
├── research-api    :8001           ├── ComfyUI       :8188
├── scraper-api     :8010           ├── WanGP H3      (multishot)
├── browser-worker  :8020           ├── Qwen3.8-Max   (VLM)
├── publisher       :8030           └── SDXL/FLUX     workflows
├── renderer        :8031
├── postgres        :5432
├── redis           :6379
└── qdrant          :6333
```

**Data pipeline:** `SCRAPE → ANALYZE → INSIGHTS → PRODUCE`

Each stage is an independent HTTP service. Swap any node for your own implementation.

## Services

| Service | Port | Replaces | Status |
|---------|------|----------|--------|
| research-api | 8001 | Lightreel `/v1/chat` | ✅ Live |
| scraper-api | 8010 | ScrapeCreators | ✅ Live |
| browser-worker | 8020 | ego-browser automation | ✅ Live |
| publisher | 8030 | Multi-platform posting | ✅ Live |
| renderer | 8031 | Doublespeed rendering | ✅ Live |
| mcp-server | 8020 | Model Context Protocol | 🔄 Native alternative — shares 8020 with browser-worker; run either as a native process, not as a simultaneous Compose service |

## Quick Start

```bash
git clone https://github.com/jmanhype/viral-bench-local.git
cd viral-bench-local

# Set up environment
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Copy the env template and fill in EVERY required value:
cp .env.example .env
# Edit .env → required values (all have no default):
#   MODELSCOPE_API_KEY  — ModelScope API key (subagent LLM)
#   POSTGRES_PASSWORD   — database password
#   MINIO_PASSWORD      — object-storage password
#   MCP_AUTH_TOKEN      — MCP server auth token
#   SCRAPER_API_KEY     — scraper API auth key
# Compose refuses to start until these are set (no insecure defaults).

# Load them into the environment, then start every service:
set -a; source .env; set +a
./start-all.sh

# Verify
curl http://localhost:8001/health
curl "http://localhost:8001/v1/insights?dimension=all&min_n=3"
```

## Key Endpoints

```bash
# Analyze a TikTok video with VLM
curl -X POST http://localhost:8001/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"post_url": "https://www.tiktok.com/@user/video/123"}'

# Get aggregated insights across corpus
curl "http://localhost:8001/v1/insights?dimension=all&min_n=3"

# Generate content brief with production prompts
curl -X POST http://localhost:8001/v1/agent/brief \
  -H "Content-Type: application/json" \
  -d '{
    "niche": "comedy",
    "goal": "maximize engagement",
    "include_production_prompts": true
  }'

# Scrape all videos from a TikTok profile
curl -H "x-api-key: *** \
  "http://localhost:8010/v3/tiktok/profile/videos?handle=charlidamelio"

# Bulk ingest + analyze
python -m services.research.bulk_ingest --min-posts 500
```

## Production Prompts

The `/v1/agent/brief` endpoint returns tool-ready production prompts when `include_production_prompts: true`:

```json
{
  "hook": "When your mama say 'come inside' but you already outside with the boys",
  "format_type": "hood_native",
  "production_prompts": {
    "flux3": "/t2v prompt: 1990s South Central golden hour, three Black teenagers on porch steps laughing, palm shadows, stucco apartments, warm 35mm film grain, slow dolly shot duration:10 aspect_ratio:9:16",
    "kling": {
      "prompt": "1990s South Central golden hour, three Black teenagers on porch steps laughing, palm shadows, stucco apartments, warm 35mm film grain, slow dolly shot",
      "aspect_ratio": "9:16",
      "duration": 10,
      "model": "kling-v2-master",
      "mode": "pro",
      "camera_control": {"type": "simple"}
    },
    "h3_job_json": {
      "prompt": "1990s South Central golden hour, three Black teenagers on porch steps laughing, palm shadows, stucco apartments, warm 35mm film grain, slow dolly shot",
      "image_start": "/path/to/your/first_frame.png",
      "resolution": "480x832",
      "video_length": 124,
      "num_inference_steps": 20,
      "sample_solver": "euler",
      "guidance_scale": 1.0,
      "embedded_guidance_scale": 6.0
    },
    "voiceover_text": "When your mama say 'come inside' but you already outside with the boys\n\nTell me you grew up in the hood without telling me",
    "text_overlays": [
      {"time": "0-3s", "text": "When your mama say 'come inside' but you already outside with the boys", "style": "bold center"},
      {"time": "5-8s", "text": "Tell me you grew up in the hood without telling me", "style": "bold center"}
    ]
  }
}
```

Copy these directly into your video generation tools (FLUX3, Kling, H3/WanGP) and CapCut/text overlay editors.

## Viral Pattern Insights (from corpus)

The analysis pipeline discovered these patterns separate top 10% from bottom 50%:

| Dimension | Top 10% | Bottom 50% |
|-----------|---------|------------|
| Energy | 63% high, 15% extreme | 43% medium, 36% high |
| Hook | Shock visual + curiosity-gap text overlay | Generic text overlay only |
| Best combo ER | 7.5% (shock + high energy) | 2.1% |
| Median views | 10.6M | 800K |
| Pacing | Rapid two-beat OR slow-burn→rapid | Uniform single take |

## Requirements

- **macOS** (orchestration host) or Linux
- **Ubuntu + RTX 3090** (GPU server, optional but recommended for H3/WanGP)
- **Python 3.11+** with [uv](https://github.com/astral-sh/uv)
- **ModelScope API key** (free tier works) — [get one here](https://modelscope.cn/my/myaccesstoken)
- **Docker** (for postgres, redis, qdrant)

## License

MIT
