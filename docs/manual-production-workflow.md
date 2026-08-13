# Manual Production Workflow Guide

This guide walks you through using VBL's production prompts to create viral hood content manually with video generation tools and CapCut.

## Overview

VBL generates **tool-ready production prompts** that you copy-paste into:
- **FLUX3 / Kling / H3** for video generation
- **CapCut** for text overlays and voiceover
- **Final editing tools** for assembly

No automation required. You control the creative process.

---

## Step 1: Generate a Brief

Request a content brief with production prompts:

```bash
curl -X POST http://localhost:8001/v1/agent/brief \
  -H "Content-Type: application/json" \
  -d '{
    "niche": "comedy",
    "goal": "maximize engagement",
    "include_production_prompts": true
  }'
```

Response includes:
- **hook**: The opening line (e.g., "When your mama say 'come inside' but you already outside with the boys")
- **format_type**: Content format (e.g., `hood_native`, `pov_relatability`)
- **production_prompts**: Tool-ready commands for FLUX3, Kling, H3
- **voiceover_text**: Full narration script
- **text_overlays**: Timed captions with styling

---

## Step 2: Generate Video Footage

### Option A: FLUX3 (WanGP)

Copy the `flux3` command from the brief:

```bash
/t2v prompt: 1990s South Central golden hour, three Black teenagers on porch steps laughing, palm shadows, stucco apartments, warm 35mm film grain, slow dolly shot duration:10 aspect_ratio:9:16
```

**In WanGP:**
1. Paste the prompt into the text field
2. Set `duration: 10` (or your preferred length)
3. Set `aspect_ratio: 9:16` (vertical for TikTok/IG)
4. Generate and download the video

### Option B: Kling

Use the `kling` JSON object:

```json
{
  "prompt": "1990s South Central golden hour, three Black teenagers on porch steps laughing, palm shadows, stucco apartments, warm 35mm film grain, slow dolly shot",
  "aspect_ratio": "9:16",
  "duration": 10,
  "model": "kling-v2-master",
  "mode": "pro",
  "camera_control": {"type": "simple"}
}
```

**Via API or UI:**
- Use Kling's API or web interface
- Paste the JSON or fill fields manually
- Download generated video

### Option C: H3 (WanGP Advanced)

Use the `h3_job_json` for precise control:

```json
{
  "prompt": "1990s South Central golden hour, three Black teenagers on porch steps laughing, palm shadows, stucco apartments, warm 35mm film grain, slow dolly shot",
  "image_start": "/path/to/your/first_frame.png",
  "resolution": "480x832",
  "video_length": 124,
  "num_inference_steps": 20,
  "sample_solver": "euler",
  "guidance_scale": 1.0,
  "embedded_guidance_scale": 6.0
}
```

**In WanGP:**
1. Generate or upload a first frame image
2. Update `image_start` path
3. Load the JSON job configuration
4. Generate video

---

## Step 3: Add Text Overlays (CapCut)

VBL provides timed text overlays:

```json
"text_overlays": [
  {
    "time": "0-3s",
    "text": "When your mama say 'come inside' but you already outside with the boys",
    "style": "bold center"
  },
  {
    "time": "5-8s",
    "text": "Tell me you grew up in the hood without telling me",
    "style": "bold center"
  }
]
```

**In CapCut:**
1. Import your generated video
2. Add text layers at the specified timestamps
3. Use **bold, centered** styling (or customize)
4. Adjust duration to match `time` ranges

**Pro tip:** Use CapCut's auto-captions as a starting point, then manually adjust to match VBL's overlays.

---

## Step 4: Add Voiceover

VBL provides the full voiceover script:

```
When your mama say 'come inside' but you already outside with the boys

Tell me you grew up in the hood without telling me
```

**In CapCut:**
1. Record voiceover reading the script
2. Sync to video timeline
3. Adjust pacing to match text overlays

**Alternative:** Use TTS tools (ElevenLabs, Coqui, edge-tts) to generate audio, then import into CapCut.

---

## Step 5: Final Assembly

1. **Trim video** to 10-15 seconds (optimal for TikTok)
2. **Add music** (trending sounds or lo-fi beats)
3. **Export** at 1080x1920 (9:16 vertical)
4. **Upload** to TikTok/Instagram with relevant hashtags

---

## Workflow Example: Hood Comedy

**Brief response:**
```json
{
  "hook": "POV: it's Friday night on the block and you already know what time it is",
  "format_type": "hood_native",
  "production_prompts": {
    "flux3": "/t2v prompt: 2000s Bay Area hyphy camcorder street-party aesthetic, Black urban car culture, Oakland parking lot at night, candy-painted cars, white tees, oversized sunglasses, turf dancing circle, handheld camcorder with on-camera light, shaky zooms, headlights streaking, crowd energy duration:12 aspect_ratio:9:16",
    "kling": {...},
    "h3_job_json": {...},
    "voiceover_text": "POV: it's Friday night on the block and you already know what time it is\n\nTell me you grew up in the hood without telling me",
    "text_overlays": [...]
  }
}
```

**Manual workflow:**
1. Generate video with FLUX3 using the prompt
2. Open in CapCut
3. Add text overlay at 0-3s: "POV: it's Friday night on the block and you already know what time it is"
4. Add text overlay at 5-8s: "Tell me you grew up in the hood without telling me"
5. Record voiceover
6. Add trending sound
7. Export and post

---

## Tips for Best Results

### Video Generation
- **Use hood-specific styles**: The visual register includes 10 hood aesthetics (South Central golden hour, Brooklyn brownstone summer, Atlanta trap house, etc.)
- **Match energy to format**: High-energy formats (hyphy, drill) need dynamic prompts; low-energy (nostalgia, documentary) need slower pacing
- **Iterate**: Generate 2-3 versions and pick the best

### Text Overlays
- **Keep it short**: Under 40 characters for "bold center", under 80 for "center"
- **Timing matters**: Show hook in first 3 seconds (critical for retention)
- **Use authentic language**: VBL's hood_native hooks use AAVE naturally

### Voiceover
- **Pace it**: Don't rush. Hood storytelling thrives on pauses and emphasis
- **Match the vibe**: Casual, conversational tone (not announcer voice)
- **Add ad-libs**: "You know what I'm sayin'", "For real", "On god"

### Final Polish
- **Add captions**: Even with text overlays, add full captions for accessibility
- **Trending audio**: Use sounds that match the visual style (90s hip-hop for golden hour, trap for drill)
- **Hashtags**: Mix niche (#hoodcomedy, #growinguphood) with broad (#fyp, #viral)

---

## Troubleshooting

**"Video looks too clean/generic"**
→ Check if the visual style is being applied. Add film grain, specific camera movements, or era-specific details to the prompt.

**"Text overlays don't match the vibe"**
→ Verify the hook is `hood_native` format (not `pov_contributor` or generic). Hood hooks use AAVE naturally.

**"Voiceover sounds robotic"**
→ Record it yourself with authentic delivery, or use a TTS model trained on Black voices (ElevenLabs has good options).

**"Engagement is low"**
→ Check if the hook lands in the first 1-2 seconds. If not, regenerate with a stronger hood_native hook.

---

## Advanced: Batch Production

Generate multiple briefs and queue them:

```bash
# Generate 5 briefs
for i in {1..5}; do
  curl -X POST http://localhost:8001/v1/agent/brief \
    -H "Content-Type: application/json" \
    -d '{"niche": "comedy", "include_production_prompts": true}' \
    > brief_$i.json
done
```

Process each brief through your video generation pipeline, then batch-upload to TikTok/IG.

---

## Summary

1. **Generate brief** with `include_production_prompts: true`
2. **Copy prompts** into FLUX3/Kling/H3 for video
3. **Add text overlays** in CapCut using VBL's timed captions
4. **Record voiceover** from VBL's script
5. **Assemble, export, post**

You control the creative process. VBL provides the intelligence.

**Zero automation. Full creative control. Maximum viral potential.**
