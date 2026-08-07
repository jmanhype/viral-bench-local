"""Viral-Bench Local — Renderer Service

Slide rendering and image generation for TikTok carousels.
- Pillow fallback: instant text-on-background compositing
- ComfyUI integration: SDXL/FLUX image generation on 3090 GPU
- Style presets: configurable templates for consistent branding

Port: 8030 (or 8031 if publisher is on 8030)
"""
import asyncio
import json
import logging
import os
import textwrap
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [vbl-renderer] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HOST = os.environ.get("RENDERER_HOST", "0.0.0.0")
PORT = int(os.environ.get("RENDERER_PORT", "8031"))
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://192.168.1.143:8188")
RENDERS_DIR = Path("/tmp/vbl-renders")
FONTS_DIR = Path("/tmp/vbl-fonts")
RENDERS_DIR.mkdir(parents=True, exist_ok=True)
FONTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VBL Renderer", version="1.0.0")
app.mount("/renders", StaticFiles(directory=str(RENDERS_DIR)), name="renders")


# ─── Models ───────────────────────────────────────────────────────────────────

class SlideSpec(BaseModel):
    text: str = ""
    position: str = "middle"  # top | middle | bottom
    background_image_url: str | None = None
    background_color: str = "#1e1e1e"
    text_color: str = "#ffffff"
    font_size: int | None = None
    style_preset: str | None = None


class RenderRequest(BaseModel):
    slides: list[SlideSpec] = Field(min_length=1)
    width: int = 1080
    height: int = 1920
    use_comfyui: bool = False
    comfyui_prompt: str | None = None


class GenerateImageRequest(BaseModel):
    prompt: str = Field(min_length=1)
    width: int = 1080
    height: int = 1920
    negative_prompt: str = ""
    steps: int = 30
    cfg_scale: float = 7.0
    seed: int | None = None


class StylePreset(BaseModel):
    name: str
    description: str
    background_color: str
    text_color: str
    font_size_ratio: float  # relative to width
    text_position: str
    overlay_opacity: float = 0.0


# ─── Built-in style presets ──────────────────────────────────────────────────

STYLE_PRESETS: dict[str, StylePreset] = {
    "dark-bold": StylePreset(
        name="dark-bold",
        description="Dark background with bold white text, high contrast",
        background_color="#0a0a0a",
        text_color="#ffffff",
        font_size_ratio=0.055,
        text_position="middle",
    ),
    "gradient-blue": StylePreset(
        name="gradient-blue",
        description="Deep blue gradient with white text",
        background_color="#0f172a",
        text_color="#f1f5f9",
        font_size_ratio=0.05,
        text_position="bottom",
    ),
    "warm-minimal": StylePreset(
        name="warm-minimal",
        description="Warm cream background with dark text, minimal aesthetic",
        background_color="#fef3c7",
        text_color="#1c1917",
        font_size_ratio=0.045,
        text_position="middle",
    ),
    "neon-dark": StylePreset(
        name="neon-dark",
        description="Black background with neon green accent text",
        background_color="#000000",
        text_color="#39ff14",
        font_size_ratio=0.05,
        text_position="top",
    ),
    "clean-white": StylePreset(
        name="clean-white",
        description="White background with dark text, clean editorial look",
        background_color="#ffffff",
        text_color="#111111",
        font_size_ratio=0.048,
        text_position="middle",
    ),
}


# ─── Pillow rendering ─────────────────────────────────────────────────────────

def _load_font(size: int):
    """Try system fonts in order of preference."""
    from PIL import ImageFont
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSRounded.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b)


def render_slide_pillow(slide: SlideSpec, width: int = 1080, height: int = 1920) -> Path:
    """Render a single slide using Pillow."""
    from PIL import Image, ImageDraw

    preset = STYLE_PRESETS.get(slide.style_preset or "") if slide.style_preset else None

    bg_color = _hex_to_rgb(preset.background_color if preset else slide.background_color)
    text_color = preset.text_color if preset else slide.text_color
    position = preset.text_position if preset else slide.position
    font_size = slide.font_size or int(width * (preset.font_size_ratio if preset else 0.05))

    # Background
    img = Image.new("RGB", (width, height), bg_color)

    # Optional background image
    if slide.background_image_url:
        bg_path = Path(slide.background_image_url)
        if bg_path.exists():
            try:
                bg_img = Image.open(bg_path).convert("RGB").resize((width, height))
                # Blend with solid color for text readability
                from PIL import ImageEnhance
                bg_img = ImageEnhance.Brightness(bg_img).enhance(0.4)
                img = Image.blend(bg_img, Image.new("RGB", (width, height), bg_color), 0.3)
            except Exception as e:
                logger.warning("Failed to load background %s: %s", bg_path, e)

    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    # Wrap text
    max_chars = max(20, width // (font_size // 2))
    wrapped = textwrap.fill(slide.text[:500], width=max_chars)

    bbox = draw.textbbox((0, 0), wrapped, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2

    if position == "top":
        y = int(height * 0.1)
    elif position == "bottom":
        y = int(height * 0.85) - th
    else:
        y = (height - th) // 2

    # Shadow
    shadow_color = "#000000" if text_color != "#000000" else "#333333"
    draw.text((x + 3, y + 3), wrapped, fill=shadow_color, font=font, align="center")
    draw.text((x, y), wrapped, fill=text_color, font=font, align="center")

    fname = f"slide_{uuid.uuid4().hex[:12]}.png"
    out = RENDERS_DIR / fname
    img.save(out, "PNG", optimize=True)
    return out


# ─── ComfyUI integration ─────────────────────────────────────────────────────

async def generate_image_comfyui(
    prompt: str, width: int, height: int,
    negative_prompt: str = "", steps: int = 30,
    cfg_scale: float = 7.0, seed: int | None = None,
) -> Path | None:
    """Generate an image via ComfyUI API on the 3090 server."""
    import random

    # Ensure dimensions are multiples of 8
    width = max(64, (width // 8) * 8)
    height = max(64, (height // 8) * 8)

    workflow = {
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["3", 1]}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt or "blurry, low quality, distorted", "clip": ["3", 1]}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1}
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["5", 0],
                "seed": seed or random.randint(0, 2**32),
                "steps": steps, "cfg": cfg_scale,
                "sampler_name": "dpmpp_2m_sde", "scheduler": "karras", "denoise": 1.0,
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 2]}
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": "vbl"}
        },
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Queue the prompt
            resp = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
            if resp.status_code != 200:
                logger.error("ComfyUI queue failed: %s %s", resp.status_code, resp.text[:200])
                return None

            prompt_id = resp.json().get("prompt_id")
            if not prompt_id:
                logger.error("ComfyUI returned no prompt_id")
                return None

            # Poll for completion
            for _ in range(120):  # 4 min max
                await asyncio.sleep(2)
                hist = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
                if hist.status_code == 200 and prompt_id in hist.json():
                    outputs = hist.json()[prompt_id].get("outputs", {})
                    images = outputs.get("10", {}).get("images", [])
                    if images:
                        img_info = images[0]
                        fname = img_info["filename"]
                        subfolder = img_info.get("subfolder", "")
                        # Download the image
                        img_url = f"{COMFYUI_URL}/view?filename={fname}&subfolder={subfolder}&type=output"
                        img_resp = await client.get(img_url)
                        if img_resp.status_code == 200:
                            out = RENDERS_DIR / f"gen_{uuid.uuid4().hex[:12]}.png"
                            out.write_bytes(img_resp.content)
                            return out
            logger.error("ComfyUI generation timed out")
            return None

    except Exception as e:
        logger.error("ComfyUI error: %s", e)
        return None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    comfyui_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{COMFYUI_URL}/system_stats")
            comfyui_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "service": "renderer",
        "comfyui_available": comfyui_ok,
        "comfyui_url": COMFYUI_URL,
        "presets": list(STYLE_PRESETS.keys()),
        "renders_dir": str(RENDERS_DIR),
    }


@app.get("/presets")
async def list_presets():
    return {p.name: {"description": p.description, "text_position": p.text_position} for p in STYLE_PRESETS.values()}


@app.post("/render")
async def render_slides(req: RenderRequest):
    """Render one or more slides. Returns URLs to rendered images."""
    results = []
    for i, slide in enumerate(req.slides):
        try:
            path = render_slide_pillow(slide, req.width, req.height)
            url = f"http://{HOST}:{PORT}/renders/{path.name}"
            results.append({"index": i, "url": url, "file": str(path)})
        except Exception as e:
            logger.error("Slide %d render failed: %s", i, e)
            results.append({"index": i, "error": str(e)})

    return {"slides": results, "count": len(results)}


@app.post("/generate-image")
async def generate_image(req: GenerateImageRequest):
    """Generate an image via ComfyUI (falls back to Pillow placeholder)."""
    # Try ComfyUI first
    path = await generate_image_comfyui(
        req.prompt, req.width, req.height,
        req.negative_prompt, req.steps, req.cfg_scale, req.seed,
    )

    if path:
        return {
            "url": f"http://{HOST}:{PORT}/renders/{path.name}",
            "file": str(path),
            "backend": "comfyui",
        }

    # Fallback: Pillow placeholder with prompt as text
    logger.info("ComfyUI unavailable, using Pillow fallback")
    slide = SlideSpec(text=req.prompt[:200], position="middle", background_color="#1a1a2e", text_color="#e0e0e0")
    path = render_slide_pillow(slide, req.width, req.height)
    return {
        "url": f"http://{HOST}:{PORT}/renders/{path.name}",
        "file": str(path),
        "backend": "pillow-fallback",
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting renderer on %s:%d", HOST, PORT)
    logger.info("ComfyUI URL: %s", COMFYUI_URL)
    uvicorn.run(app, host=HOST, port=PORT)
