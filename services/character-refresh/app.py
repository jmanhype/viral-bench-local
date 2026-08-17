"""
Character Bible Refresh Service — automated character catalog expansion.

Uses ego-bridge to drive the Lost Future GPT and extract new character
bible entries in JSON format.

Host-only service (depends on ego-browser which is macOS-native).
Start: .venv/bin/python services/character-refresh/app.py
"""
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("character-refresh")

app = FastAPI(title="Character Refresh Service", version="0.1.0")

EGO_BRIDGE = os.environ.get("EGO_BRIDGE_URL", "http://127.0.0.1:8040")
GPT_URL = "https://chatgpt.com/g/g-6a13163085c08191ba23589bf22f0844-lost-future"
DATA_DIR = Path(os.environ.get("VBL_DATA_DIR", str(Path(__file__).resolve().parent.parent.parent / "data")))
REGISTER_PATH = DATA_DIR / "character_register.json"

REQUIRED_FIELDS = {
    "id", "name", "franchise", "morphology_lock", "cranial_lock",
    "surface_lock", "signature_visual_hooks", "prompt_template",
}


class RefreshRequest(BaseModel):
    count: int = Field(5, ge=1, le=30)
    franchise: str = Field("", description="Optional franchise filter")


class RefreshResponse(BaseModel):
    added: int
    skipped: int
    total: int
    new_ids: list[str]
    errors: list[str] = []


def _load_register() -> dict:
    if REGISTER_PATH.exists():
        return json.loads(REGISTER_PATH.read_text())
    return {"characters": []}


def _save_register(data: dict):
    REGISTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTER_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


async def _bridge_call(method: str, path: str, body: dict | None = None, timeout: float = 60) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        if method == "GET":
            r = await client.get(f"{EGO_BRIDGE}{path}")
        else:
            r = await client.post(f"{EGO_BRIDGE}{path}", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"Bridge {method} {path}: {r.status_code} {r.text[:200]}")
        return r.json()


@app.get("/health")
async def health():
    try:
        bh = await _bridge_call("GET", "/health", timeout=5)
        return {"status": "ok", "bridge": bh, "register_path": str(REGISTER_PATH)}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/v1/characters/refresh", response_model=RefreshResponse)
async def refresh_characters(req: RefreshRequest):
    register = _load_register()
    existing_ids = {c["id"] for c in register.get("characters", [])}
    errors = []

    franchise_clause = f" from the {req.franchise} franchise" if req.franchise else ""
    prompt = (
        f"Generate {req.count} new character bible entries{franchise_clause} as a JSON array. "
        f"Each must have: id (kebab-case), name, franchise, morphology_lock (body type/posture), "
        f"cranial_lock (face/hair/head shape), surface_lock (clothing/accessories/colors), "
        f"signature_visual_hooks (array of defining visual mannerisms), "
        f"prompt_template (ready-to-use image generation prompt with {{action}} placeholder). "
        f"Output ONLY valid JSON array, no markdown."
    )

    # Navigate
    logger.info("Navigating to Lost Future GPT...")
    try:
        await _bridge_call("POST", "/v1/navigate", {"url": GPT_URL, "wait": True, "timeout": 30})
    except Exception as e:
        raise HTTPException(502, f"Navigation failed: {e}")

    time.sleep(3)

    # Type prompt
    input_selectors = ["#prompt-textarea", "div[contenteditable='true']", "textarea"]
    typed = False
    for sel in input_selectors:
        try:
            await _bridge_call("POST", "/v1/type", {"selector": sel, "text": prompt}, timeout=15)
            typed = True
            break
        except Exception:
            continue

    if not typed:
        raise HTTPException(502, "Could not find chat input")

    # Submit
    time.sleep(1)
    try:
        await _bridge_call("POST", "/v1/click", {"selector": "[data-testid='send-button'], button[aria-label='Send']"}, timeout=10)
    except Exception:
        pass

    # Wait for response
    logger.info("Waiting for GPT response...")
    response_text = ""
    for attempt in range(24):
        time.sleep(5)
        try:
            js = (
                "(() => { const msgs = document.querySelectorAll('[data-message-author-role=\"assistant\"]'); "
                "if(!msgs.length) return ''; return msgs[msgs.length-1].innerText; })()"
            )
            result = await _bridge_call("POST", "/v1/evaluate", {"js": js}, timeout=15)
            text = result.get("result", "").strip()
            if text and len(text) > 50 and "[" in text:
                response_text = text
                break
        except Exception:
            continue

    if not response_text:
        raise HTTPException(504, "Timeout waiting for GPT response")

    # Parse
    cleaned = re.sub(r'```(?:json)?\s*', '', response_text).strip()
    start, end = cleaned.find('['), cleaned.rfind(']')
    if start == -1 or end <= start:
        raise HTTPException(422, "No JSON array in response")
    try:
        new_chars = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"JSON parse error: {e}")

    added = skipped = 0
    new_ids = []
    for char in new_chars:
        missing = REQUIRED_FIELDS - set(char.keys())
        if missing:
            errors.append(f"'{char.get('id','?')}': missing {missing}")
            skipped += 1
            continue
        if char["id"] in existing_ids:
            skipped += 1
            continue
        register["characters"].append(char)
        existing_ids.add(char["id"])
        new_ids.append(char["id"])
        added += 1

    _save_register(register)
    return RefreshResponse(added=added, skipped=skipped, total=len(register["characters"]), new_ids=new_ids, errors=errors)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("CHAR_REFRESH_PORT", "8051"))
    logger.info(f"Starting character-refresh on :{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
