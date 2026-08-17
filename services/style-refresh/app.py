"""
Lost Futures Style Refresh Service — automated style catalog expansion.

Uses ego-bridge (host-side ego-browser proxy) to drive the Lost Future GPT
at chatgpt.com and extract new visual styles in JSON format.

Host-only service (depends on ego-browser which is macOS-native).
Start: .venv/bin/python services/style-refresh/app.py
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("style-refresh")

app = FastAPI(title="Style Refresh Service", version="0.1.0")

EGO_BRIDGE = os.environ.get("EGO_BRIDGE_URL", "http://127.0.0.1:8040")
GPT_URL = "https://chatgpt.com/g/g-6a13163085c08191ba23589bf22f0844-lost-future"
DATA_DIR = Path(os.environ.get("VBL_DATA_DIR", str(Path(__file__).resolve().parent.parent.parent / "data")))
REGISTER_PATH = DATA_DIR / "visual_register.json"

REQUIRED_FIELDS = {
    "id", "name", "visual_description", "wrongness", "prompt_seed",
    "niche_affinity", "energy_level", "accurate_artifacts",
    "forbidden_artifacts", "color_science", "best_uses",
}


class RefreshRequest(BaseModel):
    count: int = Field(5, ge=1, le=50, description="Number of new styles to generate")
    focus: str = Field("", description="Optional focus area (e.g. 'Asian cinema', 'Soviet animation')")


class RefreshResponse(BaseModel):
    added: int
    skipped: int
    total: int
    new_ids: list[str]
    errors: list[str] = []


def _load_register() -> dict:
    if REGISTER_PATH.exists():
        return json.loads(REGISTER_PATH.read_text())
    return {"styles": [], "matching_rules": {}}


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
            raise RuntimeError(f"Bridge {method} {path} failed: {r.status_code} {r.text[:200]}")
        return r.json()


def _build_prompt(count: int, focus: str, existing_ids: set[str]) -> str:
    focus_clause = f" Focus on {focus}." if focus else " Focus on underrepresented aesthetics."
    sample_ids = ", ".join(list(existing_ids)[:10])
    return (
        f"Generate {count} new lost-future visual styles as a JSON array. "
        f"Each must have: id (kebab-case), name, visual_description, wrongness, "
        f"prompt_seed (detailed image generation prompt with film stock and era), "
        f"niche_affinity, energy_level, accurate_artifacts (array), "
        f"forbidden_artifacts (array), color_science, best_uses (array). "
        f"Do NOT reuse these existing ids: {sample_ids}. "
        f"{focus_clause} "
        f"Output ONLY valid JSON array, no markdown fences, no commentary."
    )


def _extract_json_from_response(text: str) -> list[dict]:
    """Extract JSON array from GPT response, handling markdown fences."""
    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = cleaned.strip()
    # Find the JSON array
    start = cleaned.find('[')
    end = cleaned.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON array found in response (len={len(text)})")
    json_str = cleaned[start:end + 1]
    return json.loads(json_str)


def _validate_style(style: dict) -> list[str]:
    missing = REQUIRED_FIELDS - set(style.keys())
    errors = []
    if missing:
        errors.append(f"missing fields: {missing}")
    if not isinstance(style.get("accurate_artifacts"), list):
        errors.append("accurate_artifacts must be array")
    if not isinstance(style.get("forbidden_artifacts"), list):
        errors.append("forbidden_artifacts must be array")
    if not isinstance(style.get("best_uses"), list):
        errors.append("best_uses must be array")
    return errors


@app.get("/health")
async def health():
    try:
        bridge_health = await _bridge_call("GET", "/health", timeout=5)
        return {"status": "ok", "bridge": bridge_health, "register_path": str(REGISTER_PATH)}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/v1/styles/refresh", response_model=RefreshResponse)
async def refresh_styles(req: RefreshRequest):
    """Generate new styles via Lost Future GPT and merge into register."""
    errors = []
    register = _load_register()
    existing_ids = {s["id"] for s in register.get("styles", [])}

    # Step 1: Navigate to the GPT
    logger.info(f"Navigating to Lost Future GPT...")
    try:
        nav_result = await _bridge_call("POST", "/v1/navigate", {"url": GPT_URL, "wait": True, "timeout": 30})
        logger.info(f"Navigation result: {nav_result}")
    except Exception as e:
        raise HTTPException(502, f"Failed to navigate to GPT: {e}")

    # Wait for page to fully load
    time.sleep(3)

    # Step 2: Take snapshot to find the input area
    logger.info("Taking snapshot to find input...")
    try:
        snap = await _bridge_call("POST", "/v1/snapshot", timeout=30)
        snapshot_text = snap.get("snapshot", "")
    except Exception as e:
        raise HTTPException(502, f"Failed to get snapshot: {e}")

    # Step 3: Type the prompt into the chat input
    prompt = _build_prompt(req.count, req.focus, existing_ids)
    logger.info(f"Sending prompt for {req.count} styles...")

    # ChatGPT uses a contenteditable div or textarea
    # Try common selectors
    input_selectors = [
        "#prompt-textarea",
        "div[contenteditable='true']",
        "textarea",
        "[data-testid='text-input']",
    ]

    typed = False
    for selector in input_selectors:
        try:
            await _bridge_call("POST", "/v1/type", {"selector": selector, "text": prompt}, timeout=15)
            typed = True
            logger.info(f"Typed prompt using selector: {selector}")
            break
        except Exception:
            continue

    if not typed:
        # Fallback: use evaluate to set content directly
        try:
            js_set = (
                f"(() => {{ const el = document.querySelector('#prompt-textarea') || "
                f"document.querySelector('[contenteditable=true]'); "
                f"if(el) {{ el.textContent = {json.dumps(prompt)}; "
                f"el.dispatchEvent(new Event('input',{{bubbles:true}})); return 'set'; }} "
                f"return 'not found'; }})()"
            )
            result = await _bridge_call("POST", "/v1/evaluate", {"js": js_set}, timeout=15)
            if "not found" in str(result.get("result", "")):
                raise HTTPException(502, "Could not find chat input element")
            typed = True
        except Exception as e:
            raise HTTPException(502, f"Failed to type prompt: {e}")

    # Step 4: Submit (press Enter or click send button)
    time.sleep(1)
    try:
        await _bridge_call("POST", "/v1/click", {"selector": "[data-testid='send-button'], button[aria-label='Send']"}, timeout=10)
    except Exception:
        # Fallback: press Enter
        try:
            await _bridge_call("POST", "/v1/evaluate", {"js": "document.querySelector('#prompt-textarea').dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',code:'Enter',bubbles:true}))"}, timeout=10)
        except Exception as e:
            raise HTTPException(502, f"Failed to submit prompt: {e}")

    # Step 5: Wait for response (poll for assistant message)
    logger.info("Waiting for GPT response (up to 180s)...")
    response_text = ""
    stable_count = 0
    prev_len = 0
    for attempt in range(36):  # 36 * 5s = 180s max
        time.sleep(5)
        try:
            js_extract = (
                "(() => { const msgs = document.querySelectorAll('[data-message-author-role=\"assistant\"]'); "
                "if(msgs.length === 0) return ''; "
                "const last = msgs[msgs.length-1]; "
                "return last.innerText; })()"
            )
            result = await _bridge_call("POST", "/v1/evaluate", {"js": js_extract}, timeout=15)
            text = result.get("result", "").strip()
            if text and len(text) > 100 and "[" in text:
                # Wait for streaming to finish: text length must be stable for 2 consecutive polls
                if len(text) == prev_len and len(text) > 200:
                    stable_count += 1
                    if stable_count >= 2:
                        response_text = text
                        logger.info(f"Got stable response ({len(text)} chars) after {(attempt+1)*5}s")
                        break
                else:
                    stable_count = 0
                prev_len = len(text)
                logger.info(f"Response growing: {len(text)} chars at {(attempt+1)*5}s")
        except Exception as e:
            logger.debug(f"Poll error: {e}")
            continue

    if not response_text:
        raise HTTPException(504, "Timeout waiting for GPT response")

    # Step 6: Parse and validate
    try:
        new_styles = _extract_json_from_response(response_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(422, f"Failed to parse GPT response as JSON: {e}")

    added = 0
    skipped = 0
    new_ids = []

    for style in new_styles:
        validation_errors = _validate_style(style)
        if validation_errors:
            errors.append(f"Style '{style.get('id', '?')}': {'; '.join(validation_errors)}")
            skipped += 1
            continue

        if style["id"] in existing_ids:
            skipped += 1
            continue

        # Add historical_context if missing (optional field in our schema)
        if "historical_context" not in style:
            style["historical_context"] = ""

        register["styles"].append(style)
        existing_ids.add(style["id"])
        new_ids.append(style["id"])
        added += 1

    _save_register(register)
    logger.info(f"Done: added={added}, skipped={skipped}, total={len(register['styles'])}")

    return RefreshResponse(
        added=added,
        skipped=skipped,
        total=len(register["styles"]),
        new_ids=new_ids,
        errors=errors,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("STYLE_REFRESH_PORT", "8050"))
    logger.info(f"Starting style-refresh service on :{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
