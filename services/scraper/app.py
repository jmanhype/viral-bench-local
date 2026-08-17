"""ScrapeCreators-compatible API — local replacement.

Implements the 3 endpoints Viral-Bench actually calls:
  GET /v2/tiktok/video?url=...&trim=false
  GET /v1/instagram/post?url=...&trim=true
  GET /v3/tiktok/profile/videos?handle=...&sort_by=latest

Auth: x-api-key header validated against the configured SCRAPER_API_KEY secret (fail closed).
Response shapes match ScrapeCreators' aweme_detail / xdt_shortcode_media / aweme_list.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from services.secure_env import effective_host, secrets_equal

logger = logging.getLogger(__name__)

app = FastAPI(title="Local ScrapeCreators Compatibility API")

# ─── Config ────────────────────────────────────────────────────────────────────
HOST = effective_host("SCRAPER_HOST")
CACHE_DIR = os.environ.get("SCRAPER_CACHE_DIR", "/tmp/scraper-cache")
YTDLP_TIMEOUT = int(os.environ.get("YTDLP_TIMEOUT", "120"))
INSTALOADER_TIMEOUT = int(os.environ.get("INSTALOADER_TIMEOUT", "60"))

os.makedirs(CACHE_DIR, exist_ok=True)


# ─── Cache helpers ─────────────────────────────────────────────────────────────
def _cache_key(prefix: str, url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{prefix}_{h}"


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _load_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_cache(key: str, data: dict) -> None:
    try:
        with open(_cache_path(key), "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning("cache write failed: %s", e)


# ─── VTT Parser ────────────────────────────────────────────────────────────────
def parse_vtt(vtt_content: str) -> str:
    """Strip VTT timestamps and deduplicate lines."""
    lines = []
    seen = set()
    for line in vtt_content.split('\n'):
        line = line.strip()
        if not line or line.startswith('WEBVTT') or '-->' in line or line.isdigit():
            continue
        # Strip HTML tags
        clean = re.sub(r'<[^>]+>', '', line)
        if clean and clean not in seen:
            seen.add(clean)
            lines.append(clean)
    return ' '.join(lines)


# ─── Transcript extraction via yt-dlp ─────────────────────────────────────────
async def run_ytdlp_transcript(url: str, timeout: int = YTDLP_TIMEOUT) -> dict[str, Any]:
    """Run yt-dlp to fetch auto-generated subtitles and parse them.
    
    Returns dict with keys: transcript (str|None), language (str), auto_generated (bool), reason (str|None)
    """
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_template = os.path.join(tmpdir, "subs")
        
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp",
            "--write-auto-sub", "--sub-lang", "en",
            "--skip-download",
            "--sub-format", "vtt",
            "--no-warnings", "--quiet",
            "-o", out_template,
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"transcript": None, "language": "en", "auto_generated": True, "reason": "yt-dlp timed out"}
        
        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace").strip()[:300]
            # Check if it's specifically "no subtitles"
            if "no subtitle" in err_msg.lower() or "unavailable" in err_msg.lower():
                return {"transcript": None, "language": "en", "auto_generated": True, "reason": "no_subtitles"}
            return {"transcript": None, "language": "en", "auto_generated": True, "reason": f"yt-dlp failed: {err_msg}"}
        
        # Look for the .vtt file
        vtt_path = os.path.join(tmpdir, "subs.en.vtt")
        if not os.path.exists(vtt_path):
            # Try alternate naming patterns
            for fname in os.listdir(tmpdir):
                if fname.endswith(".vtt"):
                    vtt_path = os.path.join(tmpdir, fname)
                    break
            else:
                return {"transcript": None, "language": "en", "auto_generated": True, "reason": "no_subtitles"}
        
        try:
            with open(vtt_path, "r", encoding="utf-8", errors="replace") as f:
                vtt_content = f.read()
        except Exception as e:
            return {"transcript": None, "language": "en", "auto_generated": True, "reason": f"failed to read vtt: {e}"}
        
        transcript = parse_vtt(vtt_content)
        if not transcript:
            return {"transcript": None, "language": "en", "auto_generated": True, "reason": "no_subtitles"}
        
        return {"transcript": transcript, "language": "en", "auto_generated": True, "reason": None}


# ─── Extractor: yt-dlp ────────────────────────────────────────────────────────
async def run_ytdlp_json(url: str, timeout: int = YTDLP_TIMEOUT) -> dict[str, Any]:
    """Run yt-dlp --dump-single-json and return parsed output."""
    proc = await asyncio.create_subprocess_exec(
        "yt-dlp",
        "--dump-single-json",
        "--skip-download",
        "--no-warnings",
        "--quiet",
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"yt-dlp timed out after {timeout}s")

    if proc.returncode != 0:
        msg = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"yt-dlp failed: {msg[:300]}")

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"yt-dlp returned invalid JSON: {e}")


async def run_ytdlp_flat_playlist(url: str, timeout: int = YTDLP_TIMEOUT) -> dict[str, Any]:
    """Run yt-dlp --flat-playlist -J for fast profile listing (no per-video metadata fetch)."""
    proc = await asyncio.create_subprocess_exec(
        "yt-dlp",
        "--flat-playlist",
        "-J",
        "--no-warnings",
        "--quiet",
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"yt-dlp flat-playlist timed out after {timeout}s")

    if proc.returncode != 0:
        msg = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"yt-dlp flat-playlist failed: {msg[:300]}")

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"yt-dlp flat-playlist returned invalid JSON: {e}")


# ─── Mappers: normalize extractor output → ScrapeCreators shapes ──────────────
def _compact(values: list[Any]) -> list[str]:
    return [str(v) for v in values if v]


def ytdlp_to_aweme_detail(data: dict[str, Any]) -> dict[str, Any]:
    """Map yt-dlp TikTok JSON to ScrapeCreators aweme_detail shape."""
    # Extract images for photo/carousel posts
    images = []
    if data.get("_type") == "playlist" or data.get("entries"):
        # Carousel detected as playlist
        entries = data.get("entries", [])
        for entry in entries:
            thumbs = entry.get("thumbnails", [])
            urls = [t.get("url") for t in thumbs if t.get("url")]
            if urls:
                images.append({"display_image": {"url_list": urls}})

    # Single image thumbnails
    thumbnails = data.get("thumbnails", [])
    cover_urls = [t.get("url") for t in thumbnails if t.get("url")]

    # Video URLs
    video_urls = []
    if data.get("url"):
        video_urls.append(data["url"])
    for fmt in data.get("formats", []):
        if fmt.get("url") and fmt.get("vcodec", "none") != "none":
            video_urls.append(fmt["url"])

    # Music
    music_title = data.get("track", "") or data.get("alt_title", "") or ""
    music_artist = data.get("artist", "") or data.get("uploader", "") or ""
    music_id = str(data.get("id", ""))

    return {
        "aweme_id": str(data.get("id", "")),
        "desc": data.get("title", "") or data.get("description", "") or "",
        "create_time": data.get("timestamp"),
        "author": {
            "uid": str(data.get("channel_id", "") or data.get("uploader_id", "")),
            "unique_id": data.get("uploader", "") or data.get("channel", "") or "",
            "nickname": data.get("uploader", "") or data.get("channel", "") or "",
        },
        "statistics": {
            "play_count": data.get("view_count", 0) or 0,
            "digg_count": data.get("like_count", 0) or 0,
            "comment_count": data.get("comment_count", 0) or 0,
            "share_count": data.get("repost_count", 0) or 0,
            "collect_count": 0,  # yt-dlp doesn't extract saves reliably
            "repost_count": data.get("repost_count", 0) or 0,
            "download_count": 0,
        },
        "music": {
            "id_str": music_id,
            "id": music_id,
            "title": music_title,
            "author": music_artist,
            "play_url": {"url_list": _compact([data.get("webpage_url")])},
        },
        "video": {
            "cover": {"url_list": cover_urls[:3]},
            "origin_cover": {"url_list": cover_urls[:1]},
            "dynamic_cover": {"url_list": cover_urls[:1]},
            "download_no_watermark_addr": {"url_list": video_urls[:3]},
        },
        "image_post_info": {
            "images": images if images else [],
        },
    }


def ytdlp_profile_to_aweme_list(data: dict[str, Any], handle: str) -> dict[str, Any]:
    """Map yt-dlp flat-playlist output to ScrapeCreators aweme_list shape."""
    entries = data.get("entries", [])
    aweme_list = []

    for entry in entries:
        thumbs = entry.get("thumbnails", [])
        cover_urls = [t.get("url") for t in thumbs if t.get("url")]

        aweme_list.append({
            "aweme_id": str(entry.get("id", "")),
            "desc": entry.get("title", "") or entry.get("description", "") or "",
            "create_time": entry.get("timestamp"),
            "author": {
                "uid": "",
                "unique_id": handle,
                "nickname": handle,
            },
            "statistics": {
                "play_count": entry.get("view_count", 0) or 0,
                "digg_count": entry.get("like_count", 0) or 0,
                "comment_count": entry.get("comment_count", 0) or 0,
                "share_count": entry.get("repost_count", 0) or 0,
                "collect_count": 0,
            },
            "video": {
                "cover": {"url_list": cover_urls[:3]},
            },
            "music": {
                "id_str": "",
                "title": entry.get("track", "") or "",
                "author": entry.get("artist", "") or "",
            },
        })

    return {
        "aweme_list": aweme_list,
        "has_more": False,
        "max_cursor": len(aweme_list),
    }


# ─── Instagram: Playwright + network interception (primary) ──────────────────
IG_PROFILE_DIR = os.environ.get("IG_PROFILE_DIR", "/tmp/vbl-ig-profile")
PLAYWRIGHT_TIMEOUT = int(os.environ.get("PLAYWRIGHT_TIMEOUT", "45"))


def _extract_shortcode_from_url(url: str) -> str | None:
    """Extract shortcode from Instagram URL."""
    m = re.search(r"/(?:p|reel|tv)/([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


EGO_BROWSER = os.environ.get("EGO_BROWSER_PATH", os.path.expanduser("~/.local/bin/ego-browser"))
_IG_SCRAPE_TEMPLATE = os.path.join(os.path.dirname(__file__), "scripts", "ig-scrape-template.js")


async def _run_ego_browser_ig(shortcode: str, url: str) -> dict[str, Any] | None:
    """Fetch IG post via ego-browser (inherits user's Chrome login state).

    Reads a JS template, substitutes shortcode/URL, pipes to `ego-browser nodejs`.
    No separate login needed — ego-browser shares the user's real Chrome session.
    """
    nav_url = f"https://www.instagram.com/p/{shortcode}/"

    # Read and substitute template
    try:
        with open(_IG_SCRAPE_TEMPLATE) as f:
            script = f.read()
    except FileNotFoundError:
        logger.error("IG scrape template not found at %s", _IG_SCRAPE_TEMPLATE)
        return None

    script = script.replace("__SHORTCODE__", shortcode).replace("__NAV_URL__", nav_url)

    output = ""
    try:
        proc = await asyncio.create_subprocess_exec(
            EGO_BROWSER, "nodejs",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=script.encode()),
            timeout=PLAYWRIGHT_TIMEOUT + 15,
        )

        # cliLog outputs to stderr in ego-browser, not stdout
        output = stderr.decode().strip()
        if not output:
            out_msg = stdout.decode().strip()
            logger.warning("ego-browser IG returned no output for %s (stdout: %s)", shortcode, out_msg[:300])
            return None

        # Parse the last line of output (cliLog prints there)
        last_line = output.strip().split("\n")[-1]
        data = json.loads(last_line)

        if not data.get("success"):
            logger.warning("ego-browser IG failed for %s: %s", shortcode, data.get("error"))
            return None

        return data

    except asyncio.TimeoutError:
        logger.warning("ego-browser IG timed out for %s", shortcode)
        return None
    except json.JSONDecodeError as e:
        logger.warning("ego-browser IG JSON parse error for %s: %s (output: %s)", shortcode, e, output[:200] if output else "empty")
        return None
    except Exception as e:
        logger.error("ego-browser IG scrape error for %s: %s", shortcode, e)
        return None


async def _extract_ig_dom(page, shortcode: str) -> dict[str, Any] | None:
    """Fallback: extract post metadata from the rendered DOM."""
    try:
        # Try to find the main post image/video
        display_url = await page.evaluate("""
            () => {
                // Post images
                const imgs = document.querySelectorAll('article img[src*="cdninstagram.com"]');
                if (imgs.length > 0) return imgs[0].src;
                // Video poster
                const vids = document.querySelectorAll('video[poster]');
                if (vids.length > 0) return vids[0].poster;
                return null;
            }
        """)
        if not display_url:
            return None

        is_video = await page.evaluate("""
            () => !!document.querySelector('article video')
        """)

        caption = await page.evaluate("""
            () => {
                const el = document.querySelector('article span[dir="auto"]');
                return el ? el.textContent : '';
            }
        """)

        owner = await page.evaluate("""
            () => {
                const links = document.querySelectorAll('article a[href^="/"]');
                for (const a of links) {
                    const href = a.getAttribute('href') || '';
                    if (href.match(/^\\/[a-zA-Z0-9_.]+\\/$/) && href.length > 2) {
                        return href.slice(1, -1);
                    }
                }
                return '';
            }
        """)

        return {
            "display_url": display_url,
            "is_video": is_video,
            "caption": caption or "",
            "owner_username": owner or "",
            "shortcode": shortcode,
        }
    except Exception as e:
        logger.warning("DOM extraction failed for %s: %s", shortcode, e)
        return None


def _normalize_playwright_media(media: dict[str, Any], shortcode: str) -> dict[str, Any]:
    """Normalize intercepted GraphQL/API media data to gallery-dl-like shape."""
    # Handle both web GraphQL (xdt_shortcode_media) and mobile API shapes
    display_url = (
        media.get("display_url", "")
        or media.get("thumbnail_src", "")
        or media.get("image_versions2", {}).get("candidates", [{}])[0].get("url", "")
        or ""
    )
    is_video = bool(
        media.get("is_video")
        or media.get("video_url")
        or media.get("video_versions")
    )
    video_url = media.get("video_url", "")
    if not video_url and media.get("video_versions"):
        versions = media["video_versions"]
        if isinstance(versions, list) and versions:
            video_url = versions[0].get("url", "")

    caption = ""
    cap_obj = media.get("edge_media_to_caption", {})
    if isinstance(cap_obj, dict):
        edges = cap_obj.get("edges", [])
        if edges:
            caption = edges[0].get("node", {}).get("text", "")
    if not caption:
        caption = media.get("caption", "")
        if isinstance(caption, dict):
            caption = caption.get("text", "")

    owner = (
        media.get("owner", {}).get("username", "")
        or media.get("owner_username", "")
        or media.get("user", {}).get("username", "")
        or ""
    )

    result: dict[str, Any] = {
        "shortcode": shortcode,
        "display_url": display_url,
        "thumbnail": display_url,
        "is_video": is_video,
        "video_url": video_url,
        "caption": caption,
        "owner": owner,
    }

    # Carousel / sidecar children
    sidecar_edges = []
    children_container = (
        media.get("edge_sidecar_to_children", {})
        or media.get("carousel_media", [])
    )
    if isinstance(children_container, dict):
        children_list = children_container.get("edges", [])
    elif isinstance(children_container, list):
        children_list = children_container
    else:
        children_list = []

    for child in children_list:
        node = child.get("node", child) if isinstance(child, dict) else {}
        if not isinstance(node, dict):
            continue
        child_display = node.get("display_url", "") or node.get("thumbnail_src", "")
        child_video = node.get("video_url", "")
        child_is_video = bool(node.get("is_video") or child_video)
        sidecar_edges.append({
            "display_url": child_display,
            "is_video": child_is_video,
            "url": child_video or child_display,
        })

    if sidecar_edges:
        result["sidecar"] = sidecar_edges

    return result


# ─── Instagram: gallery-dl + instaloader fallbacks ───────────────────────────
async def _run_gallery_dl(shortcode: str, url: str) -> dict[str, Any] | None:
    """Try gallery-dl with Chrome cookies. Returns parsed metadata or None on failure."""
    out_dir = os.path.join(CACHE_DIR, f"ig_{shortcode}")
    os.makedirs(out_dir, exist_ok=True)

    venv_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin")
    gallery_dl = os.path.join(venv_bin, "gallery-dl") if os.path.exists(os.path.join(venv_bin, "gallery-dl")) else "gallery-dl"

    # Use exported cookies file (avoids macOS Keychain access issues from subprocesses)
    cookies_file = "/tmp/ig-cookies.txt"
    if os.path.exists(cookies_file):
        cookie_args = ["--cookies", cookies_file]
    else:
        cookie_args = ["--cookies-from-browser", "chrome"]

    proc = await asyncio.create_subprocess_exec(
        gallery_dl,
        *cookie_args,
        "--get-urls",
        "--write-metadata",
        "-o", out_dir,
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=INSTALOADER_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning("gallery-dl timed out for %s", shortcode)
        return None

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace").strip()[:300]
        logger.warning("gallery-dl failed for %s: %s", shortcode, err_msg)
        # Still check if it wrote any metadata despite non-zero exit
        meta_path = os.path.join(out_dir, f"{shortcode}.json")
        if not os.path.exists(meta_path):
            return None

    # Try to find the metadata JSON gallery-dl wrote
    meta_path = os.path.join(out_dir, f"{shortcode}.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                raw = json.load(f)
            # Check if we actually got media URLs
            has_media = bool(raw.get("thumbnail") or raw.get("display_url") or raw.get("video_url") or raw.get("sidecar"))
            if has_media:
                return raw
            logger.warning("gallery-dl metadata for %s has no media URLs", shortcode)
        except Exception as e:
            logger.warning("gallery-dl metadata parse failed for %s: %s", shortcode, e)

    # Fallback: parse gallery-dl URL output lines
    urls = [line.strip() for line in stdout.decode().splitlines() if line.strip().startswith("http")]
    if urls:
        return {"urls": urls, "shortcode": shortcode}

    return None


async def _run_instaloader(shortcode: str) -> dict[str, Any] | None:
    """Fallback: instaloader with Chrome cookies. Returns parsed metadata or None."""
    out_dir = os.path.join(CACHE_DIR, f"ig_{shortcode}")
    os.makedirs(out_dir, exist_ok=True)

    venv_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin")
    instaloader_bin = os.path.join(venv_bin, "instaloader") if os.path.exists(os.path.join(venv_bin, "instaloader")) else "instaloader"

    # Use saved session if available (avoids macOS Keychain access issues from subprocesses)
    import glob
    sessions = glob.glob(os.path.expanduser("~/.config/instaloader/session-*"))
    if sessions:
        login_user = os.path.basename(sessions[0]).replace("session-", "")
        auth_args = ["--login", login_user]
    else:
        auth_args = ["--load-cookies", "chrome"]

    proc = await asyncio.create_subprocess_exec(
        instaloader_bin,
        *auth_args,
        "--no-compress-json",
        "--dirname-pattern", out_dir,
        "--filename-pattern", "{shortcode}",
        "--", f"-{shortcode}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=INSTALOADER_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning("instaloader timed out for %s", shortcode)
        return None

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace").strip()[:300]
        logger.warning("instaloader failed for %s: %s", shortcode, err_msg)

    # Look for the JSON output — instaloader writes {shortcode}.json (or .json.xz)
    json_path = os.path.join(out_dir, f"{shortcode}.json")
    json_xz_path = os.path.join(out_dir, f"{shortcode}.json.xz")

    import lzma

    raw = None
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                raw = json.load(f)
        except Exception as e:
            logger.warning("instaloader JSON parse failed for %s: %s", shortcode, e)
    elif os.path.exists(json_xz_path):
        try:
            with lzma.open(json_xz_path, "rt") as f:
                raw = json.load(f)
        except Exception as e:
            logger.warning("instaloader xz JSON parse failed for %s: %s", shortcode, e)

    if raw is None:
        return None

    # Map instaloader's native format to gallery-dl-like shape for _map_instagram_to_sc
    mapped: dict[str, Any] = {
        "shortcode": shortcode,
        "display_url": raw.get("display_url", "") or raw.get("url", ""),
        "thumbnail": raw.get("display_url", "") or raw.get("thumbnail_src", ""),
        "is_video": raw.get("is_video", False),
        "video_url": raw.get("video_url", ""),
        "caption": raw.get("caption", ""),
        "owner": raw.get("owner_profile", "") or raw.get("owner_username", ""),
    }

    # Handle sidecar/carousel children
    sidecar = raw.get("sidecar_nodes") or raw.get("edge_sidecar_to_children", {}).get("edges", [])
    if sidecar:
        children = []
        for edge in sidecar:
            node = edge.get("node", edge) if isinstance(edge, dict) else {}
            children.append({
                "display_url": node.get("display_url", ""),
                "is_video": node.get("is_video", False),
                "url": node.get("video_url", "") or node.get("display_url", ""),
            })
        mapped["sidecar"] = children

    return mapped


async def fetch_instagram_post(url: str) -> dict[str, Any]:
    """Fetch Instagram post metadata.

    Primary: Playwright + network interception (persistent browser profile).
    Fallback 1: gallery-dl with browser cookies.
    Fallback 2: instaloader with browser cookies.
    """
    # Extract shortcode from URL
    m = re.search(r"/(?:p|reel|tv)/([A-Za-z0-9_-]+)", url)
    if not m:
        raise ValueError(f"Could not extract shortcode from URL: {url}")
    shortcode = m.group(1)

    cache_k = _cache_key("ig", url)
    cached = _load_cache(cache_k)
    if cached:
        return cached

    # Ensure cache dir exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Primary: ego-browser (inherits Chrome login state)
    raw = await _run_ego_browser_ig(shortcode, url)

    # Fallback 1: gallery-dl with browser cookies
    if raw is None:
        logger.info("ego-browser returned nothing for %s, trying gallery-dl", shortcode)
        raw = await _run_gallery_dl(shortcode, url)

    # Fallback 2: instaloader with browser cookies
    if raw is None:
        logger.info("gallery-dl returned nothing for %s, trying instaloader", shortcode)
        raw = await _run_instaloader(shortcode)

    # If any method succeeded, map and return
    if raw is not None:
        result = _map_instagram_to_sc(raw, shortcode, url)
        _save_cache(cache_k, result)
        return result

    # All methods failed — return success:true with empty display_url + error_hint
    logger.warning("All IG scrapers failed for %s", shortcode)
    result = {
        "success": True,
        "data": {
            "xdt_shortcode_media": {
                "shortcode": shortcode,
                "display_url": "",
                "is_video": False,
                "edge_sidecar_to_children": None,
                "carousel_media": None,
            },
        },
        "xdt_shortcode_media": {
            "shortcode": shortcode,
            "display_url": "",
            "is_video": False,
            "edge_sidecar_to_children": None,
            "carousel_media": None,
        },
        "error_hint": (
            "Instagram scraping failed. First run requires manual login: "
            "set IG_HEADLESS=false env var and restart the scraper service, "
            "then visit an IG post URL to trigger the login flow. "
            "After logging in once, the persistent profile at /tmp/vbl-ig-profile "
            "retains cookies for future headless runs."
        ),
    }
    _save_cache(cache_k, result)
    return result


def _map_instagram_to_sc(raw: dict[str, Any], shortcode: str, url: str) -> dict[str, Any]:
    """Map scraper output (ego-browser or gallery-dl/instaloader) to ScrapeCreators shape."""
    # ego-browser format: has like_count, images[], caption, username at top level
    if "like_count" in raw or "images" in raw:
        images = raw.get("images", [])
        display_url = images[0] if images else ""
        is_video = raw.get("is_video", False)

        # Build sidecar for carousel
        sidecar_edges = []
        if len(images) > 1:
            for i, img_url in enumerate(images):
                sidecar_edges.append({
                    "node": {
                        "display_url": img_url,
                        "is_video": False,
                        "shortcode": f"{shortcode}_{i}",
                    }
                })

        media = {
            "shortcode": shortcode,
            "display_url": display_url,
            "is_video": is_video,
            "owner": {"username": raw.get("username", "")},
            "edge_media_to_caption": {
                "edges": [{"node": {"text": raw.get("caption", "")}}]
            } if raw.get("caption") else {"edges": []},
            "edge_media_preview_like": {"count": raw.get("like_count") or 0},
            "edge_media_to_comment": {"count": raw.get("comment_count") or 0},
            "video_view_count": raw.get("view_count"),
            "taken_at_timestamp": raw.get("timestamp", ""),
            "videos": raw.get("videos", []),
            "edge_sidecar_to_children": {
                "edges": sidecar_edges,
            } if sidecar_edges else None,
            "carousel_media": [
                {"display_url": e["node"]["display_url"], "is_video": e["node"]["is_video"]}
                for e in sidecar_edges
            ] if sidecar_edges else None,
        }

        return {
            "success": True,
            "data": {"xdt_shortcode_media": media},
            "xdt_shortcode_media": media,
        }

    # Handle gallery-dl/instaloader metadata format
    display_url = raw.get("thumbnail", "") or raw.get("display_url", "") or ""
    is_video = bool(raw.get("video_url") or raw.get("is_video"))

    # Carousel/sidecar children
    sidecar_edges = []
    children = raw.get("sidecar", []) or raw.get("children", []) or raw.get("urls", [])
    if isinstance(children, list):
        for i, child in enumerate(children):
            if isinstance(child, dict):
                child_url = child.get("url", "") or child.get("display_url", "") or ""
                child_is_video = child.get("is_video", False)
            elif isinstance(child, str):
                child_url = child
                child_is_video = False
            else:
                continue
            sidecar_edges.append({
                "node": {
                    "display_url": child_url,
                    "is_video": child_is_video,
                    "shortcode": f"{shortcode}_{i}",
                }
            })

    media = {
        "shortcode": shortcode,
        "display_url": display_url,
        "is_video": is_video,
        "edge_sidecar_to_children": {
            "edges": sidecar_edges,
        } if sidecar_edges else None,
        "carousel_media": [
            {"display_url": e["node"]["display_url"], "is_video": e["node"]["is_video"]}
            for e in sidecar_edges
        ] if sidecar_edges else None,
    }

    return {
        "success": True,
        "data": {
            "xdt_shortcode_media": media,
        },
        "xdt_shortcode_media": media,  # Also at top level for compatibility
    }


# ─── Auth middleware ───────────────────────────────────────────────────────────
def _expected_api_key() -> str:
    """Server-side configured Scraper API key (fail CLOSED when unset).

    Lazy so importing the module never crashes when the key is absent; the
    server refuses to authorize requests until SCRAPER_API_KEY is configured.
    """
    from services.secure_env import require_secret
    return require_secret("SCRAPER_API_KEY", hint="Set SCRAPER_API_KEY in .env before starting the scraper.")


def check_api_key(x_api_key: str | None = Header(default=None, alias="x-api-key")) -> None:
    """Authorize against the configured secret; reject missing or wrong keys.

    Uses a constant-time comparison (secrets_equal) so API-key timing does not
    leak the expected secret. Fail-closed: if SCRAPER_API_KEY is unset, the
    lazy _expected_api_key() resolves via require_secret and raises.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")
    if not secrets_equal(x_api_key, _expected_api_key()):
        raise HTTPException(status_code=403, detail="Invalid x-api-key")


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.get("/v2/tiktok/video/transcript")
async def tiktok_video_transcript(
    url: str = Query(...),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> JSONResponse:
    """Fetch transcript/captions for a TikTok video."""
    check_api_key(x_api_key)

    cache_k = _cache_key("tt_transcript", url)
    cached = _load_cache(cache_k)
    if cached:
        return JSONResponse(content=cached)

    try:
        result = await run_ytdlp_transcript(url)
        response = {
            "url": url,
            "transcript": result["transcript"],
            "language": result["language"],
            "auto_generated": result["auto_generated"],
        }
        if result.get("reason"):
            response["reason"] = result["reason"]
        _save_cache(cache_k, response)
        return JSONResponse(content=response)
    except Exception as e:
        logger.error("TikTok transcript fetch failed for %s: %s", url, e)
        return JSONResponse(
            status_code=200,
            content={"url": url, "transcript": None, "language": "en", "auto_generated": True, "reason": str(e)[:300]},
        )


@app.get("/v2/tiktok/video")
async def tiktok_video(
    url: str = Query(...),
    trim: bool = Query(False),
    include_transcript: bool = Query(False),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> JSONResponse:
    """Fetch TikTok post metadata + images + stats + music."""
    check_api_key(x_api_key)

    cache_k = _cache_key("tt_video", url)
    # When include_transcript is requested, use a separate cache key to avoid polluting base cache
    if include_transcript:
        cache_k_with_transcript = _cache_key("tt_video_wt", url)
        cached = _load_cache(cache_k_with_transcript)
        if cached:
            return JSONResponse(content=cached)
    else:
        cached = _load_cache(cache_k)
        if cached:
            return JSONResponse(content=cached)

    try:
        raw = await run_ytdlp_json(url)
        result = {"success": True, "aweme_detail": ytdlp_to_aweme_detail(raw)}

        if include_transcript:
            try:
                transcript_result = await run_ytdlp_transcript(url)
                result["transcript"] = transcript_result["transcript"]
                result["transcript_language"] = transcript_result["language"]
                result["transcript_auto_generated"] = transcript_result["auto_generated"]
                if transcript_result.get("reason"):
                    result["transcript_reason"] = transcript_result["reason"]
            except Exception as te:
                logger.warning("Transcript fetch failed alongside video for %s: %s", url, te)
                result["transcript"] = None
                result["transcript_reason"] = str(te)[:300]

            _save_cache(cache_k_with_transcript, result)
        else:
            _save_cache(cache_k, result)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error("TikTok video fetch failed for %s: %s", url, e)
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": str(e)[:300]},
        )


@app.get("/v1/instagram/post")
async def instagram_post(
    url: str = Query(...),
    trim: bool = Query(True),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> JSONResponse:
    """Fetch Instagram post/carousel metadata + images."""
    check_api_key(x_api_key)

    try:
        result = await fetch_instagram_post(url)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Instagram post fetch failed for %s: %s", url, e)
        return JSONResponse(
            status_code=200,
            content={"success": False, "error": str(e)[:300]},
        )


@app.get("/v3/tiktok/profile/videos")
async def tiktok_profile_videos(
    handle: str = Query(...),
    sort_by: str = Query("latest"),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> JSONResponse:
    """Fetch recent videos from a TikTok profile."""
    check_api_key(x_api_key)

    clean_handle = handle.lstrip("@")
    profile_url = f"https://www.tiktok.com/@{clean_handle}"

    cache_k = _cache_key("tt_profile", clean_handle)
    cached = _load_cache(cache_k)
    if cached:
        return JSONResponse(content=cached)

    try:
        raw = await run_ytdlp_flat_playlist(profile_url)
        result = ytdlp_profile_to_aweme_list(raw, clean_handle)
        result["success"] = True
        _save_cache(cache_k, result)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("TikTok profile fetch failed for @%s: %s", clean_handle, e)
        return JSONResponse(
            status_code=200,
            content={"success": False, "aweme_list": [], "has_more": False, "error": str(e)[:300]},
        )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "scraper-api"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SCRAPER_PORT", "8010"))
    uvicorn.run(app, host=HOST, port=port)
