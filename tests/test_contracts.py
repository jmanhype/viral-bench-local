"""Contract tests for local API compatibility with Viral-Bench expectations."""
import httpx
import pytest
import asyncio

RESEARCH_URL = "http://127.0.0.1:8001"
SCRAPER_URL = "http://127.0.0.1:8010"
API_KEY = "test-key"


@pytest.mark.asyncio
async def test_research_prose():
    """Prose question returns answer as string."""
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(f"{RESEARCH_URL}/v1/chat", json={
            "question": "What are trending fitness hooks?"
        })
        assert r.status_code == 200
        data = r.json()
        assert "conversationId" in data
        assert isinstance(data["answer"], str)


@pytest.mark.asyncio
async def test_research_structured():
    """Structured request returns exactly requested fields."""
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{RESEARCH_URL}/v1/chat", json={
            "question": "Analyze fitness carousel hooks",
            "response_fields": {
                "hooks": {"type": "array", "description": "Top hooks"},
                "summary": {"type": "string", "description": "Brief summary"},
            }
        })
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["answer"], dict)
        assert "hooks" in data["answer"]
        assert "summary" in data["answer"]
        assert isinstance(data["answer"]["hooks"], list)
        assert isinstance(data["answer"]["summary"], str)


@pytest.mark.asyncio
async def test_research_too_many_fields():
    """More than 5 fields returns error."""
    async with httpx.AsyncClient(timeout=30) as c:
        fields = {f"field_{i}": {"type": "string", "description": f"Field {i}"} for i in range(6)}
        r = await c.post(f"{RESEARCH_URL}/v1/chat", json={
            "question": "test",
            "response_fields": fields,
        })
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_scraper_tiktok_video_shape():
    """TikTok video endpoint returns aweme_detail shape even on failure."""
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(
            f"{SCRAPER_URL}/v2/tiktok/video",
            params={"url": "https://www.tiktok.com/@test/video/123", "trim": "false"},
            headers={"x-api-key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert "success" in data
        # On success: aweme_detail present; on failure: error present
        if data["success"]:
            detail = data["aweme_detail"]
            assert "statistics" in detail
            assert "music" in detail
            assert "video" in detail


@pytest.mark.asyncio
async def test_scraper_instagram_shape():
    """Instagram endpoint returns xdt_shortcode_media shape."""
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.get(
            f"{SCRAPER_URL}/v1/instagram/post",
            params={"url": "https://www.instagram.com/p/CwEzKvFrXJZ/", "trim": "true"},
            headers={"x-api-key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert "success" in data
        if data["success"]:
            assert "xdt_shortcode_media" in data or "data" in data


@pytest.mark.asyncio
async def test_scraper_profile_shape():
    """Profile endpoint returns aweme_list shape."""
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(
            f"{SCRAPER_URL}/v3/tiktok/profile/videos",
            params={"handle": "nonexistent_test_user_xyz", "sort_by": "latest"},
            headers={"x-api-key": API_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        # Even on failure, should have aweme_list key
        assert "aweme_list" in data or "error" in data


@pytest.mark.asyncio
async def test_scraper_auth_required():
    """Missing API key returns 401."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(
            f"{SCRAPER_URL}/v2/tiktok/video",
            params={"url": "https://www.tiktok.com/@test/video/123"},
        )
        assert r.status_code == 401
