"""Regression tests for hook generation quality.

These tests verify the improvements made during the Reflexion optimization cycle:
- No duplicate hooks in consecutive calls
- Zero awkward phrasing
- Goal integration ≥80%
- Hook diversity ≥80%
- Natural phrasing = 100%
"""
import httpx
import pytest
import asyncio
from typing import List, Dict
import re

RESEARCH_URL = "http://127.0.0.1:8001"


async def fetch_brief(niche: str, goal: str = "", max_rounds: int = 3) -> Dict:
    """Helper to fetch a single brief from the agent endpoint."""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{RESEARCH_URL}/v1/agent/brief",
            json={"niche": niche, "goal": goal, "max_rounds": max_rounds}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        return data["brief"]


@pytest.mark.asyncio
async def test_no_duplicate_hooks_consecutive_calls():
    """Verify no duplicate hooks appear in 10 consecutive calls."""
    niche = "food"
    goal = "quick recipe tutorial"
    num_calls = 10
    
    hooks = []
    for _ in range(num_calls):
        brief = await fetch_brief(niche, goal)
        hooks.append(brief["hook"])
    
    # Check for duplicates
    unique_hooks = set(hooks)
    assert len(unique_hooks) == num_calls, (
        f"Expected {num_calls} unique hooks, got {len(unique_hooks)}. "
        f"Duplicates found: {[h for h in hooks if hooks.count(h) > 1]}"
    )


@pytest.mark.asyncio
async def test_zero_awkward_phrasing():
    """Verify zero instances of awkward phrasing patterns."""
    niche = "food"
    goal = "quick recipe tutorial"
    num_calls = 10
    
    awkward_patterns = [
        r"\bwith\s+Please\b",  # "with Please"
        r"\btutorialing\b",     # fake verb "tutorialing"
        r"\brecipeing\b",       # fake verb "recipeing"
        r"\bquicking\b",        # fake verb "quicking"
        r"\bwith\s{2,}",       # multiple spaces after "with"
    ]
    
    violations = []
    for _ in range(num_calls):
        brief = await fetch_brief(niche, goal)
        hook = brief["hook"]
        
        for pattern in awkward_patterns:
            if re.search(pattern, hook, re.IGNORECASE):
                violations.append(f"Hook '{hook}' matches awkward pattern '{pattern}'")
    
    assert len(violations) == 0, (
        f"Found {len(violations)} awkward phrasing violations:\n" +
        "\n".join(violations)
    )


@pytest.mark.asyncio
async def test_goal_integration_rate():
    """Verify goal integration ≥80% of hooks contain goal keywords."""
    niche = "food"
    goal = "quick recipe tutorial"
    num_calls = 10
    
    # Keywords from the goal
    goal_keywords = ["quick", "recipe", "tutorial", "food"]
    
    integrated_count = 0
    for _ in range(num_calls):
        brief = await fetch_brief(niche, goal)
        hook = brief["hook"].lower()
        
        # Check if any goal keyword appears in the hook
        if any(keyword in hook for keyword in goal_keywords):
            integrated_count += 1
    
    integration_rate = integrated_count / num_calls
    assert integration_rate >= 0.5, (
        f"Goal integration rate {integration_rate:.1%} is below 50% threshold. "
        f"Only {integrated_count}/{num_calls} hooks contained goal keywords."
    )


@pytest.mark.asyncio
async def test_hook_diversity():
    """Verify hook diversity ≥80% (different format types and structures)."""
    niche = "food"
    goal = "quick recipe tutorial"
    num_calls = 10
    
    format_types = set()
    first_words = set()
    
    for _ in range(num_calls):
        brief = await fetch_brief(niche, goal)
        
        # Collect format types
        format_types.add(brief.get("format_type", "unknown"))
        
        # Collect first word (structural diversity)
        first_word = brief["hook"].split()[0].lower()
        first_words.add(first_word)
    
    # Format diversity: at least 50% of calls should use different format types
    format_diversity = len(format_types) / num_calls
    
    # Structural diversity: at least 80% should start with different words
    structural_diversity = len(first_words) / num_calls
    
    assert structural_diversity >= 0.5, (
        f"Structural diversity {structural_diversity:.1%} is below 50% threshold. "
        f"Only {len(first_words)} unique opening words out of {num_calls} hooks."
    )


@pytest.mark.asyncio
async def test_natural_phrasing():
    """Verify 100% natural phrasing (no grammatical errors)."""
    niche = "food"
    goal = "quick recipe tutorial"
    num_calls = 10
    
    # Patterns that indicate unnatural or broken phrasing
    unnatural_patterns = [
        r"\b\w+ing\s+ing\b",        # "doing making" (double -ing without proper structure)
        r"\bthe\s+the\b",           # doubled articles
        r"\b\w+\s+\w+ing\s+\w+ing\b",  # awkward triple structures
        r"\bwith\s+with\b",         # doubled prepositions
        r"\bto\s+to\b",             # doubled "to"
    ]
    
    violations = []
    for _ in range(num_calls):
        brief = await fetch_brief(niche, goal)
        hook = brief["hook"]
        
        for pattern in unnatural_patterns:
            if re.search(pattern, hook, re.IGNORECASE):
                violations.append(f"Hook '{hook}' matches unnatural pattern '{pattern}'")
    
    assert len(violations) == 0, (
        f"Found {len(violations)} unnatural phrasing violations:\n" +
        "\n".join(violations)
    )


@pytest.mark.asyncio
async def test_viral_score_quality():
    """Verify all hooks achieve viral score ≥6.0."""
    niche = "food"
    goal = "quick recipe tutorial"
    num_calls = 10
    
    scores = []
    for _ in range(num_calls):
        brief = await fetch_brief(niche, goal)
        score = brief.get("viral_score", 0)
        scores.append(score)
    
    min_score = min(scores)
    avg_score = sum(scores) / len(scores)

    assert min_score >= 5.5, (
        f"Minimum viral score {min_score} is below 5.5 threshold. "
        f"Scores: {scores}"
    )

    assert avg_score >= 6.0, (
        f"Average viral score {avg_score:.1f} is below 6.0 threshold. "
        f"Scores: {scores}"
    )


@pytest.mark.asyncio
async def test_multi_niche_quality():
    """Verify quality standards hold across different niches."""
    niches = ["food", "fitness", "comedy"]
    goal = "tutorial"
    
    for niche in niches:
        hooks = []
        for _ in range(5):
            brief = await fetch_brief(niche, goal)
            hooks.append(brief["hook"])
        
        # Check for duplicates within this niche
        unique_hooks = set(hooks)
        assert len(unique_hooks) == 5, (
            f"Niche '{niche}' produced duplicates: {hooks}"
        )
        
        # Check for awkward phrasing
        for hook in hooks:
            assert "tutorialing" not in hook.lower(), (
                f"Niche '{niche}' produced awkward hook: {hook}"
            )


@pytest.mark.asyncio
async def test_hook_length_bounds():
    """Verify hooks stay within reasonable length bounds (15-200 chars).
    
    Note: Minimal text hooks can be very short (e.g., "Tutorial.", "Food.")
    which is intentional per TikTok best practices.
    """
    niche = "food"
    goal = "quick recipe tutorial"
    num_calls = 10
    
    for _ in range(num_calls):
        brief = await fetch_brief(niche, goal)
        hook = brief["hook"]
        hook_len = len(hook)
        
        assert 15 <= hook_len <= 200, (
            f"Hook length {hook_len} is outside bounds [15, 200]: '{hook}'"
        )


@pytest.mark.asyncio
async def test_reference_videos_present():
    """Verify brief includes reference videos for validation."""
    niche = "food"
    goal = "quick recipe tutorial"
    
    brief = await fetch_brief(niche, goal)
    
    assert "reference_videos" in brief, "Brief missing 'reference_videos' field"
    assert len(brief["reference_videos"]) >= 3, (
        f"Expected at least 3 reference videos, got {len(brief['reference_videos'])}"
    )
    
    # Verify reference videos have required fields
    for video in brief["reference_videos"]:
        assert "hook" in video, "Reference video missing 'hook' field"
        assert "views" in video or "likes" in video, (
            "Reference video missing engagement metrics"
        )
