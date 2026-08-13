"""Viral score service — predict virality of hooks/formats against corpus."""
import sqlite3
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Corpus DB path
CORPUS_DB = Path.home() / "viral-bench-local" / "data" / "corpus.db"


def score_hook(
    hook: str,
    niche: Optional[str] = None,
    format_type: Optional[str] = None,
    k: int = 10,
) -> dict:
    """
    Score a hook against the corpus using FTS similarity + engagement stats.
    
    Returns:
        - predicted_er: dict with p25, p50, p75 engagement rates
        - confidence: 'high' (n>=30), 'medium' (n>=10), 'low' (n<10)
        - sample_size: number of similar posts found
        - nearest_neighbors: top-k most similar posts with metrics
        - pattern_dna: breakdown of which hook patterns matched
    """
    if not hook or not hook.strip():
        return {"error": "hook is required"}
    
    conn = sqlite3.connect(f"file:{CORPUS_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    
    # FTS search for similar hooks
    fts_query = " OR ".join(f'"{w}"' for w in hook.split() if len(w) > 2)
    
    # Build niche filter
    niche_filter = ""
    params = {"query": fts_query, "limit": k * 3}  # oversample for filtering
    if niche:
        from services.research.app import NICHE_CREATORS
        creators = NICHE_CREATORS.get(niche.lower())
        if creators:
            placeholders = ",".join(f"@h{i}" for i in range(len(creators)))
            niche_filter = f"AND creator_handle IN ({placeholders})"
            for i, c in enumerate(creators):
                params[f"h{i}"] = c.lstrip("@")
    
    sql = f"""
        SELECT 
            p.rowid, p.caption, p.hook, p.vlm_hook, p.vlm_format, p.format,
            p.creator_handle, p.post_url,
            p.views, p.likes, p.engagement_rate, p.published_at
        FROM posts_fts
        JOIN posts p ON p.rowid = posts_fts.rowid
        WHERE (p.hook IS NOT NULL AND p.hook != '')
          AND posts_fts MATCH :query
          {niche_filter}
        ORDER BY (bm25(posts_fts) * -1) + (p.engagement_rate * 0.5) DESC
        LIMIT :limit
    """
    
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        logger.warning(f"FTS query failed: {e}")
        return {"error": "corpus query failed", "details": str(e)}
    finally:
        conn.close()
    
    if not rows:
        return {
            "predicted_er": {"p25": 0, "p50": 0, "p75": 0},
            "confidence": "low",
            "sample_size": 0,
            "nearest_neighbors": [],
            "pattern_dna": [],
            "message": "No similar posts found in corpus"
        }
    
    # Extract engagement rates
    ers = sorted([r["engagement_rate"] for r in rows if r["engagement_rate"]])
    n = len(ers)
    
    # Compute percentiles
    def percentile(data, p):
        if not data:
            return 0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (data[c] - data[f]) * (k - f)
    
    predicted_er = {
        "p25": round(percentile(ers, 25), 4),
        "p50": round(percentile(ers, 50), 4),
        "p75": round(percentile(ers, 75), 4),
    }
    
    # Confidence tier
    if n >= 30:
        confidence = "high"
    elif n >= 10:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Nearest neighbors (top-k)
    neighbors = []
    for r in rows[:k]:
        hook_text = r["vlm_hook"] if r["vlm_hook"] else r["hook"]
        format_text = r["vlm_format"] if r["vlm_format"] else r["format"]
        neighbors.append({
            "creator": r["creator_handle"],
            "hook": hook_text,
            "format": format_text,
            "views": r["views"],
            "likes": r["likes"],
            "engagement_rate": round(r["engagement_rate"], 4),
            "url": r["post_url"] or f"https://tiktok.com/@{r['creator_handle'].lstrip('@')}/video/{r['rowid']}"
        })
    
    # Pattern DNA — which hook patterns matched
    pattern_counts = {}
    for r in rows:
        hook_text = r["vlm_hook"] if r["vlm_hook"] else r["hook"]
        if hook_text:
            # Simple pattern extraction (could be enhanced with VLM)
            patterns = extract_patterns(hook_text)
            for p in patterns:
                pattern_counts[p] = pattern_counts.get(p, 0) + 1
    
    pattern_dna = [
        {"pattern": p, "frequency": c, "pct": round(c / n * 100, 1)}
        for p, c in sorted(pattern_counts.items(), key=lambda x: -x[1])
    ]
    
    # Overall score (0-10)
    score = min(10, max(0, round(predicted_er["p50"] * 100, 1)))
    
    return {
        "score": score,
        "predicted_er": predicted_er,
        "confidence": confidence,
        "sample_size": n,
        "nearest_neighbors": neighbors,
        "pattern_dna": pattern_dna,
    }


def extract_patterns(hook: str) -> list[str]:
    """Extract pattern keywords from a hook string."""
    patterns = []
    hook_lower = hook.lower()
    
    pattern_keywords = {
        "curiosity-gap": ["why", "how", "secret", "truth", "actually", "turns out"],
        "direct-address": ["you", "your", "tell me", "show me"],
        "relatable": ["when", "every time", "always", "never"],
        "shock": ["wait", "omg", "no way", "literally"],
        "question": ["?", "what", "which", "who"],
        "imperative": ["try", "watch", "see", "check"],
    }
    
    for pattern, keywords in pattern_keywords.items():
        if any(kw in hook_lower for kw in keywords):
            patterns.append(pattern)
    
    return patterns if patterns else ["unknown"]
