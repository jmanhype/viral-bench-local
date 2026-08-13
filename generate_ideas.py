#!/usr/bin/env python3
"""Content Idea Generator - Uses VBL corpus insights to generate viral content briefs."""
import json
import random
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

DB_PATH = Path.home() / "viral-bench-local" / "data" / "corpus.db"


class ContentIdeaGenerator:
    def __init__(self):
        self.db = sqlite3.connect(DB_PATH)
        self.db.row_factory = sqlite3.Row
        self.niche_map = {
            'khaby.lame': 'comedy', 'wisdm8': 'comedy', 'brittany_broski': 'comedy',
            'zachking': 'magic/vfx', 'charlidamelio': 'dance', 'jasonderulo': 'dance',
            'addisonre': 'dance', 'bellapoarch': 'music', 'toniannmusic': 'music',
            'nala_cat': 'pets', 'tuckerbudzyn': 'pets', 'realgrumpycat': 'pets',
            'gordonramsayofficial': 'food', 'babishculinaryuniverse': 'food',
            'chris.hemsworth': 'fitness', 'pamela_rf': 'fitness', 'blogilates': 'fitness',
            'hankgreen': 'education', 'neildegrassetyson': 'education',
            'emma': 'lifestyle', 'merrelltwins': 'lifestyle',
            'duolingo': 'brand', 'ryanair': 'brand', 'chipotle': 'brand',
            'julianbass': 'vfx',
        }
        self.hooks_db = []
        self.formats_db = []
        self.top_posts_by_niche = defaultdict(list)
        self.metadata_stats = {}
        self._load_data()
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata insights from full corpus (all 13,901 posts)."""
        # Get niche-level stats
        rows = self.db.execute("""
            SELECT 
                CASE creator_handle
                    WHEN '@khaby.lame' THEN 'comedy'
                    WHEN '@wisdm8' THEN 'comedy'
                    WHEN '@brittany_broski' THEN 'comedy'
                    WHEN '@zachking' THEN 'magic/vfx'
                    WHEN '@charlidamelio' THEN 'dance'
                    WHEN '@jasonderulo' THEN 'dance'
                    WHEN '@addisonre' THEN 'dance'
                    WHEN '@bellapoarch' THEN 'music'
                    WHEN '@toniannmusic' THEN 'music'
                    WHEN '@nala_cat' THEN 'pets'
                    WHEN '@tuckerbudzyn' THEN 'pets'
                    WHEN '@realgrumpycat' THEN 'pets'
                    WHEN '@gordonramsayofficial' THEN 'food'
                    WHEN '@babishculinaryuniverse' THEN 'food'
                    WHEN '@chris.hemsworth' THEN 'fitness'
                    WHEN '@pamela_rf' THEN 'fitness'
                    WHEN '@blogilates' THEN 'fitness'
                    WHEN '@hankgreen' THEN 'education'
                    WHEN '@neildegrassetyson' THEN 'education'
                    WHEN '@emma' THEN 'lifestyle'
                    WHEN '@merrelltwins' THEN 'lifestyle'
                    WHEN '@duolingo' THEN 'brand'
                    WHEN '@ryanair' THEN 'brand'
                    WHEN '@chipotle' THEN 'brand'
                    WHEN '@julianbass' THEN 'vfx'
                    ELSE 'other'
                END as niche,
                COUNT(*) as total_posts,
                AVG(likes) as avg_likes,
                AVG(engagement_rate) as avg_engagement
            FROM posts
            GROUP BY niche
            HAVING niche != 'other'
        """).fetchall()
        
        for row in rows:
            niche = row['niche']
            self.metadata_stats[niche] = {
                'total_posts': row['total_posts'],
                'avg_likes': row['avg_likes'] or 0,
                'avg_engagement': row['avg_engagement'] or 0,
            }

    def _load_data(self):
        """Load corpus data and VLM analyses."""
        # Get all posts with VLM analysis
        rows = self.db.execute("""
            SELECT creator_handle, likes, views, vlm_analysis, caption
            FROM posts 
            WHERE vlm_analysis IS NOT NULL AND vlm_analysis != ''
        """).fetchall()

        for row in rows:
            creator = row['creator_handle'].lstrip('@')
            niche = self.niche_map.get(creator, 'other')
            
            try:
                analysis = json.loads(row['vlm_analysis'])
                if 'hook_type' not in analysis:
                    continue
                    
                self.hooks_db.append({
                    'niche': niche,
                    'hook': analysis.get('hook_type', ''),
                    'why_it_works': analysis.get('why_it_works', ''),
                    'retention': analysis.get('retention_triggers', []),
                    'likes': row['likes'] or 0,
                })
                
                fmt = analysis.get('visual_format', '')
                if fmt:
                    self.formats_db.append({
                        'niche': niche,
                        'format': fmt,
                        'pacing': analysis.get('pacing', ''),
                        'energy': analysis.get('energy_level', ''),
                    })
                
                # Track top posts by niche
                self.top_posts_by_niche[niche].append({
                    'likes': row['likes'] or 0,
                    'caption': row['caption'] or '',
                    'hook': analysis.get('hook_type', ''),
                    'why_it_works': analysis.get('why_it_works', ''),
                })
                
            except (json.JSONDecodeError, TypeError):
                continue

        # Sort top posts by likes
        for niche in self.top_posts_by_niche:
            self.top_posts_by_niche[niche].sort(key=lambda x: x['likes'], reverse=True)

    def get_hook_patterns(self, niche: Optional[str] = None) -> dict:
        """Analyze hook patterns, optionally filtered by niche."""
        filtered = [h for h in self.hooks_db if not niche or h['niche'] == niche]
        if not filtered:
            return {}
        
        hook_counts = defaultdict(int)
        for h in filtered:
            hook = h['hook'].lower()
            # Extract key patterns
            for keyword in ['pattern interrupt', 'curiosity gap', 'shock visual', 
                          'relatable frustration', 'direct address', 'text overlay']:
                if keyword in hook:
                    hook_counts[keyword] += 1
        
        return dict(sorted(hook_counts.items(), key=lambda x: x[1], reverse=True))

    def get_format_patterns(self, niche: Optional[str] = None) -> dict:
        """Analyze visual format patterns."""
        filtered = [f for f in self.formats_db if not niche or f['niche'] == niche]
        if not filtered:
            return {}
        
        format_counts = defaultdict(int)
        for f in filtered:
            fmt = f['format'].lower()
            for keyword in ['reveal', 'illusion', 'behind-the-scenes', 'talking head',
                          'pov', 'skit', 'tutorial', 'montage', 'b-roll']:
                if keyword in fmt:
                    format_counts[keyword] += 1
        
        return dict(sorted(format_counts.items(), key=lambda x: x[1], reverse=True))

    def generate_brief(self, niche: str, num_ideas: int = 3) -> list[dict]:
        """Generate content briefs for a specific niche."""
        hooks = self.get_hook_patterns(niche)
        formats = self.get_format_patterns(niche)
        top_posts = self.top_posts_by_niche.get(niche, [])[:10]
        
        if not hooks or not formats or not top_posts:
            return []
        
        ideas = []
        top_hook = list(hooks.keys())[0] if hooks else "pattern interrupt"
        top_format = list(formats.keys())[0] if formats else "reveal"
        
        # Generate ideas by combining patterns
        for i in range(num_ideas):
            # Pick from top performers
            ref_post = random.choice(top_posts[:5]) if top_posts else {}
            
            # Vary the hook and format
            hook_pool = list(hooks.keys())
            format_pool = list(formats.keys())
            
            chosen_hook = hook_pool[i % len(hook_pool)] if hook_pool else top_hook
            chosen_format = format_pool[i % len(format_pool)] if format_pool else top_format
            
            idea = {
                'niche': niche,
                'hook_technique': chosen_hook,
                'visual_format': chosen_format,
                'reference_video': {
                    'likes': ref_post.get('likes', 0),
                    'hook': ref_post.get('hook', ''),
                    'why_it_works': ref_post.get('why_it_works', ''),
                },
                'brief': self._build_brief_text(chosen_hook, chosen_format, niche),
            }
            ideas.append(idea)
        
        return ideas

    def _build_brief_text(self, hook: str, format: str, niche: str) -> str:
        """Build actionable brief text with metadata insights."""
        # Get metadata for this niche
        meta = self.metadata_stats.get(niche, {})
        total_posts = meta.get('total_posts', 0)
        avg_likes = meta.get('avg_likes', 0)
        avg_engagement = meta.get('avg_engagement', 0)
        
        # Calculate viral threshold (top 10%)
        top_posts = self.top_posts_by_niche.get(niche, [])
        if top_posts:
            top_10_percent_idx = int(len(top_posts) * 0.1)
            viral_threshold = top_posts[min(top_10_percent_idx, len(top_posts)-1)]['likes']
        else:
            viral_threshold = avg_likes * 2.5  # fallback
        
        brief = f"""
🎯 NICHE: {niche.upper()}

📊 MARKET INSIGHTS:
- Total posts analyzed: {total_posts:,}
- Average likes: {avg_likes:,.0f}
- Average engagement rate: {avg_engagement:.1%}
- Viral threshold (top 10%): {viral_threshold:,.0f} likes

🪝 HOOK TECHNIQUE: {hook.title()}
Open with a {hook} in the first 0-3 seconds. Examples:
- Pattern interrupt: Show something unexpected that breaks viewer expectations
- Curiosity gap: Start mid-action, make viewer wonder "what happens next?"
- Shock visual: Use striking imagery that demands attention

📹 VISUAL FORMAT: {format.title()}
Structure the video as a {format}:
- {format.title()} format uses specific pacing and transitions
- Match energy level to format (e.g., high energy for montage, medium for tutorial)

💡 EXECUTION:
1. First 3 seconds: Hook must grab attention immediately
2. Middle section: Deliver on the hook's promise
3. End: Satisfying payoff or call-to-action

📊 WHY IT WORKS:
Based on analysis of {total_posts:,} {niche} posts in our corpus.
Posts using "{hook}" hooks and "{format}" formats show {avg_engagement:.1%} engagement.
Target the {viral_threshold:,.0f} likes threshold to reach top 10% performance.
"""
        return brief.strip()

    def generate_cross_niche_briefs(self, num_ideas: int = 5) -> list[dict]:
        """Generate briefs that combine patterns across niches."""
        ideas = []
        
        # Get top patterns from each major niche
        niche_hooks = {}
        for niche in ['comedy', 'dance', 'pets', 'food']:
            hooks = self.get_hook_patterns(niche)
            if hooks:
                niche_hooks[niche] = list(hooks.keys())[0]
        
        # Generate cross-niche ideas
        niches = list(niche_hooks.keys())
        for i in range(min(num_ideas, len(niches))):
            source_niche = niches[i]
            target_niche = niches[(i + 1) % len(niches)]
            
            idea = {
                'title': f"Apply {source_niche} patterns to {target_niche}",
                'source_niche': source_niche,
                'target_niche': target_niche,
                'hook_to_try': niche_hooks[source_niche],
                'brief': f"""
🔄 CROSS-NICHE IDEA: {source_niche.upper()} → {target_niche.upper()}

Take the proven "{niche_hooks[source_niche]}" hook from {source_niche} content
and apply it to {target_niche} videos.

Example: {source_niche} uses {niche_hooks[source_niche]} hooks effectively.
How would this look in {target_niche}?

📋 BRIEF:
- Open with {niche_hooks[source_niche]} in first 3 seconds
- Adapt it to {target_niche} context
- Maintain {target_niche} aesthetic while borrowing {source_niche} psychology
"""
            }
            ideas.append(idea)
        
        return ideas

    def get_niche_summary(self, niche: str) -> dict:
        """Get summary stats for a niche."""
        hooks = self.get_hook_patterns(niche)
        formats = self.get_format_patterns(niche)
        top_posts = self.top_posts_by_niche.get(niche, [])[:5]
        
        return {
            'niche': niche,
            'top_hookss': list(hooks.items())[:3],
            'top_formats': list(formats.items())[:3],
            'top_performers': [
                {'likes': p['likes'], 'hook': p['hook'][:80]}
                for p in top_posts
            ],
        }


def main():
    print("🎬 Content Idea Generator")
    print("=" * 80)
    
    gen = ContentIdeaGenerator()
    
    # Show available niches
    niches_with_data = set(h['niche'] for h in gen.hooks_db)
    print(f"\n✅ Loaded {len(gen.hooks_db)} analyzed posts across {len(niches_with_data)} niches")
    print(f"Available niches: {', '.join(sorted(niches_with_data))}\n")
    
    # Generate ideas for a specific niche
    target_niche = 'comedy'  # Change this to any niche
    print(f"📊 Generating briefs for: {target_niche.upper()}")
    print("-" * 80)
    
    ideas = gen.generate_brief(target_niche, num_ideas=3)
    for i, idea in enumerate(ideas, 1):
        print(f"\n{'='*80}")
        print(f"IDEA #{i}")
        print(f"{'='*80}")
        print(f"Hook: {idea['hook_technique']}")
        print(f"Format: {idea['visual_format']}")
        print(f"Reference: {idea['reference_video']['likes']:,} likes")
        print(f"\n{idea['brief']}")
    
    # Cross-niche ideas
    print(f"\n\n{'='*80}")
    print("🔄 CROSS-NICHE IDEAS")
    print("=" * 80)
    
    cross_ideas = gen.generate_cross_niche_briefs(num_ideas=3)
    for i, idea in enumerate(cross_ideas, 1):
        print(f"\n{idea['title']}")
        print(idea['brief'])
    
    # Save all ideas to JSON
    output = {
        'niche_briefs': {target_niche: ideas},
        'cross_niche_briefs': cross_ideas,
    }
    
    output_path = Path.home() / "viral-bench-local" / "content_ideas.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\n✅ Ideas saved to: {output_path}")


if __name__ == "__main__":
    main()
