#!/usr/bin/env python3
"""Quick metadata analysis on 13,901 posts - engagement patterns by niche/creator."""
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

DB = Path.home() / "viral-bench-local" / "data" / "corpus.db"

# Load all posts
conn = sqlite3.connect(DB)
cursor = conn.execute("""
    SELECT 
        creator_handle,
        platform,
        likes,
        views,
        comments,
        shares,
        saves,
        engagement_rate,
        published_at
    FROM posts
    WHERE likes > 0
""")
rows = cursor.fetchall()
conn.close()

print(f"📊 Analyzing {len(rows):,} posts...\n")

# Parse into structured data
posts = []
for row in rows:
    posts.append({
        'creator': row[0],
        'platform': row[1],
        'likes': row[2],
        'views': row[3],
        'comments': row[4],
        'shares': row[5],
        'saves': row[6],
        'engagement_rate': row[7],
        'published_at': row[8],
    })

# Map creators to niches
NICHE_MAP = {
    # Comedy
    'khaby.lame': 'comedy', 'wisdm8': 'comedy', 'brittany_broski': 'comedy',
    'zachking': 'magic/vfx',
    
    # Dance
    'charlidamelio': 'dance', 'jasonderulo': 'dance', 'addisonre': 'dance',
    
    # Music
    'bellapoarch': 'music', 'toniannmusic': 'music',
    
    # Pets
    'nala_cat': 'pets', 'tuckerbudzyn': 'pets', 'realgrumpycat': 'pets',
    
    # Food
    'gordonramsayofficial': 'food', 'babishculinaryuniverse': 'food',
    
    # Fitness
    'chris.hemsworth': 'fitness', 'pamela_rf': 'fitness', 'blogilates': 'fitness',
    
    # Education
    'hankgreen': 'education', 'neildegrassetyson': 'education',
    
    # Lifestyle
    'emma': 'lifestyle', 'merrelltwins': 'lifestyle',
    
    # Brands
    'duolingo': 'brand', 'ryanair': 'brand', 'chipotle': 'brand',
    
    # VFX
    'julianbass': 'vfx',
}

for post in posts:
    creator_clean = post['creator'].lstrip('@')
    post['niche'] = NICHE_MAP.get(creator_clean, 'other')

# Group by niche
niche_groups = defaultdict(list)
for post in posts:
    niche_groups[post['niche']].append(post)

# 1. NICHE PERFORMANCE
print("=" * 80)
print("1. NICHE PERFORMANCE")
print("=" * 80)
niche_stats = []
for niche, niche_posts in niche_groups.items():
    likes_list = [p['likes'] for p in niche_posts]
    eng_list = [p['engagement_rate'] for p in niche_posts]
    views_list = [p['views'] for p in niche_posts]
    
    niche_stats.append({
        'niche': niche,
        'posts': len(niche_posts),
        'avg_likes': statistics.mean(likes_list),
        'median_likes': statistics.median(likes_list),
        'std_likes': statistics.stdev(likes_list) if len(likes_list) > 1 else 0,
        'avg_engagement': statistics.mean(eng_list),
        'avg_views': statistics.mean(views_list),
    })

niche_stats.sort(key=lambda x: x['avg_likes'], reverse=True)

print(f"{'niche':<12} {'posts':>6} {'avg_likes':>12} {'median':>12} {'std':>12} {'avg_eng':>10} {'avg_views':>12}")
print("-" * 80)
for ns in niche_stats:
    print(f"{ns['niche']:<12} {ns['posts']:>6,} {ns['avg_likes']:>12,.0f} {ns['median_likes']:>12,.0f} {ns['std_likes']:>12,.0f} {ns['avg_engagement']:>10.2%} {ns['avg_views']:>12,.0f}")
print()

# 2. CREATOR CONSISTENCY
print("=" * 80)
print("2. TOP CREATERS BY CONSISTENCY (low variance, high avg)")
print("=" * 80)
creator_groups = defaultdict(list)
for post in posts:
    creator_groups[post['creator']].append(post)

creator_stats = []
for creator, creator_posts in creator_groups.items():
    if len(creator_posts) < 10:
        continue
    
    likes_list = [p['likes'] for p in creator_posts]
    eng_list = [p['engagement_rate'] for p in creator_posts]
    
    avg_likes = statistics.mean(likes_list)
    std_likes = statistics.stdev(likes_list)
    cv = std_likes / avg_likes if avg_likes > 0 else 0
    consistency_score = avg_likes / (cv + 0.1)
    
    creator_stats.append({
        'creator': creator.lstrip('@'),
        'posts': len(creator_posts),
        'avg_likes': avg_likes,
        'std_likes': std_likes,
        'cv': cv,
        'consistency_score': consistency_score,
        'avg_engagement': statistics.mean(eng_list),
    })

creator_stats.sort(key=lambda x: x['consistency_score'], reverse=True)
top_consistent = creator_stats[:15]

print(f"{'creator':<20} {'posts':>6} {'avg_likes':>12} {'std':>12} {'cv':>6} {'consistency':>12}")
print("-" * 80)
for cs in top_consistent:
    print(f"{cs['creator']:<20} {cs['posts']:>6} {cs['avg_likes']:>12,.0f} {cs['std_likes']:>12,.0f} {cs['cv']:>6.2f} {cs['consistency_score']:>12,.0f}")
print()

# 3. ENGAGEMENT DISTRIBUTION BY NICHE
print("=" * 80)
print("3. ENGAGEMENT RATE PERCENTILES BY NICHE")
print("=" * 80)

def percentile(data, p):
    """Calculate percentile without numpy."""
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1

for niche in ['comedy', 'dance', 'music', 'pets', 'brand']:
    niche_data = [p['engagement_rate'] for p in niche_groups.get(niche, [])]
    if niche_data:
        p25 = percentile(niche_data, 25)
        p50 = percentile(niche_data, 50)
        p75 = percentile(niche_data, 75)
        p90 = percentile(niche_data, 90)
        p95 = percentile(niche_data, 95)
        print(f"{niche:12s} | 25%: {p25:.2%} | 50%: {p50:.2%} | 75%: {p75:.2%} | 90%: {p90:.2%} | 95%: {p95:.2%}")
print()

# 4. TOP PERFORMERS OVERALL
print("=" * 80)
print("4. TOP 20 POSTS BY LIKES")
print("=" * 80)
top_posts = sorted(posts, key=lambda x: x['likes'], reverse=True)[:20]
print(f"{'creator':<20} {'niche':<12} {'likes':>12} {'views':>12} {'engagement':>10}")
print("-" * 80)
for tp in top_posts:
    print(f"{tp['creator'].lstrip('@'):<20} {tp['niche']:<12} {tp['likes']:>12,} {tp['views']:>12,} {tp['engagement_rate']:>10.2%}")
print()

# 5. VIRAL THRESHOLDS
print("=" * 80)
print("5. VIRAL THRESHOLDS BY NICHE (top 10%)")
print("=" * 80)
for ns in niche_stats[:10]:  # top 10 niches
    niche_data = [p['likes'] for p in niche_groups[ns['niche']]]
    threshold = percentile(niche_data, 90)
    avg = ns['avg_likes']
    ratio = threshold / avg if avg > 0 else 0
    print(f"{ns['niche']:12s} | Top 10% threshold: {threshold:>12,.0f} likes | {ratio:.1f}x average")

print("\n✅ Metadata analysis complete!")
