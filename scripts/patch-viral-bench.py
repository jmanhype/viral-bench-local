#!/usr/bin/env python3
"""Patch Viral-Bench marketing-agent.ts to use local API endpoints.

Usage: python scripts/patch-viral-bench.py /path/to/marketing-agent.ts

Adds env var overrides for all hard-coded vendor URLs:
  SCRAPECREATORS_API_URL → ScrapeCreators endpoints
  DOUBLESPEED_API_URL    → Doublespeed MCP/OAuth endpoints
  LIGHTREEL_API_URL      → Lightreel chat endpoint (already supported)
"""
import sys
import re

def patch_file(path: str) -> None:
    with open(path, 'r') as f:
        content = f.read()

    original = content

    # 1. Add SC_BASE constant before scGet function
    content = content.replace(
        "async function scGet(pathname: string, urlParam: string, trim: boolean): Promise<any> {\n"
        "  const u = new URL(`https://api.scrapecreators.com${pathname}`);",
        "const SC_BASE = process.env.SCRAPECREATORS_API_URL || 'https://api.scrapecreators.com';\n\n"
        "async function scGet(pathname: string, urlParam: string, trim: boolean): Promise<any> {\n"
        "  const u = new URL(`${SC_BASE}${pathname}`);"
    )

    # 2. Patch profile videos URL
    content = content.replace(
        "const u = new URL('https://api.scrapecreators.com/v3/tiktok/profile/videos');",
        "const u = new URL(`${SC_BASE}/v3/tiktok/profile/videos`);"
    )

    # 3. Add DS_BASE and replace all Doublespeed URLs
    content = content.replace(
        "const DS_MCP_URL = 'https://app.doublespeed.ai/api/mcp';",
        "const DS_BASE = process.env.DOUBLESPEED_API_URL || 'https://app.doublespeed.ai';\n"
        "const DS_MCP_URL = `${DS_BASE}/api/mcp`;"
    )
    content = content.replace(
        "const DS_AUTHORIZATION_ENDPOINT = 'https://app.doublespeed.ai/oauth/authorize';",
        "const DS_AUTHORIZATION_ENDPOINT = `${DS_BASE}/oauth/authorize`;"
    )
    content = content.replace(
        "const DS_TOKEN_ENDPOINT = 'https://app.doublespeed.ai/api/oauth/token';",
        "const DS_TOKEN_ENDPOINT = `${DS_BASE}/api/oauth/token`;"
    )
    content = content.replace(
        "const DS_USERINFO_ENDPOINT = 'https://app.doublespeed.ai/api/oauth/userinfo';",
        "const DS_USERINFO_ENDPOINT = `${DS_BASE}/api/oauth/userinfo`;"
    )

    # 4. Patch review link
    content = content.replace(
        "`https://app.doublespeed.ai/review/${draft.shareLink}`",
        "`${DS_BASE}/review/${draft.shareLink}`"
    )

    if content == original:
        print("WARNING: No changes made — file may already be patched or format differs")
        sys.exit(1)

    with open(path, 'w') as f:
        f.write(content)

    # Verify no hard-coded URLs remain (except fallback defaults)
    remaining = [line for line in content.split('\n')
                 if ('scrapecreators.com' in line or 'doublespeed.ai' in line)
                 and 'process.env' not in line
                 and '||' not in line]
    if remaining:
        print(f"WARNING: {len(remaining)} hard-coded URLs still present:")
        for r in remaining:
            print(f"  {r.strip()}")
    else:
        print("✅ All vendor URLs replaced with env var overrides")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/marketing-agent.ts")
        sys.exit(1)
    patch_file(sys.argv[1])
