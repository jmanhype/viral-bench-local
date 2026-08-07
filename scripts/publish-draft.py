#!/usr/bin/env python3
"""
publish-draft.py — CLI to publish a draft via the publisher service.

Usage:
    python scripts/publish-draft.py <draft_id> [--account-id ID] [--token TOKEN]
"""

import argparse
import json
import sys

import httpx


def main():
    parser = argparse.ArgumentParser(description="Publish a draft to TikTok")
    parser.add_argument("draft_id", help="Draft ID to publish")
    parser.add_argument("--account-id", default="default", help="TikTok account ID")
    parser.add_argument("--token", default="", help="TikTok access token (or set TIKTOK_ACCESS_TOKEN env)")
    parser.add_argument("--host", default="http://localhost:8030", help="Publisher service URL")
    args = parser.parse_args()

    import os
    access_token = args.token or os.environ.get("TIKTOK_ACCESS_TOKEN", "")

    payload = {
        "draft_id": args.draft_id,
        "account_id": args.account_id,
        "access_token": access_token,
    }

    print(f"Publishing draft: {args.draft_id}")
    print(f"Service: {args.host}")

    try:
        resp = httpx.post(
            f"{args.host}/publish",
            json=payload,
            timeout=60.0,
        )
        result = resp.json()

        if resp.status_code == 200:
            print("\n✅ Publish successful!")
            print(json.dumps(result, indent=2))
        else:
            print(f"\n❌ Publish failed ({resp.status_code})")
            print(json.dumps(result, indent=2))
            sys.exit(1)

    except httpx.ConnectError:
        print(f"\n❌ Could not connect to publisher at {args.host}")
        print("   Start the service: python services/publisher/app.py")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
