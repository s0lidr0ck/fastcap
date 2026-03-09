#!/usr/bin/env python3
"""
List site members via Wix Members API. Use this to get a valid member ID
for WIX_BLOG_MEMBER_ID (Blog API expects a site member ID, not a contact ID).
Requires .env with WIX_BEARER_TOKEN and WIX_SITE_ID.
Members API scope: SCOPE.DC-MEMBERS.READ-MEMBERS
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib import error, request

# Load .env (same pattern as wix_cms)
def _load_env() -> None:
    from wix_cms import get_wix_config
    try:
        get_wix_config()
    except ValueError:
        pass


def main() -> None:
    _load_env()
    token = (os.environ.get("WIX_BEARER_TOKEN") or "").strip()
    site_id = (os.environ.get("WIX_SITE_ID") or "").strip()
    base = (os.environ.get("WIX_API_BASE") or "https://www.wixapis.com").rstrip("/")

    if not token:
        print("WIX_BEARER_TOKEN not set in .env")
        sys.exit(1)
    if not site_id:
        print("WIX_SITE_ID not set in .env")
        sys.exit(1)

    url = f"{base}/members/v1/members/query"
    body = json.dumps({
        "query": {"paging": {"limit": 20}},
        "fieldsets": ["FULL"],
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": token if token.startswith("Bearer ") else f"Bearer {token}",
        "wix-site-id": site_id,
    }
    req = request.Request(url, data=body, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            err = json.loads(err_body)
            msg = err.get("message") or err.get("error") or err_body
        except Exception:
            msg = err_body
        print(f"Members API error ({e.code}): {msg}")
        if e.code == 403:
            print("Your token may need scope: SCOPE.DC-MEMBERS.READ-MEMBERS")
        sys.exit(1)

    members = data.get("members") or []
    if not members:
        print("No members returned. (Site may have no members, or token may lack READ-MEMBERS scope.)")
        sys.exit(0)

    print(f"Site ID: {site_id}")
    print(f"Found {len(members)} member(s). Use one of these IDs as WIX_BLOG_MEMBER_ID:\n")
    for m in members:
        mid = m.get("id") or ""
        email = (m.get("loginEmail") or "").strip()
        contact = m.get("contact") or {}
        first = (contact.get("firstName") or "").strip()
        last = (contact.get("lastName") or "").strip()
        name = f"{first} {last}".strip() or "(no name)"
        profile = m.get("profile") or {}
        nickname = (profile.get("nickname") or "").strip()
        label = nickname or email or name or mid
        print(f"  {mid}")
        print(f"    -> {label}")
        if email and email != label:
            print(f"       email: {email}")
        print()
    print("Copy one of the IDs above into .env as WIX_BLOG_MEMBER_ID (36-char GUID only).")


if __name__ == "__main__":
    main()
