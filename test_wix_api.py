#!/usr/bin/env python3
"""
Test Wix Data API request/response. Run from project root with .env set.
Prints config (sanitized), then tries query and insert; prints full response.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib import error, request

# Load .env
def _load_env() -> None:
    for dotenv_path in [Path(__file__).resolve().parent / ".env", Path.cwd() / ".env"]:
        if not dotenv_path.is_file():
            continue
        try:
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if val and val[0] in ("'", '"') and val[-1:] == val[0]:
                    val = val[1:-1]
                os.environ.setdefault(key.strip(), val)
        except OSError:
            pass


_load_env()

BASE = (os.environ.get("WIX_API_BASE") or "https://www.wixapis.com").rstrip("/")
TOKEN = (os.environ.get("WIX_BEARER_TOKEN") or "").strip()
COLLECTION_ID = (os.environ.get("WIX_COLLECTION_ID") or "").strip()
SITE_ID = (os.environ.get("WIX_SITE_ID") or "").strip()


def wix_request(
    path: str,
    body: dict,
    *,
    extra_headers: dict | None = None,
    timeout: int = 30,
) -> tuple[int, dict, str]:
    """POST to wix-data path. Returns (status_code, response_json, raw_body)."""
    url = f"{BASE}{path}"
    auth = TOKEN if TOKEN.startswith("Bearer ") else f"Bearer {TOKEN}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": auth,
    }
    if extra_headers:
        headers.update(extra_headers)
    req = request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {}
            return resp.status, data, raw
    except error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            data = {"raw": raw}
        return e.code, data, raw


def main() -> None:
    print("Wix config (sanitized):")
    print("  WIX_BEARER_TOKEN: ", "set (" + str(len(TOKEN)) + " chars)" if TOKEN else "NOT SET")
    print("  WIX_COLLECTION_ID:", "set (" + str(len(COLLECTION_ID)) + " chars)" if COLLECTION_ID else "NOT SET")
    print("  WIX_SITE_ID:      ", repr(SITE_ID[:8] + "..." + SITE_ID[-4:]) if len(SITE_ID) > 12 else repr(SITE_ID) if SITE_ID else "NOT SET")
    print("  WIX_API_BASE:    ", BASE)
    print()

    if not TOKEN or not COLLECTION_ID:
        print("Set WIX_BEARER_TOKEN and WIX_COLLECTION_ID in .env")
        sys.exit(1)

    # Test 1: Query with wix-site-id
    print("--- 1) POST /wix-data/v2/items/query (with wix-site-id) ---")
    status, data, raw = wix_request(
        "/wix-data/v2/items/query",
        {"dataCollectionId": COLLECTION_ID, "query": {"paging": {"limit": 1}}},
        extra_headers={"wix-site-id": SITE_ID} if SITE_ID else None,
    )
    print("Status:", status)
    print("Response:", json.dumps(data, indent=2, ensure_ascii=False)[:2000])
    if len(raw) > 2000:
        print("... (truncated)")
    print()

    # If that failed, try without wix-site-id to see if error changes
    if status != 200 and SITE_ID:
        print("--- 2) Same query WITHOUT wix-site-id ---")
        status2, data2, _ = wix_request(
            "/wix-data/v2/items/query",
            {"dataCollectionId": COLLECTION_ID, "query": {"paging": {"limit": 1}}},
        )
        print("Status:", status2)
        print("Response:", json.dumps(data2, indent=2, ensure_ascii=False)[:1500])
        print()
    elif status == 200:
        print("Query succeeded. Trying insert with minimal data...")
        print("--- 3) POST /wix-data/v2/items (insert minimal item) ---")
        status3, data3, raw3 = wix_request(
            "/wix-data/v2/items",
            {
                "dataCollectionId": COLLECTION_ID,
                "dataItem": {"data": {"title": "FastCap API test item (delete me)"}},
            },
            extra_headers={"wix-site-id": SITE_ID} if SITE_ID else None,
        )
        print("Status:", status3)
        print("Response:", json.dumps(data3, indent=2, ensure_ascii=False)[:1500])
        if status3 != 200:
            print("Full body:", raw3[:1000])


if __name__ == "__main__":
    main()
