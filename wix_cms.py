#!/usr/bin/env python3
"""Wix CMS create-only client using pre-issued bearer token from environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import error, request

def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key, value = key.strip(), value.strip()
    if not key:
        return None
    if value and value[0] in ("'", '"') and value[-1:] == value[0]:
        value = value[1:-1]
    return key, value


def _load_project_env(*, overwrite_wix: bool = False) -> None:
    """Load .env from project dir and cwd. If overwrite_wix=True, set WIX_* (so latest .env wins)."""
    module_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    dotenv_paths = [module_dir / ".env"]
    if cwd != module_dir:
        dotenv_paths.append(cwd / ".env")
    wix_keys = {"WIX_BEARER_TOKEN", "WIX_COLLECTION_ID", "WIX_SITE_ID", "WIX_API_BASE", "WIX_TITLE_FIELD", "WIX_BLOG_MEMBER_ID"}
    for dotenv_path in dotenv_paths:
        if not dotenv_path.is_file():
            continue
        try:
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                parsed = _parse_dotenv_line(line)
                if not parsed:
                    continue
                key, value = parsed
                if overwrite_wix and key in wix_keys:
                    os.environ[key] = value
                else:
                    os.environ.setdefault(key, value)
        except OSError:
            continue


_load_project_env()

DEFAULT_API_BASE = "https://www.wixapis.com"


def get_wix_config() -> tuple[str, str, str, str]:
    """Return (bearer_token, collection_id, api_base, site_id). Raises ValueError if any required value missing."""
    _load_project_env(overwrite_wix=True)  # use latest .env from cwd so GUI always sees current settings
    token = (os.environ.get("WIX_BEARER_TOKEN") or "").strip()
    collection_id = (os.environ.get("WIX_COLLECTION_ID") or "").strip()
    base = (os.environ.get("WIX_API_BASE") or DEFAULT_API_BASE).rstrip("/")
    site_id = (os.environ.get("WIX_SITE_ID") or "").strip()
    if not token:
        raise ValueError("WIX_BEARER_TOKEN is not set. Add it to .env or environment.")
    if not collection_id:
        raise ValueError("WIX_COLLECTION_ID is not set. Add it to .env or environment.")
    if not site_id:
        raise ValueError(
            "WIX_SITE_ID is not set. Add your Wix site ID (MetaSite ID) to .env. "
            "Find it in the site dashboard URL after /dashboard/ or in Wix Developer Center."
        )
    return token, collection_id, base, site_id


def sermon_payload_to_wix_data(
    payload: dict,
    transcript: str = "",
    video_url: str = "",
    name: str = "",
    date_preached: str = "",
) -> dict:
    """
    Map normalized sermon metadata + optional transcript, videoUrl, name, date to Wix data item payload.
    Returns a flat dict suitable for dataItem.data (field names as used in Wix collection).
    date_preached should be YYYY-MM-DD for Wix Date type.
    """
    title_value = payload.get("title") or ""
    title_key = (os.environ.get("WIX_TITLE_FIELD") or "title").strip() or "title"
    data = {
        title_key: title_value,
        "description": payload.get("description") or "",
        "scriptures": payload.get("scriptures") or [],
        "mainPoints": payload.get("mainPoints") or [],
        "tags": payload.get("tags") or [],
        "propheticStatements": payload.get("propheticStatements") or [],
        "keyMoments": payload.get("keyMoments") or [],
        "topics": payload.get("topics") or [],
        "teachingStatements": payload.get("teachingStatements") or [],
    }
    if transcript:
        data["transcript"] = transcript
    if video_url:
        data["videoUrl"] = video_url
    if name:
        data["name"] = name
    if date_preached:
        data["date"] = date_preached
    return data


def create_sermon_item(
    payload: dict,
    transcript: str = "",
    video_url: str = "",
    name: str = "",
    date_preached: str = "",
    timeout_sec: int = 30,
) -> dict:
    """
    Create a new sermon item in the configured Wix collection (create-only).
    payload: normalized sermon metadata from parse_sermon_metadata().
    date_preached: YYYY-MM-DD string for Wix Date field.
    Returns the API response JSON. Raises ValueError for config/auth errors, RuntimeError for API errors.
    """
    token, collection_id, base, site_id = get_wix_config()
    data = sermon_payload_to_wix_data(
        payload,
        transcript=transcript,
        video_url=video_url,
        name=name,
        date_preached=date_preached,
    )

    url = f"{base}/wix-data/v2/items"
    body = json.dumps(
        {"dataCollectionId": collection_id, "dataItem": {"data": data}},
        ensure_ascii=False,
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": token if token.startswith("Bearer ") else f"Bearer {token}",
        "wix-site-id": site_id,
    }
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
            err = json.loads(body) if body.strip() else {}
        except Exception:
            body = e.read().decode("utf-8", errors="replace")
            err = {"message": body or e.reason}
        code = e.code
        if code == 401:
            raise ValueError("Wix authentication failed (401). Check WIX_BEARER_TOKEN.") from e
        if code == 403:
            raise ValueError("Wix permission denied (403). Check app scope SCOPE.DC-DATA.WRITE.") from e
        msg = err.get("message") or err.get("error") or str(err)
        raise RuntimeError(f"Wix API error ({code}): {msg}") from e
    except error.URLError as e:
        raise RuntimeError(f"Wix request failed: {e.reason}") from e
