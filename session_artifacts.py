#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
SESSION_FILENAME = "session.json"
SESSION_SCHEMA_VERSION = 1

ARTIFACT_BASENAMES = {
    "subtitles_srt": "subtitles.srt",
    "subtitles_vtt": "subtitles.vtt",
    "transcript": "transcript.txt",
    "words": "words.json",
    "energy": "energy.json",
    "cadence": "cadence.json",
    "moments": "moments.json",
    "ranked_moments": "ranked_moments.json",
    "audio": "audio.wav",
    "sermon_metadata_raw": "sermon_metadata.raw.txt",
    "sermon_metadata": "sermon_metadata.json",
    "blog": "blog.md",
    "youtube": "youtube.json",
    "facebook": "facebook.txt",
}


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def speaker_to_slug(speaker: str) -> str:
    if not speaker or not speaker.strip():
        return "Speaker"
    slug = re.sub(r"[^\w\s-]", "", speaker.strip())
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")
    return slug or "Speaker"


def asset_prefix(date_str: str, speaker_slug: str) -> str:
    return f"{date_str}-{speaker_slug}"


def get_video_asset_dir(
    video_path: Path,
    date_str: str | None = None,
    speaker_slug: str | None = None,
) -> Path:
    video_path = video_path.resolve()
    parent = video_path.parent
    if date_str and speaker_slug:
        return parent / asset_prefix(date_str, speaker_slug)
    stem = video_path.stem
    if parent.name.lower() == stem.lower():
        return parent
    return parent / stem


def session_path(asset_dir: Path) -> Path:
    return asset_dir / SESSION_FILENAME


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def load_session(asset_dir: Path) -> dict[str, Any] | None:
    path = session_path(asset_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_session(asset_dir: Path, data: dict[str, Any]) -> Path:
    asset_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["schema_version"] = SESSION_SCHEMA_VERSION
    payload.setdefault("prefix", asset_dir.name)
    path = session_path(asset_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _deep_merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def upsert_session(asset_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    existing = load_session(asset_dir) or {}
    merged = _deep_merge_dicts(existing, updates)
    write_session(asset_dir, merged)
    return merged


def prefixed_artifact_name(prefix: str, base_name: str) -> str:
    return f"{prefix}-{base_name}"


def artifact_filename(
    key: str,
    *,
    prefix: str,
    suffix: str | None = None,
    clip_number: int | None = None,
) -> str:
    if key == "video":
        if not suffix:
            raise ValueError("artifact_filename(video) requires suffix")
        return f"{prefix}-video{suffix}"
    if key == "clip":
        if clip_number is None or not suffix:
            raise ValueError("artifact_filename(clip) requires clip_number and suffix")
        return f"{prefix}-clip_{int(clip_number):03d}{suffix}"
    base_name = ARTIFACT_BASENAMES.get(key)
    if not base_name:
        raise KeyError(f"Unknown artifact key: {key}")
    return prefixed_artifact_name(prefix, base_name)


def session_artifact_filename(session: dict[str, Any] | None, key: str) -> str | None:
    if not isinstance(session, dict):
        return None
    artifacts = session.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def artifact_path(
    asset_dir: Path,
    key: str,
    *,
    session: dict[str, Any] | None = None,
    prefer_session: bool = True,
) -> Path:
    prefix = str((session or {}).get("prefix") or asset_dir.name).strip() or asset_dir.name
    session_name = session_artifact_filename(session, key) if prefer_session else None
    if session_name:
        return asset_dir / session_name
    base_name = ARTIFACT_BASENAMES.get(key)
    if not base_name:
        raise KeyError(f"Unknown artifact key: {key}")
    prefixed = asset_dir / prefixed_artifact_name(prefix, base_name)
    if prefixed.is_file():
        return prefixed
    return asset_dir / base_name


def ranked_report_path(asset_dir: Path, session: dict[str, Any] | None = None) -> Path:
    return artifact_path(asset_dir, "ranked_moments", session=session)


def expected_caption_artifacts(
    asset_dir: Path,
    *,
    prefix: str,
    write_vtt: bool = False,
    keep_audio: bool = False,
) -> dict[str, Path]:
    out = {
        "subtitles_srt": asset_dir / artifact_filename("subtitles_srt", prefix=prefix),
        "transcript": asset_dir / artifact_filename("transcript", prefix=prefix),
        "words": asset_dir / artifact_filename("words", prefix=prefix),
        "energy": asset_dir / artifact_filename("energy", prefix=prefix),
        "cadence": asset_dir / artifact_filename("cadence", prefix=prefix),
        "moments": asset_dir / artifact_filename("moments", prefix=prefix),
    }
    if write_vtt:
        out["subtitles_vtt"] = asset_dir / artifact_filename("subtitles_vtt", prefix=prefix)
    if keep_audio:
        out["audio"] = asset_dir / artifact_filename("audio", prefix=prefix)
    return out


def resolve_main_video_path(asset_dir: Path, session: dict[str, Any] | None = None) -> Path | None:
    session_name = session_artifact_filename(session, "video")
    if session_name:
        candidate = (asset_dir / session_name).resolve()
        if candidate.is_file() and is_video_path(candidate):
            return candidate

    prefix = str((session or {}).get("prefix") or asset_dir.name).strip() or asset_dir.name
    canonical = sorted(asset_dir.glob(f"{prefix}-video.*"))
    for candidate in canonical:
        if candidate.is_file() and is_video_path(candidate):
            return candidate.resolve()

    if isinstance(session, dict):
        source_video_name = str(session.get("source_video_name", "")).strip()
        if source_video_name:
            candidate = (asset_dir / source_video_name).resolve()
            if candidate.is_file() and is_video_path(candidate):
                return candidate

    words_path = artifact_path(asset_dir, "words", session=session)
    if words_path.is_file():
        try:
            words_payload = json.loads(words_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            words_payload = {}
        media_name = str(((words_payload.get("media", {}) or {}).get("file", ""))).strip()
        if media_name:
            candidate = (asset_dir / media_name).resolve()
            if candidate.is_file() and is_video_path(candidate):
                return candidate

    non_clip_videos = []
    clip_videos = []
    for child in sorted(asset_dir.iterdir()):
        if child.is_file() and is_video_path(child):
            if re.search(r"-clip_\d{3}\.", child.name, re.IGNORECASE):
                clip_videos.append(child.resolve())
            else:
                non_clip_videos.append(child.resolve())
    if non_clip_videos:
        return non_clip_videos[0]
    if clip_videos:
        return clip_videos[0]
    return None


def update_words_media_file(words_path: Path, video_filename: str) -> bool:
    try:
        payload = json.loads(words_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    media = payload.get("media")
    if not isinstance(media, dict):
        media = {}
        payload["media"] = media
    media["file"] = video_filename
    try:
        words_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return False
    return True
