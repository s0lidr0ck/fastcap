#!/usr/bin/env python3
r"""
Rename files in existing asset folders to the convention:
  {Date}-{Speaker}-{Asset}.{ext}

Usage:
  python rename_assets_to_convention.py --folder "C:\path\to\asset\folder" --date 2026-03-01 --speaker "Pastor Chris"
  python rename_assets_to_convention.py --folder "C:\path\to\folder" --date 2026-03-01 --speaker "Chris Tidwell" --execute

Without --execute, only prints what would be renamed (dry run).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from session_artifacts import (
    VIDEO_EXTENSIONS,
    asset_prefix,
    load_session,
    speaker_to_slug,
    update_words_media_file,
    write_session,
)

CLIP_PATTERN = re.compile(r"^(.+)_(\d{3})\.(\w+)$", re.IGNORECASE)  # prefix_001.mp4


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def plan_renames(asset_dir: Path, prefix: str) -> list[tuple[Path, Path]]:
    """Plan renames for files in asset_dir to {prefix}-{asset}.{ext}. Returns [(old_path, new_path), ...]."""
    renames: list[tuple[Path, Path]] = []
    seen_new: set[Path] = set()

    # Files that map 1:1 from legacy name to prefixed name
    legacy_json = ["words.json", "energy.json", "cadence.json", "moments.json", "ranked_moments.json"]
    for name in legacy_json:
        old_p = asset_dir / name
        if old_p.is_file() and old_p.name != f"{prefix}-{name}":
            new_p = asset_dir / f"{prefix}-{name}"
            if new_p not in seen_new and (not new_p.is_file() or new_p == old_p):
                renames.append((old_p, new_p))
                seen_new.add(new_p)

    # Single .srt -> {prefix}-subtitles.srt
    srt_files = [f for f in asset_dir.iterdir() if f.is_file() and f.suffix.lower() == ".srt"]
    if len(srt_files) == 1 and not (asset_dir / f"{prefix}-subtitles.srt").is_file():
        old_p = srt_files[0]
        if not old_p.name.startswith(prefix + "-"):
            new_p = asset_dir / f"{prefix}-subtitles.srt"
            if new_p not in seen_new:
                renames.append((old_p, new_p))
                seen_new.add(new_p)

    # Single .txt (plain transcript; skip if we already have prefix-transcript.txt)
    txt_files = [f for f in asset_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
    if len(txt_files) == 1 and not (asset_dir / f"{prefix}-transcript.txt").is_file():
        old_p = txt_files[0]
        if not old_p.name.startswith(prefix + "-"):
            new_p = asset_dir / f"{prefix}-transcript.txt"
            if new_p not in seen_new:
                renames.append((old_p, new_p))
                seen_new.add(new_p)

    # Single .wav -> {prefix}-audio.wav
    wav_files = [f for f in asset_dir.iterdir() if f.is_file() and f.suffix.lower() == ".wav"]
    if len(wav_files) == 1 and not (asset_dir / f"{prefix}-audio.wav").is_file():
        old_p = wav_files[0]
        if not old_p.name.startswith(prefix + "-"):
            new_p = asset_dir / f"{prefix}-audio.wav"
            if new_p not in seen_new:
                renames.append((old_p, new_p))
                seen_new.add(new_p)

    # Single .vtt -> {prefix}-subtitles.vtt
    vtt_files = [f for f in asset_dir.iterdir() if f.is_file() and f.suffix.lower() == ".vtt"]
    if len(vtt_files) == 1 and not (asset_dir / f"{prefix}-subtitles.vtt").is_file():
        old_p = vtt_files[0]
        if not old_p.name.startswith(prefix + "-"):
            new_p = asset_dir / f"{prefix}-subtitles.vtt"
            if new_p not in seen_new:
                renames.append((old_p, new_p))
                seen_new.add(new_p)

    # Video clips: anything matching something_001.mp4 -> {prefix}-clip_001.mp4
    # Other videos: single main video -> {prefix}-video.ext
    video_files = [f for f in asset_dir.iterdir() if f.is_file() and is_video_path(f)]
    clip_videos = []
    other_videos = []
    for f in video_files:
        if f.name.startswith(prefix + "-"):
            continue
        if CLIP_PATTERN.match(f.name):
            clip_videos.append(f)
        else:
            other_videos.append(f)
    for f in clip_videos:
        match = CLIP_PATTERN.match(f.name)
        if match:
            _, num, ext = match.groups()
            new_name = f"{prefix}-clip_{num}.{ext}"
            new_p = asset_dir / new_name
            if new_p not in seen_new and new_p != f:
                renames.append((f, new_p))
                seen_new.add(new_p)
    if len(other_videos) == 1 and not (asset_dir / f"{prefix}-video{other_videos[0].suffix}").is_file():
        old_p = other_videos[0]
        new_p = asset_dir / f"{prefix}-video{old_p.suffix}"
        if new_p not in seen_new:
            renames.append((old_p, new_p))
            seen_new.add(new_p)

    return renames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename asset folder files to {Date}-{Speaker}-{Asset}.{ext}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--folder",
        required=True,
        action="append",
        metavar="DIR",
        help="Asset folder path (can be repeated for multiple folders).",
    )
    parser.add_argument(
        "--date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Date for the prefix (e.g. 2026-03-01).",
    )
    parser.add_argument(
        "--speaker",
        required=True,
        metavar="NAME",
        help="Speaker name (e.g. 'Pastor Chris' or 'Chris Tidwell'); will be slugified for filenames.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename files. Default is dry run (print only).",
    )
    parser.add_argument(
        "--rename-folder",
        action="store_true",
        help="If the folder name is not already {Date}-{Speaker}, rename the folder (requires --execute).",
    )
    args = parser.parse_args()

    date_str = args.date.strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        print("Error: --date must be YYYY-MM-DD", file=sys.stderr)
        return 1
    speaker_slug = speaker_to_slug(args.speaker)
    prefix = asset_prefix(date_str, speaker_slug)

    total_renames = 0
    for folder_raw in args.folder:
        asset_dir = Path(folder_raw).resolve()
        if not asset_dir.is_dir():
            print(f"Skip (not a directory): {asset_dir}", file=sys.stderr)
            continue

        renames = plan_renames(asset_dir, prefix)
        applied_renames: dict[str, str] = {}
        if not renames:
            print(f"No renames planned for: {asset_dir}")
            continue

        print(f"\n{asset_dir}")
        for old_p, new_p in renames:
            print(f"  {old_p.name}  ->  {new_p.name}")
            if args.execute:
                if new_p.exists() and new_p != old_p:
                    print(f"    WARNING: target exists, skip: {new_p.name}", file=sys.stderr)
                else:
                    old_p.rename(new_p)
                    applied_renames[old_p.name] = new_p.name
                    total_renames += 1

        final_dir = asset_dir
        if args.execute and args.rename_folder and asset_dir.name != prefix:
            parent = asset_dir.parent
            new_dir = parent / prefix
            if new_dir.exists() and new_dir != asset_dir:
                print(f"  WARNING: target folder exists, skip rename: {asset_dir.name} -> {prefix}", file=sys.stderr)
            else:
                asset_dir.rename(new_dir)
                print(f"  Folder renamed: {asset_dir.name}  ->  {prefix}")
                final_dir = new_dir

        if args.execute and applied_renames:
            session = load_session(final_dir) or {}
            if session:
                artifacts = session.get("artifacts", {}) if isinstance(session.get("artifacts", {}), dict) else {}
                updated_artifacts = dict(artifacts)
                for key, filename in list(updated_artifacts.items()):
                    if isinstance(filename, str) and filename in applied_renames:
                        updated_artifacts[key] = applied_renames[filename]
                source_name = str(session.get("source_video_name", "")).strip()
                if source_name in applied_renames:
                    session["source_video_name"] = applied_renames[source_name]
                session["prefix"] = prefix
                session["artifacts"] = updated_artifacts
                write_session(final_dir, session)

            renamed_video_names = [name for name in applied_renames.values() if re.search(r"-video\.[^.]+$", name, re.IGNORECASE)]
            if renamed_video_names:
                candidate_words = [f"{prefix}-words.json", "words.json"]
                if session and isinstance(session.get("artifacts", {}), dict):
                    words_name = session["artifacts"].get("words")
                    if isinstance(words_name, str) and words_name not in candidate_words:
                        candidate_words.insert(0, words_name)
                for words_name in candidate_words:
                    words_path = final_dir / words_name
                    if words_path.is_file():
                        update_words_media_file(words_path, renamed_video_names[0])
                        break

    if not args.execute and total_renames == 0:
        print("\nDry run. Use --execute to apply renames.", file=sys.stderr)
    elif args.execute:
        print(f"\nRenamed {total_renames} file(s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
