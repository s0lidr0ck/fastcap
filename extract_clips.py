#!/usr/bin/env python3
"""
Extract clips from a video using in/out points from JSON.
Uses ffmpeg stream copy (no transcoding). Outputs one file per clip, labeled by clip number.

Requires: ffmpeg on PATH, Python 3.9+

Example JSON:
  {
    "clips": [
      { "clip_number": 1, "start_time": "00:00:42.579", "end_time": "00:01:39.299" },
      ...
    ]
  }
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from session_artifacts import artifact_filename


def parse_timestamp_to_seconds(ts: str) -> float:
    """Parse HH:MM:SS.mmm string into seconds."""
    try:
        hhmm, sec = ts.rsplit(":", 1)
        hh, mm = hhmm.split(":")
        return int(hh) * 3600 + int(mm) * 60 + float(sec)
    except ValueError as e:
        raise ValueError(
            f"Invalid timestamp '{ts}'. Expected format HH:MM:SS.mmm"
        ) from e


def format_seconds_to_timestamp(total_seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm (millisecond precision)."""
    if total_seconds < 0:
        total_seconds = 0.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds - (hours * 3600 + minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def load_clips_from_data(data: dict) -> list[dict]:
    """Validate clips payload. Returns list of clip dicts."""
    clips = data.get("clips")
    if not clips:
        raise ValueError("JSON must contain a non-empty 'clips' array")
    for c in clips:
        if "clip_number" not in c or "start_time" not in c or "end_time" not in c:
            raise ValueError(
                "Each clip must have 'clip_number', 'start_time', and 'end_time'"
            )
        # Validate timestamp format early for clearer errors.
        parse_timestamp_to_seconds(c["start_time"])
        parse_timestamp_to_seconds(c["end_time"])
    return clips


def load_clips_from_file(json_path: Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return load_clips_from_data(data)


def load_clips_from_stdin() -> list[dict]:
    """
    Read JSON from stdin so users can paste directly in terminal.
    On Windows PowerShell: paste JSON, then press Ctrl+Z and Enter.
    """
    payload = sys.stdin.read().strip()
    if not payload:
        raise ValueError("No JSON received from stdin.")
    data = json.loads(payload)
    return load_clips_from_data(data)


def extract_clip(
    video_path: Path,
    output_path: Path,
    start_time: str,
    duration_seconds: float,
) -> None:
    """
    Extract one segment with ffmpeg using stream copy.
    -ss before -i for fast input seeking; -t enforces exact output duration cap.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", start_time,
        "-i", str(video_path),
        "-t", f"{duration_seconds:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"ffmpeg failed with code {result.returncode} for clip {output_path.name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract clips from a video using JSON in/out points (stream copy, no transcode)."
    )
    parser.add_argument(
        "video_file",
        type=Path,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        type=Path,
        default=None,
        help="Optional JSON file path. Omit when using --json-stdin.",
    )
    parser.add_argument(
        "--json-stdin",
        action="store_true",
        help="Read clips JSON from stdin (paste JSON directly).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Directory for extracted clips. Default: same directory as video.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="clip",
        help="Filename prefix for outputs (default: clip -> clip_001.mp4).",
    )
    parser.add_argument(
        "--pad-seconds",
        type=float,
        default=5.0,
        help="Seconds to extend before start and after end of each clip (default: 5).",
    )
    args = parser.parse_args()

    video_path = args.video_file.resolve()

    if not video_path.is_file():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if args.pad_seconds < 0:
        print("Error: --pad-seconds must be >= 0", file=sys.stderr)
        sys.exit(1)
    if not args.json_stdin and args.json_file is None:
        print(
            "Error: Provide json_file or use --json-stdin to paste JSON.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.json_stdin and args.json_file is not None:
        print(
            "Error: Use either json_file or --json-stdin, not both.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.json_file is not None and not args.json_file.resolve().is_file():
        print(f"Error: JSON file not found: {args.json_file.resolve()}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir.resolve() if args.output_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.json_stdin:
            print("Paste JSON, then end input (Ctrl+Z then Enter on Windows).")
            clips = load_clips_from_stdin()
        else:
            clips = load_clips_from_file(args.json_file.resolve())
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    suffix = video_path.suffix or ".mp4"

    for c in clips:
        num = c["clip_number"]
        raw_start = c["start_time"]
        raw_end = c["end_time"]
        start_sec = max(0.0, parse_timestamp_to_seconds(raw_start) - args.pad_seconds)
        end_sec = parse_timestamp_to_seconds(raw_end) + args.pad_seconds
        if end_sec <= start_sec:
            print(
                f"Error: clip {num} has invalid range after padding: {raw_start} -> {raw_end}",
                file=sys.stderr,
            )
            sys.exit(1)
        start = format_seconds_to_timestamp(start_sec)
        end = format_seconds_to_timestamp(end_sec)
        duration = end_sec - start_sec
        out_name = (
            f"clip_{int(num):03d}{suffix}"
            if args.prefix == "clip"
            else artifact_filename("clip", prefix=args.prefix, clip_number=int(num), suffix=suffix)
        )
        out_path = out_dir / out_name
        print(
            f"Extracting clip {num}: "
            f"{raw_start} -> {raw_end} (padded to {start} -> {end}) -> {out_path.name}"
        )
        try:
            extract_clip(video_path, out_path, start, duration)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Done. Extracted {len(clips)} clips to {out_dir}")


if __name__ == "__main__":
    main()
