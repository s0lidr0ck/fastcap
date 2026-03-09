#!/usr/bin/env python3
"""
Closed-caption video by extracting audio with ffmpeg and transcribing with faster-whisper.
Outputs SRT, TXT (and optionally VTT) alongside the video file.

Requires: ffmpeg on PATH, Python 3.9+
"""

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from session_artifacts import artifact_filename, now_iso, upsert_session


def video_to_audio(video_path: Path, audio_path: Path, sample_rate: int = 16000) -> None:
    """Extract audio from video to mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", str(video_path),
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",  # mono
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}. Is ffmpeg installed and on PATH?")


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap_caption_text(text: str, max_chars_per_line: int) -> str:
    """Wrap caption text to a max line length using word boundaries."""
    text = text.strip()
    if not text or max_chars_per_line <= 0:
        return text

    words = text.split()
    if not words:
        return ""

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars_per_line:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def split_words_into_lines(words, max_chars_per_line: int):
    """Split a faster-whisper words list into line groups by max character length."""
    lines = []
    current = []
    current_len = 0

    for w in words:
        token = (getattr(w, "word", "") or "").strip()
        if not token:
            continue
        add_len = len(token) + (1 if current else 0)
        if max_chars_per_line > 0 and current and (current_len + add_len) > max_chars_per_line:
            lines.append(current)
            current = [w]
            current_len = len(token)
        else:
            current.append(w)
            current_len += add_len

    if current:
        lines.append(current)
    return lines


def segments_to_cues(segments, max_chars_per_line: int = 20, max_lines_per_cue: int = 2):
    """Build subtitle cues from segments with timing-aware line/cue splitting."""
    cues = []
    for seg in segments:
        words = list(getattr(seg, "words", []) or [])
        if not words:
            text = wrap_caption_text(seg.text, max_chars_per_line)
            if text:
                cues.append((float(seg.start), float(seg.end), text))
            continue

        lines = split_words_into_lines(words, max_chars_per_line)
        if not lines:
            continue

        step = max(1, max_lines_per_cue)
        for i in range(0, len(lines), step):
            cue_lines = lines[i : i + step]
            line_texts = []
            for line_words in cue_lines:
                line_texts.append(" ".join((getattr(w, "word", "") or "").strip() for w in line_words).strip())
            text = "\n".join(t for t in line_texts if t)
            if not text:
                continue
            first_word = cue_lines[0][0]
            last_word = cue_lines[-1][-1]
            start = float(getattr(first_word, "start", seg.start))
            end = float(getattr(last_word, "end", seg.end))
            cues.append((start, end, text))

    return cues


def cues_to_srt(cues) -> str:
    """Convert prepared cues to SRT text."""
    lines = []
    for i, (start_s, end_s, text) in enumerate(cues, 1):
        start = format_srt_time(start_s)
        end = format_srt_time(end_s)
        if not text:
            continue
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def segments_to_txt(segments) -> str:
    """Convert faster-whisper segments to plain text (one segment per line)."""
    lines = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            lines.append(text)
    return "\n".join(lines) + ("\n" if lines else "")


def format_vtt_time(seconds: float) -> str:
    """Convert seconds to VTT timestamp HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def cues_to_vtt(cues) -> str:
    """Convert prepared cues to WebVTT text."""
    lines = ["WEBVTT", ""]
    for start_s, end_s, text in cues:
        start = format_vtt_time(start_s)
        end = format_vtt_time(end_s)
        if not text:
            continue
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def format_progress_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS for progress display."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def _clean_word_token(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return ""
    return token


def words_payload_from_segments(
    video_path: Path,
    duration_sec: float,
    sample_rate: int,
    segments,
    stored_media_name: str | None = None,
) -> dict:
    words = []
    for seg in segments:
        for w in (getattr(seg, "words", []) or []):
            token = _clean_word_token(getattr(w, "word", ""))
            start = float(getattr(w, "start", 0.0) or 0.0)
            end = float(getattr(w, "end", start) or start)
            if not token:
                continue
            words.append(
                {
                    "w": token,
                    "s": round(start, 3),
                    "e": round(end, 3),
                    "c": round(float(getattr(w, "probability", 1.0) or 1.0), 3),
                }
            )
    return {
        "media": {
            "file": (stored_media_name or video_path.name),
            "source_file": video_path.name,
            "duration_sec": round(float(duration_sec or 0.0), 3),
            "sample_rate": int(sample_rate),
        },
        "words": words,
    }


def compute_energy_map(wav_path: Path, window_sec: float = 0.1, hop_sec: float = 0.1) -> dict:
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)

    if sample_width != 2:
        raise RuntimeError("Expected 16-bit PCM WAV from ffmpeg extraction.")

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size == 0:
        return {"window_sec": window_sec, "hop_sec": hop_sec, "frames": []}

    frame_size = max(1, int(round(window_sec * sample_rate)))
    hop_size = max(1, int(round(hop_sec * sample_rate)))
    eps = 1e-12
    max_i16 = 32768.0
    out = []
    i = 0
    while i < samples.size:
        frame = samples[i : i + frame_size]
        if frame.size == 0:
            break
        norm = frame / max_i16
        rms = float(np.sqrt(np.mean(norm * norm)))
        peak = float(np.max(np.abs(norm)))
        rms_db = 20.0 * math.log10(max(rms, eps))
        peak_db = 20.0 * math.log10(max(peak, eps))
        out.append(
            {
                "t": round(i / sample_rate, 3),
                "rms_db": round(rms_db, 1),
                "peak_db": round(peak_db, 1),
            }
        )
        i += hop_size

    return {"window_sec": window_sec, "hop_sec": hop_sec, "frames": out}


def _normalize_0_100(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (value - lo) / (hi - lo)))


def build_cadence_payload(words_payload: dict, pause_threshold_sec: float = 0.3) -> dict:
    words = words_payload.get("words", [])
    if not words:
        return {"phrases": []}

    phrases = []
    current = [words[0]]
    for prev, nxt in zip(words, words[1:]):
        gap = float(nxt["s"]) - float(prev["e"])
        hard_stop = str(prev["w"]).strip().endswith((".", "!", "?"))
        if gap > pause_threshold_sec or hard_stop:
            phrases.append(current)
            current = [nxt]
        else:
            current.append(nxt)
    if current:
        phrases.append(current)

    cadence_phrases = []
    prev_wps = None
    for phrase in phrases:
        start = float(phrase[0]["s"])
        end = float(phrase[-1]["e"])
        duration = max(0.001, end - start)
        phrase_words = [w["w"] for w in phrase]
        text = " ".join(t.strip() for t in phrase_words if t.strip())
        word_count = len(phrase_words)
        wps = word_count / duration

        pauses_ms = []
        for a, b in zip(phrase, phrase[1:]):
            gap = max(0.0, float(b["s"]) - float(a["e"]))
            pauses_ms.append(gap * 1000.0)
        avg_pause_ms = sum(pauses_ms) / len(pauses_ms) if pauses_ms else 0.0
        max_pause_ms = max(pauses_ms) if pauses_ms else 0.0

        markers = []
        lowered_tokens = [re.sub(r"[^\w']", "", t.lower()) for t in phrase_words]
        lowered_tokens = [t for t in lowered_tokens if t]
        repeated = any(lowered_tokens.count(t) >= 2 for t in set(lowered_tokens))
        if repeated:
            markers.append("repetition")
        if sum(1 for t in phrase_words if t.strip().endswith((".", "!", "?"))) >= 2:
            markers.append("stacked_statements")
        if max_pause_ms >= 350:
            markers.append("pause_punch")
        if prev_wps is not None:
            if wps - prev_wps >= 0.9:
                markers.append("pace_shift_fast")
            elif prev_wps - wps >= 0.9:
                markers.append("pace_shift_slow")
        if prev_wps is not None and wps > prev_wps:
            markers.append("rising_intensity")

        score = (
            _normalize_0_100(wps, 1.0, 5.0) * 0.45
            + _normalize_0_100(max_pause_ms, 0, 650) * 0.25
            + min(30.0, len(markers) * 8.0)
        )
        cadence_score = int(round(max(0.0, min(100.0, score))))

        cadence_phrases.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "word_count": word_count,
                "words_per_sec": round(wps, 2),
                "avg_pause_ms": int(round(avg_pause_ms)),
                "max_pause_ms": int(round(max_pause_ms)),
                "markers": markers,
                "cadence_score": cadence_score,
            }
        )
        prev_wps = wps

    return {"phrases": cadence_phrases}


def build_moments_payload(words_payload: dict, energy_payload: dict, cadence_payload: dict) -> dict:
    duration = float(words_payload.get("media", {}).get("duration_sec", 0.0) or 0.0)
    if duration <= 0:
        words = words_payload.get("words", [])
        if words:
            duration = float(words[-1]["e"])
    if duration <= 0:
        return {"moments": []}

    words = words_payload.get("words", [])
    cadence_phrases = cadence_payload.get("phrases", [])
    frames = energy_payload.get("frames", [])
    rms_values = [float(f["rms_db"]) for f in frames] or [-60.0]
    rms_global_min = min(rms_values)
    rms_global_max = max(rms_values)

    window_sec = 45.0
    hop_sec = 15.0
    moments = []
    t = 0.0
    while t < duration:
        end_t = min(duration, t + window_sec)
        if end_t - t < 20.0:
            break

        win_phrases = [p for p in cadence_phrases if p["end"] > t and p["start"] < end_t]
        win_words = [w for w in words if float(w["e"]) >= t and float(w["s"]) <= end_t]
        win_frames = [f for f in frames if float(f["t"]) >= t and float(f["t"]) <= end_t]

        if not win_words:
            t += hop_sec
            continue

        hook = " ".join(str(w["w"]).strip() for w in win_words[:8]).strip()
        if len(hook) > 80:
            hook = hook[:77].rstrip() + "..."

        cadence_score = 0.0
        cadence_markers = []
        if win_phrases:
            cadence_score = float(sum(float(p["cadence_score"]) for p in win_phrases) / len(win_phrases))
            marker_set = set()
            for p in win_phrases:
                marker_set.update(p.get("markers", []))
            cadence_markers = sorted(marker_set)

        if win_frames:
            rms_win = [float(f["rms_db"]) for f in win_frames]
            energy_score = _normalize_0_100(sum(rms_win) / len(rms_win), rms_global_min, rms_global_max)
            contrast_score = _normalize_0_100(max(rms_win) - min(rms_win), 2.0, 20.0)
        else:
            energy_score = 0.0
            contrast_score = 0.0

        hook_punct_hits = sum(1 for w in win_words if str(w["w"]).strip().endswith(("!", "?")))
        hook_density = _normalize_0_100(hook_punct_hits, 0, 8)

        reasons = list(cadence_markers)
        if energy_score >= 65:
            reasons.append("energy_lift")
        if contrast_score >= 60:
            reasons.append("dynamic_contrast")
        if any(
            (float(b["s"]) - float(a["e"])) >= 0.35
            for a, b in zip(win_words, win_words[1:])
        ):
            reasons.append("clean_pause_cut")
        reasons = sorted(set(reasons))

        overall = (
            cadence_score * 0.4
            + energy_score * 0.3
            + contrast_score * 0.2
            + hook_density * 0.1
        )

        moments.append(
            {
                "start": round(t, 3),
                "end": round(end_t, 3),
                "hook": hook,
                "cadence_score": int(round(cadence_score)),
                "energy_score": int(round(energy_score)),
                "contrast_score": int(round(contrast_score)),
                "overall": int(round(max(0.0, min(100.0, overall)))),
                "reasons": reasons,
            }
        )
        t += hop_sec

    moments.sort(key=lambda m: m["overall"], reverse=True)
    top_n = 12
    return {"moments": moments[:top_n]}


def main() -> None:
    from faster_whisper import WhisperModel

    parser = argparse.ArgumentParser(
        description="Closed-caption video: extract audio with ffmpeg, transcribe with faster-whisper, output SRT/TXT/VTT."
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the input video file",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Base path for caption files (default: same as video, e.g. video.srt)",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
        help="faster-whisper model size (default: base)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g. en, es). Auto-detect if not set.",
    )
    parser.add_argument(
        "--vtt",
        action="store_true",
        help="Also write a .vtt caption file",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the extracted audio file after captioning",
    )
    parser.add_argument(
        "--max-chars-per-line",
        type=int,
        default=20,
        help="Maximum characters per subtitle line (default: 20, use 0 to disable wrapping)",
    )
    parser.add_argument(
        "--max-lines-per-cue",
        type=int,
        default=2,
        help="Maximum wrapped lines per subtitle cue (default: 2)",
    )
    parser.add_argument(
        "--rank-model",
        default="",
        help="(Deprecated, no longer used. Ranking is done via the GUI Rank Moments tab.)",
    )
    parser.add_argument(
        "--rank-host",
        default="",
        help="(Deprecated, no longer used. Ranking is done via the GUI Rank Moments tab.)",
    )
    args = parser.parse_args()

    video_path = args.video.resolve()
    if not video_path.is_file():
        print(f"Error: not a file: {video_path}", file=sys.stderr)
        sys.exit(1)

    out_base = args.output.resolve() if args.output else video_path.with_suffix("")
    out_dir = out_base.parent
    prefix = out_base.name  # All assets: {prefix}-{asset}.{ext} for S3-friendly archiving
    srt_path = out_dir / artifact_filename("subtitles_srt", prefix=prefix)
    txt_path = out_dir / artifact_filename("transcript", prefix=prefix)
    words_path = out_dir / artifact_filename("words", prefix=prefix)
    energy_path = out_dir / artifact_filename("energy", prefix=prefix)
    cadence_path = out_dir / artifact_filename("cadence", prefix=prefix)
    moments_path = out_dir / artifact_filename("moments", prefix=prefix)
    stored_video_name = video_path.name

    print("Loading faster-whisper model...")
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type="int8" if args.device == "cpu" else "float16",
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = Path(f.name)

    try:
        print("Extracting audio with ffmpeg...")
        sample_rate = 16000
        video_to_audio(video_path, audio_path, sample_rate=sample_rate)

        print("Transcribing...")
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=args.language,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        )
        total_duration = getattr(info, "duration", 0) or 0
        segments = []
        for seg in segments_gen:
            segments.append(seg)
            if total_duration > 0:
                pct = min(100, 100 * seg.end / total_duration)
                print(
                    f"\rTranscribing... {pct:.0f}% ({format_progress_time(seg.end)} / {format_progress_time(total_duration)})  ",
                    end="",
                    flush=True,
                )
            else:
                print(f"\rTranscribing... {len(segments)} segments  ", end="", flush=True)
        print()
        if info.language_probability:
            print(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")

        cues = segments_to_cues(
            segments,
            max_chars_per_line=args.max_chars_per_line,
            max_lines_per_cue=args.max_lines_per_cue,
        )
        srt_text = cues_to_srt(cues)
        srt_path.write_text(srt_text, encoding="utf-8")
        print(f"Wrote: {srt_path}")
        txt_path.write_text(segments_to_txt(segments), encoding="utf-8")
        print(f"Wrote: {txt_path}")

        # Delivery-aware outputs for reel candidate analysis.
        words_payload = words_payload_from_segments(
            video_path=video_path,
            duration_sec=float(getattr(info, "duration", 0.0) or 0.0),
            sample_rate=sample_rate,
            segments=segments,
            stored_media_name=stored_video_name,
        )
        words_path.write_text(json.dumps(words_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {words_path}")

        energy_payload = compute_energy_map(audio_path, window_sec=0.1, hop_sec=0.1)
        energy_path.write_text(json.dumps(energy_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {energy_path}")

        cadence_payload = build_cadence_payload(words_payload, pause_threshold_sec=0.3)
        cadence_path.write_text(json.dumps(cadence_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {cadence_path}")

        moments_payload = build_moments_payload(words_payload, energy_payload, cadence_payload)
        moments_path.write_text(json.dumps(moments_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {moments_path}")

        if args.vtt:
            vtt_path = out_dir / artifact_filename("subtitles_vtt", prefix=prefix)
            vtt_path.write_text(cues_to_vtt(cues), encoding="utf-8")
            print(f"Wrote: {vtt_path}")

        session_updates = {
            "prefix": prefix,
            "source_video_name": stored_video_name,
            "source_video_original_name": video_path.name,
            "artifacts": {
                "subtitles_srt": srt_path.name,
                "transcript": txt_path.name,
                "words": words_path.name,
                "energy": energy_path.name,
                "cadence": cadence_path.name,
                "moments": moments_path.name,
            },
            "steps": {
                "caption": {
                    "status": "saved",
                    "generated_at": now_iso(),
                    "model": args.model,
                    "device": args.device,
                    "language": args.language or "",
                    "write_vtt": bool(args.vtt),
                    "keep_audio": bool(args.keep_audio),
                }
            },
        }
        if args.vtt:
            session_updates["artifacts"]["subtitles_vtt"] = vtt_path.name
        upsert_session(out_dir, session_updates)
        print(f"Wrote: {out_dir / 'session.json'}")

    finally:
        if not args.keep_audio:
            audio_path.unlink(missing_ok=True)
        else:
            keep_path = out_dir / artifact_filename("audio", prefix=prefix)
            if audio_path != keep_path:
                if keep_path.exists():
                    keep_path.unlink()
                audio_path.rename(keep_path)
            print(f"Kept audio: {keep_path}")
            upsert_session(
                out_dir,
                {
                    "artifacts": {"audio": keep_path.name},
                    "steps": {
                        "caption": {
                            "status": "saved",
                            "generated_at": now_iso(),
                            "model": args.model,
                            "device": args.device,
                            "language": args.language or "",
                            "write_vtt": bool(args.vtt),
                            "keep_audio": True,
                        }
                    },
                },
            )


if __name__ == "__main__":
    main()
