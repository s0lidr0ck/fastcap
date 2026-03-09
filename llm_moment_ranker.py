#!/usr/bin/env python3
"""Context-first ranking for sermon reel candidates."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib import error, request

from session_artifacts import artifact_path, load_session

try:
    import boto3
except Exception:  # noqa: BLE001
    boto3 = None


REQUIRED_ANALYSIS_FILES = ("words.json", "energy.json", "cadence.json")
SCORE_PAD = 15
SCORE_RAW_MAX = 100
SCORE_PADDED_MAX = SCORE_RAW_MAX + SCORE_PAD
PREFERENCE_PROFILE_PATH = Path.home() / ".fastcap" / "preference_profile.json"
PREFERENCE_CLIP_TYPES = ("Teaching", "Conviction", "Declaration", "Encouragement")


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if value and value[0] in ("'", '"') and value[-1:] == value[0]:
        value = value[1:-1]
    return key, value


def _load_project_env() -> None:
    module_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    dotenv_paths = [module_dir / ".env"]
    if cwd != module_dir:
        dotenv_paths.append(cwd / ".env")

    for dotenv_path in dotenv_paths:
        if not dotenv_path.is_file():
            continue
        try:
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                parsed = _parse_dotenv_line(line)
                if not parsed:
                    continue
                key, value = parsed
                os.environ.setdefault(key, value)
        except OSError:
            continue

    if "AWS_REGION" not in os.environ and "BEDROCK_REGION" in os.environ:
        os.environ["AWS_REGION"] = os.environ["BEDROCK_REGION"]


_load_project_env()


class RankerError(RuntimeError):
    """Raised when ranking cannot complete."""


@dataclass
class CandidateMoment:
    candidate_id: int
    start: float
    end: float
    duration_sec: float
    hook: str
    cadence_score: int
    energy_score: int
    contrast_score: int
    hook_score: int
    hook_score_source: str
    hook_confidence: str
    hook_evidence: list[str]
    overall: int
    reasons: list[str]
    markers: list[str]
    text_excerpt: str


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RankerError(f"Missing required file: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise RankerError(f"Invalid JSON in {path.name}: {exc}") from exc


def _parse_timestamp_to_seconds(ts: str) -> float:
    try:
        parts = ts.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError as exc:
        raise RankerError(f"Invalid timestamp format: {ts}") from exc


def _format_seconds_to_timestamp(total_seconds: float) -> str:
    if total_seconds < 0:
        total_seconds = 0.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds - (hours * 3600 + minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def _normalize_0_100(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (value - lo) / (hi - lo)))


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _pad_score(value: int) -> int:
    return int(value) + SCORE_PAD


def _clip_type_from_markers(markers: list[str]) -> str:
    marker_set = set(markers)
    if "rising_intensity" in marker_set or "energy_lift" in marker_set:
        return "Conviction"
    if "stacked_statements" in marker_set or "repetition" in marker_set:
        return "Declaration"
    if "metaphor" in marker_set:
        return "Teaching"
    return "Encouragement"


def _scroll_strength(score: int) -> str:
    if score >= 90:
        return "High"
    if score >= 80:
        return "Medium"
    return "Low"


def _best_platform_fit(duration_sec: float, score: int) -> str:
    if duration_sec <= 35 and score >= 85:
        return "TikTok"
    if duration_sec >= 50:
        return "Shorts"
    return "Reels"


def _window_text(words: list[dict], start: float, end: float, max_words: int = 210) -> str:
    selected = []
    for item in words:
        w_start = float(item.get("s", 0.0))
        w_end = float(item.get("e", w_start))
        if w_end >= start and w_start <= end:
            token = str(item.get("w", "")).strip()
            if token:
                selected.append(token)
            if len(selected) >= max_words:
                break
    return " ".join(selected)


def _collect_markers(cadence_phrases: list[dict], start: float, end: float) -> list[str]:
    markers = set()
    for phrase in cadence_phrases:
        p_start = float(phrase.get("start", 0.0))
        p_end = float(phrase.get("end", p_start))
        if p_end > start and p_start < end:
            markers.update(phrase.get("markers", []))
    return sorted(markers)


def _analysis_path(asset_dir: Path, base_name: str) -> Path:
    """Resolve path to an analysis file; prefers session/prefixed artifacts, then legacy."""
    session = load_session(asset_dir)
    key_by_name = {
        "words.json": "words",
        "energy.json": "energy",
        "cadence.json": "cadence",
        "moments.json": "moments",
        "ranked_moments.json": "ranked_moments",
        "transcript.txt": "transcript",
    }
    key = key_by_name.get(base_name)
    if key:
        return artifact_path(asset_dir, key, session=session)
    prefixed = asset_dir / f"{asset_dir.name}-{base_name}"
    if prefixed.is_file():
        return prefixed
    return asset_dir / base_name


def load_analysis_bundle(asset_dir: Path) -> dict:
    words_path = _analysis_path(asset_dir, "words.json")
    energy_path = _analysis_path(asset_dir, "energy.json")
    cadence_path = _analysis_path(asset_dir, "cadence.json")
    missing = [
        name
        for name, path in [
            ("words.json", words_path),
            ("energy.json", energy_path),
            ("cadence.json", cadence_path),
        ]
        if not path.is_file()
    ]
    if missing:
        missing_display = ", ".join(missing)
        raise RankerError(f"Missing required analysis files in {asset_dir}: {missing_display}")

    words_payload = _read_json(words_path)
    energy_payload = _read_json(energy_path)
    cadence_payload = _read_json(cadence_path)
    return {
        "words": words_payload,
        "energy": energy_payload,
        "cadence": cadence_payload,
    }


def _phase_for_time(time_s: float, duration: float) -> str:
    if duration <= 0:
        return "mid"
    one_third = duration / 3.0
    if time_s < one_third:
        return "early"
    if time_s < one_third * 2:
        return "mid"
    return "late"


def _sentence_like_chunks(words: list[dict], min_words: int = 8) -> list[dict]:
    chunks: list[dict] = []
    current: list[dict] = []
    for i, w in enumerate(words):
        current.append(w)
        token = str(w.get("w", "")).strip()
        cur_end = float(w.get("e", w.get("s", 0.0)))
        next_start = cur_end
        if i + 1 < len(words):
            next_start = float(words[i + 1].get("s", cur_end))
        gap = max(0.0, next_start - cur_end)
        hard_stop = token.endswith((".", "!", "?"))
        pause_stop = gap >= 0.55
        if (hard_stop or pause_stop) and len(current) >= min_words:
            start = float(current[0].get("s", 0.0))
            end = float(current[-1].get("e", start))
            text = " ".join(str(x.get("w", "")).strip() for x in current).strip()
            if text:
                chunks.append({"start": start, "end": end, "text": text})
            current = []
    if current:
        start = float(current[0].get("s", 0.0))
        end = float(current[-1].get("e", start))
        text = " ".join(str(x.get("w", "")).strip() for x in current).strip()
        if text:
            chunks.append({"start": start, "end": end, "text": text})
    return chunks


def _build_transcript_blocks(words: list[dict], target_words: int = 120) -> list[dict]:
    chunks = _sentence_like_chunks(words)
    if not chunks:
        return []
    blocks = []
    cur_chunks = []
    cur_words = 0
    for ch in chunks:
        c_words = len(ch["text"].split())
        if cur_chunks and cur_words + c_words > target_words:
            start = cur_chunks[0]["start"]
            end = cur_chunks[-1]["end"]
            text = " ".join(c["text"] for c in cur_chunks)
            blocks.append({"start": start, "end": end, "text": text})
            cur_chunks = [ch]
            cur_words = c_words
        else:
            cur_chunks.append(ch)
            cur_words += c_words
    if cur_chunks:
        start = cur_chunks[0]["start"]
        end = cur_chunks[-1]["end"]
        text = " ".join(c["text"] for c in cur_chunks)
        blocks.append({"start": start, "end": end, "text": text})
    return blocks


def _fallback_context_candidates(words: list[dict], candidate_target: int) -> list[dict]:
    chunks = _sentence_like_chunks(words, min_words=6)
    if not chunks:
        return []
    candidates = []
    i = 0
    while i < len(chunks):
        start = chunks[i]["start"]
        end = chunks[i]["end"]
        j = i
        while j + 1 < len(chunks) and (end - start) < 36.0:
            j += 1
            end = chunks[j]["end"]
        while j + 1 < len(chunks) and (chunks[j + 1]["end"] - start) <= 60.0:
            j += 1
            end = chunks[j]["end"]
        if (end - start) >= 30.0:
            text = " ".join(chunks[k]["text"] for k in range(i, j + 1))
            hook = " ".join(text.split()[:10]).strip()
            candidates.append({"start": start, "end": end, "hook": hook})
        i += max(1, (j - i + 1) // 2)
    if not candidates:
        return []
    step = max(1, math.ceil(len(candidates) / max(1, candidate_target)))
    return candidates[::step][:candidate_target]


def _clip_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    denom = max(1e-6, min(a_end - a_start, b_end - b_start))
    return overlap / denom


def _dedupe_overlaps(candidates: list[dict], max_overlap: float = 0.5) -> list[dict]:
    kept: list[dict] = []
    for c in sorted(candidates, key=lambda x: x["start"]):
        if any(_clip_overlap_ratio(c["start"], c["end"], k["start"], k["end"]) > max_overlap for k in kept):
            continue
        kept.append(c)
    return kept


def _select_phase_spread(candidates: list[dict], duration: float, target: int) -> list[dict]:
    if not candidates:
        return []
    by_phase = {"early": [], "mid": [], "late": []}
    for c in candidates:
        by_phase[_phase_for_time(c["start"], duration)].append(c)

    selected = []
    per_phase = max(1, target // 3)
    for phase in ("early", "mid", "late"):
        selected.extend(by_phase[phase][:per_phase])
    if len(selected) < target:
        leftovers = [c for c in candidates if c not in selected]
        selected.extend(leftovers[: max(0, target - len(selected))])
    return selected[:target]


def _extract_candidate_list(raw: dict, duration: float) -> list[dict]:
    raw_candidates = raw.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise RankerError("Pass1 model response missing 'candidates' array.")
    normalized = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        s = item.get("start_time")
        e = item.get("end_time")
        if not isinstance(s, str) or not isinstance(e, str):
            continue
        start = _parse_timestamp_to_seconds(s)
        end = _parse_timestamp_to_seconds(e)
        if end <= start:
            continue
        dur = end - start
        if dur < 30 or dur > 60:
            continue
        normalized.append(
            {
                "start": start,
                "end": end,
                "hook": str(item.get("opening_hook", "")).strip(),
            }
        )
    if not normalized:
        raise RankerError("Pass1 model returned zero valid candidates.")
    normalized = _dedupe_overlaps(normalized, max_overlap=0.6)
    normalized = _select_phase_spread(normalized, duration=duration, target=min(30, len(normalized)))
    return normalized


def _build_pass1_prompt(words: list[dict], candidate_target: int, duration: float) -> str:
    blocks = _build_transcript_blocks(words, target_words=115)
    payload = []
    for b in blocks:
        payload.append(
            {
                "start_time": _format_seconds_to_timestamp(float(b["start"])),
                "end_time": _format_seconds_to_timestamp(float(b["end"])),
                "phase": _phase_for_time(float(b["start"]), duration),
                "text": b["text"],
            }
        )
    return (
        "You are selecting contextually coherent sermon clips for short-form video.\n"
        "Task: choose 20-30 candidates from transcript blocks.\n"
        "Each candidate must be 30-60 seconds, coherent standalone, and have a strong opening line.\n"
        "Balance across early/mid/late sermon phases.\n"
        "Output JSON only with this schema:\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        '      "start_time": "HH:MM:SS.mmm",\n'
        '      "end_time": "HH:MM:SS.mmm",\n'
        '      "opening_hook": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Target candidate count: {candidate_target}\n"
        f"Transcript blocks:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _build_context_candidates(
    words: list[dict],
    model: str,
    host: str,
    candidate_target: int,
    duration: float,
    logger: Callable[[str], None] | None = None,
) -> tuple[list[dict], bool, str | None]:
    prompt = _build_pass1_prompt(words, candidate_target=candidate_target, duration=duration)
    parse_error = None
    used_fallback = False
    on_chunk, flush_stream = _make_stream_logger(logger, "Pass1 stream:")
    try:
        response = call_llm_generate(model=model, prompt=prompt, host=host, on_chunk=on_chunk)
        flush_stream()
        parsed = _extract_json_object(response)
        candidates = _extract_candidate_list(parsed, duration=duration)
        if len(candidates) > candidate_target:
            candidates = candidates[:candidate_target]
    except Exception as exc:  # noqa: BLE001
        flush_stream()
        used_fallback = True
        parse_error = str(exc)
        candidates = _fallback_context_candidates(words, candidate_target=candidate_target)
    if not candidates:
        raise RankerError("Could not build context candidates from transcript.")
    for idx, c in enumerate(candidates, start=1):
        c["candidate_id"] = idx
    return candidates, used_fallback, parse_error


def _score_candidates_from_features(bundle: dict, candidates: list[dict]) -> list[CandidateMoment]:
    words = list(bundle["words"].get("words", []) or [])
    energy_frames = list(bundle["energy"].get("frames", []) or [])
    cadence_phrases = list(bundle["cadence"].get("phrases", []) or [])
    rms_values = [float(f.get("rms_db", -60.0)) for f in energy_frames] or [-60.0]
    rms_global_min = min(rms_values)
    rms_global_max = max(rms_values)

    scored: list[CandidateMoment] = []
    for cand in candidates:
        start = float(cand["start"])
        end = float(cand["end"])
        duration = max(0.001, end - start)
        if duration < 30.0 or duration > 60.0:
            continue
        excerpt = _window_text(words, start, end)
        if not excerpt:
            continue
        win_phrases = [p for p in cadence_phrases if float(p.get("end", 0.0)) > start and float(p.get("start", 0.0)) < end]
        if win_phrases:
            cadence_score = sum(float(p.get("cadence_score", 0.0)) for p in win_phrases) / len(win_phrases)
        else:
            cadence_score = 0.0
        markers = _collect_markers(cadence_phrases, start, end)
        win_frames = [f for f in energy_frames if float(f.get("t", 0.0)) >= start and float(f.get("t", 0.0)) <= end]
        if win_frames:
            rms_win = [float(f.get("rms_db", -60.0)) for f in win_frames]
            energy_score = _normalize_0_100(sum(rms_win) / len(rms_win), rms_global_min, rms_global_max)
            contrast_score = _normalize_0_100(max(rms_win) - min(rms_win), 2.0, 20.0)
        else:
            energy_score = 0.0
            contrast_score = 0.0
        win_words = [w for w in words if float(w.get("e", 0.0)) >= start and float(w.get("s", 0.0)) <= end]
        hook_punct_hits = sum(1 for w in win_words[:35] if str(w.get("w", "")).strip().endswith(("!", "?")))
        hook_density = _normalize_0_100(hook_punct_hits, 0, 4)
        hook_score = int(round(hook_density))
        reasons = []
        if energy_score >= 65:
            reasons.append("energy_lift")
        if contrast_score >= 60:
            reasons.append("dynamic_contrast")
        if any(
            (float(b.get("s", 0.0)) - float(a.get("e", 0.0))) >= 0.35
            for a, b in zip(win_words, win_words[1:])
        ):
            reasons.append("clean_pause_cut")
        reasons = sorted(set(reasons))
        overall = (
            cadence_score * 0.30
            + energy_score * 0.25
            + contrast_score * 0.15
            + hook_score * 0.30
        )
        hook = str(cand.get("hook", "")).strip() or " ".join(excerpt.split()[:10]).strip()
        scored.append(
            CandidateMoment(
                candidate_id=int(cand["candidate_id"]),
                start=round(start, 3),
                end=round(end, 3),
                duration_sec=round(duration, 3),
                hook=hook,
                cadence_score=int(round(cadence_score)),
                energy_score=int(round(energy_score)),
                contrast_score=int(round(contrast_score)),
                hook_score=_clamp_int(hook_score, 0, SCORE_RAW_MAX),
                hook_score_source="deterministic_fallback",
                hook_confidence="low",
                hook_evidence=["Opening punctuation density fallback"],
                overall=int(round(max(0.0, min(100.0, overall)))),
                reasons=reasons,
                markers=markers,
                text_excerpt=excerpt,
            )
        )
    if not scored:
        raise RankerError("No candidate features could be scored from transcript/audio data.")
    return scored


def _build_hook_score_prompt(candidates: list[CandidateMoment]) -> str:
    payload = []
    for c in candidates:
        opening_excerpt = " ".join(str(c.text_excerpt or "").split()[:18]).strip()
        payload.append(
            {
                "candidate_id": int(c.candidate_id),
                "opening_hook": str(c.hook or "").strip(),
                "opening_excerpt": opening_excerpt,
                "start_sec": round(float(c.start), 3),
                "end_sec": round(float(c.end), 3),
            }
        )
    return (
        "You are scoring opening hooks for short-form sermon clips.\n"
        "Return JSON only. No markdown.\n"
        "Score only scroll-stopping hook quality of the opening line(s), not theology.\n"
        "Output schema:\n"
        "{\n"
        '  "hook_scores": [\n'
        "    {\n"
        '      "candidate_id": 1,\n'
        '      "llm_hook_score": 0,\n'
        '      "confidence": "low|medium|high",\n'
        '      "evidence": ["string"],\n'
        '      "reason_short": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Return every candidate_id in the input exactly once.\n"
        "- llm_hook_score must be an integer between 0 and 100.\n"
        "- confidence must be low, medium, or high.\n"
        "- evidence list must have 1 to 3 short items.\n"
        "- reason_short should be one sentence <= 180 chars.\n\n"
        f"Candidates:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _score_hooks_with_llm(
    candidates: list[CandidateMoment],
    model: str,
    host: str,
    logger: Callable[[str], None] | None = None,
) -> dict[int, dict]:
    if not candidates:
        return {}
    prompt = _build_hook_score_prompt(candidates)
    try:
        raw = call_llm_generate(model=model, prompt=prompt, host=host)
        parsed = _extract_json_object(raw)
        rows = parsed.get("hook_scores", [])
        if not isinstance(rows, list):
            raise RankerError("Hook score response missing 'hook_scores' list.")
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger(f"Pass1.5 fallback: hook scoring unavailable ({exc})")
        return {}

    out: dict[int, dict] = {}
    seen: set[int] = set()
    expected_ids = {int(c.candidate_id) for c in candidates}
    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = int(row.get("candidate_id", 0) or 0)
        if cid <= 0 or cid not in expected_ids or cid in seen:
            continue
        seen.add(cid)
        score = _clamp_int(int(row.get("llm_hook_score", 0)), 0, SCORE_RAW_MAX)
        confidence = str(row.get("confidence", "medium")).strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        evidence = row.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        short_evidence = [str(x).strip() for x in evidence if str(x).strip()][:3]
        if not short_evidence:
            short_evidence = ["LLM hook score"]
        out[cid] = {
            "score": int(score),
            "source": "llm",
            "confidence": confidence,
            "evidence": short_evidence,
            "reason_short": str(row.get("reason_short", "")).strip(),
        }

    missing = sorted(expected_ids - set(out))
    if missing and logger:
        logger(f"Pass1.5 partial fallback: missing LLM hook scores for candidate_ids {missing[:8]}")
    return out


def _apply_hook_scores(candidates: list[CandidateMoment], hook_results: dict[int, dict]) -> list[CandidateMoment]:
    for c in candidates:
        row = hook_results.get(int(c.candidate_id), {})
        if isinstance(row, dict) and row:
            c.hook_score = _clamp_int(int(row.get("score", c.hook_score)), 0, SCORE_RAW_MAX)
            c.hook_score_source = str(row.get("source", "llm")).strip() or "llm"
            c.hook_confidence = str(row.get("confidence", "medium")).strip().lower() or "medium"
            evidence = row.get("evidence", [])
            if isinstance(evidence, list):
                c.hook_evidence = [str(x).strip() for x in evidence if str(x).strip()][:3]
        c.overall = _clamp_int(
            int(
                round(
                    c.cadence_score * 0.30
                    + c.energy_score * 0.25
                    + c.contrast_score * 0.15
                    + c.hook_score * 0.30
                )
            ),
            0,
            SCORE_RAW_MAX,
        )
    return candidates


def build_prompt(candidates: list[CandidateMoment], output_count: int) -> str:
    total_duration = max((c.end for c in candidates), default=0.0)
    phase_a = total_duration / 3.0
    phase_b = 2.0 * total_duration / 3.0

    def _phase_for_start(start: float) -> str:
        if start < phase_a:
            return "early"
        if start < phase_b:
            return "mid"
        return "late"

    candidate_payload = []
    for c in candidates:
        candidate_payload.append(
            {
                "candidate_id": c.candidate_id,
                "start_sec": round(c.start, 3),
                "end_sec": round(c.end, 3),
                "duration_sec": round(c.duration_sec, 3),
                "hook": c.hook,
                "scores": {
                    "cadence": c.cadence_score,
                    "energy": c.energy_score,
                    "contrast": c.contrast_score,
                    "hook": c.hook_score,
                    "overall": c.overall,
                },
                "hook_score_source": c.hook_score_source,
                "markers": c.markers,
                "reasons": c.reasons,
                "phase": _phase_for_start(c.start),
                "text_excerpt": c.text_excerpt,
            }
        )

    return (
        "You are a short-form sermon video editor. Rank clip candidates by reel potential.\n"
        "Focus on hook strength, cadence clarity, standalone clarity, conviction impact, and platform performance.\n"
        "Prioritize spoken rhythm and delivery over abstract theology.\n"
        f"Return exactly {output_count} clips when possible.\n\n"
        "Output MUST be valid JSON only, no markdown, no commentary.\n"
        "Use this exact schema:\n"
        "{\n"
        '  "clips": [\n'
        "    {\n"
        '      "candidate_id": 1,\n'
        '      "start_time": "HH:MM:SS.mmm",\n'
        '      "end_time": "HH:MM:SS.mmm",\n'
        '      "opening_hook": "string",\n'
        '      "clip_type": "Teaching|Conviction|Declaration|Encouragement",\n'
        '      "cadence_marker": "Punch Phrase|Rising Stack|Repetition|Metaphor|Pause Punch",\n'
        '      "editorial_scores": {\n'
        '        "editor": 0,\n'
        '        "hook": 0,\n'
        '        "cadence": 0,\n'
        '        "standalone": 0,\n'
        '        "emotion": 0\n'
        "      },\n"
        '      "editor_score": 0,\n'
        '      "editor_reason": "string",\n'
        '      "scroll_stopping_strength": "Low|Medium|High",\n'
        '      "best_platform_fit": "Reels|TikTok|Shorts"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Keep clips in chronological order.\n"
        "- Keep 30-60 second windows.\n"
        "- editor_score must be between 0 and 100.\n"
        "- editorial_scores values must each be between 0 and 100.\n"
        "- Use candidate_id values from input.\n\n"
        "Reject criteria (do not pick):\n"
        "- Opening first 5 seconds has weak/no hook.\n"
        "- Excessive setup with low standalone clarity.\n"
        "- Flat cadence with no strong delivery marker.\n\n"
        "Diversity constraints:\n"
        "- Balance selections across early, mid, and late phases.\n"
        "- Avoid clustering all clips in one phase unless candidates are clearly weak elsewhere.\n\n"
        "Reasoning constraints:\n"
        "- editor_reason must mention at least two concrete signals from persisted metrics/signals.\n"
        "- Only reference these metrics by name if used: editor, hook, cadence, standalone, emotion, energy, contrast, overall_candidate.\n"
        "- Prefer clips where short declarative lines stack back-to-back.\n\n"
        f"Candidate input:\n{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}"
    )


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)
    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        raise RankerError("Model response did not contain a JSON object.")
    return json.loads(match.group(0))


def call_ollama_generate(
    model: str,
    prompt: str,
    host: str = "http://127.0.0.1:11434",
    timeout_sec: int = 180,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.15, "top_p": 0.9},
    }
    req = request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            parts: list[str] = []
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                item = json.loads(line)
                delta = str(item.get("response", ""))
                if delta:
                    parts.append(delta)
                    if on_chunk is not None:
                        on_chunk(delta)
                if item.get("done", False):
                    break
            parsed = {"response": "".join(parts)}
    except error.URLError as exc:
        raise RankerError(
            f"Could not reach Ollama at {host}. Start it with 'ollama serve'. Details: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RankerError(f"Ollama returned non-JSON response: {exc}") from exc

    response = parsed.get("response")
    if not response:
        raise RankerError("Ollama response did not include generated text.")
    return str(response)


def _resolve_bedrock_region(host: str | None) -> str:
    if host:
        host_value = host.strip()
        if host_value.startswith("bedrock://"):
            region = host_value.replace("bedrock://", "", 1).strip()
            if region:
                return region
        if re.fullmatch(r"[a-z]{2}-[a-z]+-\d", host_value):
            return host_value
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


def call_bedrock_generate(
    model: str,
    prompt: str,
    host: str = "",
    timeout_sec: int = 180,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    if boto3 is None:
        raise RankerError(
            "boto3 is required for Bedrock models. Install it with: pip install boto3"
        )
    model_id = model.strip()
    if model_id.startswith("bedrock/"):
        model_id = model_id.replace("bedrock/", "", 1).strip()
    elif model_id.startswith("bedrock:"):
        model_id = model_id.replace("bedrock:", "", 1).strip()
    if not model_id:
        raise RankerError("Bedrock model id is missing. Use format: bedrock/<model-id>")
    region = _resolve_bedrock_region(host)
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=boto3.session.Config(read_timeout=timeout_sec, connect_timeout=15),
    )

    def _converse(inference_config: dict) -> dict:
        return client.converse_stream(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=inference_config,
        )

    try:
        try:
            response = _converse({"temperature": 0.15, "topP": 0.9, "maxTokens": 4096})
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "temperature" in msg and "top_p" in msg and "both" in msg:
                response = _converse({"temperature": 0.15, "maxTokens": 4096})
            else:
                raise
        parts: list[str] = []
        for event in response.get("stream", []):
            content_delta = event.get("contentBlockDelta", {})
            delta = content_delta.get("delta", {}) if isinstance(content_delta, dict) else {}
            text = str(delta.get("text", ""))
            if text:
                parts.append(text)
                if on_chunk is not None:
                    on_chunk(text)
        out = "".join(parts).strip()
    except Exception as exc:  # noqa: BLE001
        raise RankerError(
            f"Bedrock request failed for model '{model_id}' in region '{region}'. "
            "Check AWS credentials/permissions (bedrock:InvokeModelWithResponseStream) "
            f"and model access. Details: {exc}"
        ) from exc
    if not out:
        raise RankerError("Bedrock response did not include generated text.")
    return out


def _looks_like_bedrock_model_id(model: str) -> bool:
    value = (model or "").strip()
    if not value:
        return False
    if value.startswith("arn:aws:bedrock:"):
        return True
    return bool(
        re.match(
            (
                r"^(?:[a-z]{2}\.)?"
                r"(anthropic|amazon|ai21|cohere|meta|mistral|deepseek|google|openai|"
                r"qwen|stability|writer|nvidia|minimax|moonshotai|moonshot|zai|twelvelabs)\."
            ),
            value,
        )
    )


def call_llm_generate(
    model: str,
    prompt: str,
    host: str = "http://127.0.0.1:11434",
    timeout_sec: int = 180,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    if (
        model.startswith("bedrock/")
        or model.startswith("bedrock:")
        or _looks_like_bedrock_model_id(model)
    ):
        return call_bedrock_generate(
            model=model,
            prompt=prompt,
            host=host,
            timeout_sec=timeout_sec,
            on_chunk=on_chunk,
        )
    return call_ollama_generate(
        model=model,
        prompt=prompt,
        host=host,
        timeout_sec=timeout_sec,
        on_chunk=on_chunk,
    )


def _make_stream_logger(
    logger: Callable[[str], None] | None,
    prefix: str,
    flush_threshold: int = 140,
) -> tuple[Callable[[str], None] | None, Callable[[], None]]:
    if logger is None:
        return None, (lambda: None)

    buf: list[str] = []
    size = 0

    def _flush() -> None:
        nonlocal size
        if not buf:
            return
        text = "".join(buf).replace("\r", " ").strip()
        if text:
            logger(f"{prefix} {text}")
        buf.clear()
        size = 0

    def _on_chunk(chunk: str) -> None:
        nonlocal size
        if not chunk:
            return
        buf.append(chunk)
        size += len(chunk)
        if size >= flush_threshold or "\n" in chunk:
            _flush()

    return _on_chunk, _flush


def _normalize_clip(
    raw_clip: dict,
    candidates_by_id: dict[int, CandidateMoment],
) -> dict:
    candidate_id = int(raw_clip.get("candidate_id", raw_clip.get("clip_number", 0)) or 0)
    candidate = candidates_by_id.get(candidate_id)
    if candidate is None:
        raise RankerError(f"Model returned unknown candidate_id: {candidate_id}")

    start_time = raw_clip.get("start_time")
    end_time = raw_clip.get("end_time")
    if isinstance(start_time, str):
        start_sec = _parse_timestamp_to_seconds(start_time)
    else:
        start_sec = candidate.start
    if isinstance(end_time, str):
        end_sec = _parse_timestamp_to_seconds(end_time)
    else:
        end_sec = candidate.end
    if end_sec <= start_sec:
        start_sec = candidate.start
        end_sec = candidate.end

    duration_sec = round(end_sec - start_sec, 3)
    if duration_sec < 30.0 or duration_sec > 60.0:
        start_sec = candidate.start
        end_sec = candidate.end
        duration_sec = round(candidate.duration_sec, 3)

    raw_editor_score = raw_clip.get("editor_score", raw_clip.get("score", candidate.overall))
    score = _clamp_int(int(raw_editor_score), 0, SCORE_RAW_MAX)
    raw_editorial = raw_clip.get("editorial_scores", {}) if isinstance(raw_clip.get("editorial_scores", {}), dict) else {}
    raw_sub_scores = raw_clip.get("sub_scores", {}) if isinstance(raw_clip.get("sub_scores", {}), dict) else {}
    src_scores = raw_editorial if raw_editorial else raw_sub_scores
    score = _clamp_int(int(src_scores.get("editor", score)), 0, SCORE_RAW_MAX)
    padded_score = _pad_score(score)
    editorial_scores = {
        "editor": _pad_score(_clamp_int(int(src_scores.get("editor", score)), 0, SCORE_RAW_MAX)),
        "hook": _pad_score(_clamp_int(int(src_scores.get("hook", score)), 0, SCORE_RAW_MAX)),
        "cadence": _pad_score(
            _clamp_int(int(src_scores.get("cadence", candidate.cadence_score)), 0, SCORE_RAW_MAX)
        ),
        "standalone": _pad_score(_clamp_int(int(src_scores.get("standalone", score)), 0, SCORE_RAW_MAX)),
        "emotion": _pad_score(_clamp_int(int(src_scores.get("emotion", candidate.energy_score)), 0, SCORE_RAW_MAX)),
    }
    markers = candidate.markers + candidate.reasons
    clip_type = str(raw_clip.get("clip_type", "")).strip() or _clip_type_from_markers(markers)
    cadence_marker = str(raw_clip.get("cadence_marker", "")).strip() or (
        "Repetition" if "repetition" in markers else "Rising Stack"
    )
    opening_hook = str(raw_clip.get("opening_hook", "")).strip() or candidate.hook
    editor_reason = str(raw_clip.get("editor_reason", "")).strip()
    if not editor_reason:
        editor_reason = (
            f"Strong delivery profile with cadence {candidate.cadence_score}, "
            f"energy {candidate.energy_score}, and contrast {candidate.contrast_score}."
        )
    scroll_strength = str(raw_clip.get("scroll_stopping_strength", "")).strip() or _scroll_strength(score)
    platform = str(raw_clip.get("best_platform_fit", "")).strip() or _best_platform_fit(duration_sec, score)

    return {
        "candidate_id": candidate_id,
        "start_time": _format_seconds_to_timestamp(start_sec),
        "end_time": _format_seconds_to_timestamp(end_sec),
        "duration_sec": duration_sec,
        "opening_hook": opening_hook,
        "clip_type": clip_type,
        "cadence_marker": cadence_marker,
        "editorial_scores": editorial_scores,
        # Back-compat for existing consumers while migrating to editorial_scores.
        "sub_scores": {
            "hook": editorial_scores["hook"],
            "cadence": editorial_scores["cadence"],
            "standalone": editorial_scores["standalone"],
            "emotion": editorial_scores["emotion"],
        },
        "editor_score": padded_score,
        "editor_reason": editor_reason,
        "scroll_stopping_strength": scroll_strength,
        "best_platform_fit": platform,
        "feature_scores": {
            "overall_candidate": _pad_score(int(candidate.overall)),
            "cadence": _pad_score(int(candidate.cadence_score)),
            "energy": _pad_score(int(candidate.energy_score)),
            "contrast": _pad_score(int(candidate.contrast_score)),
        },
        "feature_signals": {
            "markers": list(candidate.markers),
            "reasons": list(candidate.reasons),
            "hook_score_llm": int(candidate.hook_score),
            "hook_score_source": str(candidate.hook_score_source),
            "hook_confidence": str(candidate.hook_confidence),
            "hook_evidence": list(candidate.hook_evidence),
        },
    }


def fallback_ranked_clips(candidates: list[CandidateMoment], output_count: int) -> list[dict]:
    picked = sorted(
        candidates,
        key=lambda c: (c.overall, c.cadence_score, c.energy_score, c.contrast_score),
        reverse=True,
    )[: max(1, output_count)]
    out = []
    for c in picked:
        raw_score = _clamp_int(c.overall, 0, SCORE_RAW_MAX)
        score = _pad_score(raw_score)
        markers = c.markers + c.reasons
        cadence_marker = "Repetition" if "repetition" in markers else "Rising Stack"
        out.append(
            {
                "candidate_id": c.candidate_id,
                "start_time": _format_seconds_to_timestamp(c.start),
                "end_time": _format_seconds_to_timestamp(c.end),
                "duration_sec": c.duration_sec,
                "opening_hook": c.hook,
                "clip_type": _clip_type_from_markers(markers),
                "cadence_marker": cadence_marker,
                "editorial_scores": {
                    "editor": score,
                    "hook": score,
                    "cadence": _pad_score(_clamp_int(c.cadence_score, 0, SCORE_RAW_MAX)),
                    "standalone": score,
                    "emotion": _pad_score(_clamp_int(c.energy_score, 0, SCORE_RAW_MAX)),
                },
                "sub_scores": {
                    "hook": score,
                    "cadence": _pad_score(_clamp_int(c.cadence_score, 0, SCORE_RAW_MAX)),
                    "standalone": score,
                    "emotion": _pad_score(_clamp_int(c.energy_score, 0, SCORE_RAW_MAX)),
                },
                "editor_score": score,
                "editor_reason": (
                    "Fallback ranking from delivery metadata because model output "
                    "was unavailable or invalid."
                ),
                "scroll_stopping_strength": _scroll_strength(score),
                "best_platform_fit": _best_platform_fit(c.duration_sec, score),
                "feature_scores": {
                    "overall_candidate": _pad_score(int(c.overall)),
                    "cadence": _pad_score(int(c.cadence_score)),
                    "energy": _pad_score(int(c.energy_score)),
                    "contrast": _pad_score(int(c.contrast_score)),
                },
                "feature_signals": {
                    "markers": list(c.markers),
                    "reasons": list(c.reasons),
                    "hook_score_llm": int(c.hook_score),
                    "hook_score_source": str(c.hook_score_source),
                    "hook_confidence": str(c.hook_confidence),
                    "hook_evidence": list(c.hook_evidence),
                },
            }
        )
    return out


def _clip_start_seconds(clip: dict, candidates_by_id: dict[int, CandidateMoment]) -> float:
    candidate = candidates_by_id.get(int(clip.get("candidate_id", 0) or 0))
    if candidate is not None:
        return float(candidate.start)
    return _parse_timestamp_to_seconds(str(clip.get("start_time", "00:00:00.000")))


def _phase_from_time(start_sec: float, total_duration: float) -> str:
    if total_duration <= 0:
        return "mid"
    phase_a = total_duration / 3.0
    phase_b = 2.0 * total_duration / 3.0
    if start_sec < phase_a:
        return "early"
    if start_sec < phase_b:
        return "mid"
    return "late"


def _editorial_scores(clip: dict) -> dict[str, int]:
    editorial = clip.get("editorial_scores", {}) if isinstance(clip.get("editorial_scores", {}), dict) else {}
    sub = clip.get("sub_scores", {}) if isinstance(clip.get("sub_scores", {}), dict) else {}
    out = {
        "editor": _clamp_int(int(editorial.get("editor", clip.get("editor_score", 0))), 0, SCORE_PADDED_MAX),
        "hook": _clamp_int(int(editorial.get("hook", sub.get("hook", 0))), 0, SCORE_PADDED_MAX),
        "cadence": _clamp_int(int(editorial.get("cadence", sub.get("cadence", 0))), 0, SCORE_PADDED_MAX),
        "standalone": _clamp_int(int(editorial.get("standalone", sub.get("standalone", 0))), 0, SCORE_PADDED_MAX),
        "emotion": _clamp_int(int(editorial.get("emotion", sub.get("emotion", 0))), 0, SCORE_PADDED_MAX),
    }
    return out


def _normalize_reasoning_text(clip: dict) -> tuple[str, dict]:
    reason = str(clip.get("editor_reason", "")).strip()
    if not reason:
        return "", {"had_mismatch": False, "severity": "none", "minor_count": 0, "severe_count": 0, "cited_metrics": []}

    editorial = _editorial_scores(clip)
    feature = clip.get("feature_scores", {}) if isinstance(clip.get("feature_scores", {}), dict) else {}
    expected = {
        "editor": int(editorial["editor"]),
        "hook": int(editorial["hook"]),
        "cadence": int(editorial["cadence"]),
        "standalone": int(editorial["standalone"]),
        "emotion": int(editorial["emotion"]),
        "overall_candidate": int(feature.get("overall_candidate", editorial["editor"])),
        "energy": int(feature.get("energy", editorial["emotion"])),
        "contrast": int(feature.get("contrast", editorial["editor"])),
    }
    aliases = {
        "editor": "editor",
        "editor score": "editor",
        "overall": "overall_candidate",
        "overall candidate": "overall_candidate",
        "hook": "hook",
        "cadence": "cadence",
        "standalone": "standalone",
        "emotion": "emotion",
        "energy": "energy",
        "contrast": "contrast",
    }
    metric_re = re.compile(
        r"\b(editor score|editor|overall candidate|overall|hook|cadence|standalone|emotion|energy|contrast)\b"
        r"(?:\s+score)?(?:\s+of|\s*[:=]|\s+is)?\s*(\d{1,3})",
        flags=re.IGNORECASE,
    )

    minor_count = 0
    severe_count = 0
    cited_metrics: set[str] = set()

    def _replace_metric(match: re.Match) -> str:
        nonlocal minor_count, severe_count
        raw_label = str(match.group(1)).strip().lower()
        key = aliases.get(raw_label)
        if not key:
            return match.group(0)
        actual = int(match.group(2))
        want = expected.get(key, actual)
        cited_metrics.add(key)
        if actual == want:
            return match.group(0)
        if abs(actual - want) >= 20:
            severe_count += 1
        else:
            minor_count += 1
        return f"{raw_label} {want}"

    normalized = metric_re.sub(_replace_metric, reason)

    def _overall_paren_repl(match: re.Match) -> str:
        nonlocal minor_count, severe_count
        actual = int(match.group(1))
        want = expected["overall_candidate"]
        if actual == want:
            return match.group(0)
        if abs(actual - want) >= 20:
            severe_count += 1
        else:
            minor_count += 1
        cited_metrics.add("overall_candidate")
        return f"overall candidate score ({want})"

    normalized = re.sub(
        r"overall candidate score\s*\((\d+)\)",
        _overall_paren_repl,
        normalized,
        flags=re.IGNORECASE,
    )
    severity = "none"
    if severe_count > 0:
        severity = "severe"
    elif minor_count > 0:
        severity = "minor"
    return normalized, {
        "had_mismatch": bool(minor_count or severe_count),
        "severity": severity,
        "minor_count": minor_count,
        "severe_count": severe_count,
        "cited_metrics": sorted(cited_metrics),
    }


def _compute_assessment_confidence(clip: dict, reasoning_meta: dict) -> dict:
    editorial = _editorial_scores(clip)
    feature = clip.get("feature_scores", {}) if isinstance(clip.get("feature_scores", {}), dict) else {}
    deltas = [
        abs(int(editorial["editor"]) - int(feature.get("overall_candidate", editorial["editor"]))),
        abs(int(editorial["cadence"]) - int(feature.get("cadence", editorial["cadence"]))),
        abs(int(editorial["emotion"]) - int(feature.get("energy", editorial["emotion"]))),
    ]
    agreement = max(0.0, 100.0 - (sum(float(x) for x in deltas) / max(1.0, float(len(deltas)))))
    penalty = 0.0
    if str(reasoning_meta.get("severity", "none")) == "minor":
        penalty = 8.0
    elif str(reasoning_meta.get("severity", "none")) == "severe":
        penalty = 18.0
    confidence_score = _clamp_int(int(round(max(0.0, min(100.0, agreement - penalty)))), 0, 100)
    if confidence_score >= 80:
        level = "high"
    elif confidence_score >= 60:
        level = "medium"
    else:
        level = "low"
    return {
        "level": level,
        "score": confidence_score,
        "agreement_delta": int(round(sum(float(x) for x in deltas) / max(1.0, float(len(deltas))))),
    }


def _clip_personalization_features(clip: dict) -> dict[str, float]:
    editorial = clip.get("editorial_scores", {}) if isinstance(clip.get("editorial_scores", {}), dict) else {}
    feature_scores = clip.get("feature_scores", {}) if isinstance(clip.get("feature_scores", {}), dict) else {}
    feature_signals = clip.get("feature_signals", {}) if isinstance(clip.get("feature_signals", {}), dict) else {}
    confidence = (
        clip.get("assessment_confidence", {})
        if isinstance(clip.get("assessment_confidence", {}), dict)
        else {}
    )
    clip_type = str(clip.get("clip_type", "")).strip().lower()
    features: dict[str, float] = {
        "editorial_editor": float(editorial.get("editor", clip.get("editor_score", 0)) or 0.0),
        "editorial_hook": float(editorial.get("hook", 0) or 0.0),
        "editorial_cadence": float(editorial.get("cadence", 0) or 0.0),
        "editorial_standalone": float(editorial.get("standalone", 0) or 0.0),
        "editorial_emotion": float(editorial.get("emotion", 0) or 0.0),
        "feature_overall_candidate": float(feature_scores.get("overall_candidate", 0) or 0.0),
        "feature_cadence": float(feature_scores.get("cadence", 0) or 0.0),
        "feature_energy": float(feature_scores.get("energy", 0) or 0.0),
        "feature_contrast": float(feature_scores.get("contrast", 0) or 0.0),
        "hook_score_llm": float(feature_signals.get("hook_score_llm", 0) or 0.0),
        "duration_sec": float(clip.get("duration_sec", 0.0) or 0.0),
        "assessment_confidence_score": float(confidence.get("score", 0) or 0.0),
    }
    for known_type in PREFERENCE_CLIP_TYPES:
        features[f"clip_type_{known_type.lower()}"] = 1.0 if clip_type == known_type.lower() else 0.0
    return features


def _load_preference_profile() -> dict | None:
    if not PREFERENCE_PROFILE_PATH.is_file():
        return None
    try:
        raw = json.loads(PREFERENCE_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw, dict):
        return None
    feature_weights = raw.get("feature_weights", {})
    if not isinstance(feature_weights, dict) or not feature_weights:
        return None
    try:
        version = int(raw.get("version", 0))
        sample_count = int(raw.get("sample_count", 0))
        accuracy_cv = float(raw.get("accuracy_cv", 0.0))
        min_threshold = float(raw.get("min_confidence_threshold", 0.55))
        intercept = float(raw.get("intercept", 0.0))
    except Exception:  # noqa: BLE001
        return None
    if version <= 0 or sample_count < 50 or accuracy_cv < min_threshold:
        return None
    normalized_weights: dict[str, float] = {}
    for key, value in feature_weights.items():
        try:
            normalized_weights[str(key)] = float(value)
        except Exception:  # noqa: BLE001
            continue
    if not normalized_weights:
        return None
    return {
        "version": version,
        "sample_count": sample_count,
        "accuracy_cv": accuracy_cv,
        "min_confidence_threshold": min_threshold,
        "intercept": intercept,
        "feature_weights": normalized_weights,
    }


def _compute_personal_fit_score(clip: dict, profile: dict) -> int:
    weights = profile.get("feature_weights", {}) if isinstance(profile.get("feature_weights", {}), dict) else {}
    intercept = float(profile.get("intercept", 0.0))
    features = _clip_personalization_features(clip)
    z = intercept
    for name, value in features.items():
        z += float(weights.get(name, 0.0)) * float(value)
    z = max(-60.0, min(60.0, z))
    probability = 1.0 / (1.0 + math.exp(-z))
    return _clamp_int(int(round(probability * 100.0)), 0, 100)


def _apply_personalized_scores(clips: list[dict], profile: dict | None) -> bool:
    if not profile:
        for clip in clips:
            clip["personal_fit_applied"] = False
        return False
    version = int(profile.get("version", 0))
    for clip in clips:
        personal_fit = _compute_personal_fit_score(clip, profile)
        clip["personal_fit_score"] = int(personal_fit)
        clip["personal_fit_applied"] = True
        clip["preference_profile_version"] = version
        blended = round(0.70 * float(clip.get("editor_score", 0)) + 0.30 * float(personal_fit))
        clip["final_rank_score"] = _clamp_int(int(blended), 0, SCORE_PADDED_MAX)
    return True


def _finalize_clip_reasoning_and_evidence(clips: list[dict]) -> tuple[list[dict], int, int]:
    severe = 0
    minor = 0
    for clip in clips:
        editorial = _editorial_scores(clip)
        clip["editor_score"] = int(editorial["editor"])
        clip["editorial_scores"] = editorial
        clip["sub_scores"] = {
            "hook": int(editorial["hook"]),
            "cadence": int(editorial["cadence"]),
            "standalone": int(editorial["standalone"]),
            "emotion": int(editorial["emotion"]),
        }
        original_reason = str(clip.get("editor_reason", "")).strip()
        normalized_reason, meta = _normalize_reasoning_text(clip)
        clip["editor_reason_original"] = original_reason
        if normalized_reason:
            clip["editor_reason"] = normalized_reason
        clip["reasoning_consistency"] = {
            "normalized": bool(meta.get("had_mismatch", False)),
            "severity": str(meta.get("severity", "none")),
            "minor_count": int(meta.get("minor_count", 0)),
            "severe_count": int(meta.get("severe_count", 0)),
        }
        clip["assessment_confidence"] = _compute_assessment_confidence(clip, meta)
        signals = clip.get("feature_signals", {}) if isinstance(clip.get("feature_signals", {}), dict) else {}
        clip["reason_evidence"] = {
            "cited_metrics": list(meta.get("cited_metrics", [])),
            "markers": list(signals.get("markers", [])) if isinstance(signals.get("markers", []), list) else [],
            "reasons": list(signals.get("reasons", [])) if isinstance(signals.get("reasons", []), list) else [],
        }
        severe += int(meta.get("severe_count", 0))
        minor += int(meta.get("minor_count", 0))
    return clips, severe, minor


def _add_overlap_and_phase_diagnostics(
    clips: list[dict],
    candidates_by_id: dict[int, CandidateMoment],
) -> dict:
    if not clips:
        return {"counts": {"early": 0, "mid": 0, "late": 0}, "imbalance_flag": False}
    total_duration = max(
        max(
            _parse_timestamp_to_seconds(str(c.get("end_time", "00:00:00.000"))),
            _clip_start_seconds(c, candidates_by_id),
        )
        for c in clips
    )
    counts = {"early": 0, "mid": 0, "late": 0}
    for idx, clip in enumerate(clips):
        start = _parse_timestamp_to_seconds(str(clip.get("start_time", "00:00:00.000")))
        end = _parse_timestamp_to_seconds(str(clip.get("end_time", "00:00:00.000")))
        phase = _phase_from_time(start, total_duration)
        counts[phase] += 1
        prev_overlap = 0.0
        next_overlap = 0.0
        if idx > 0:
            prev = clips[idx - 1]
            prev_overlap = _clip_overlap_ratio(
                start,
                end,
                _parse_timestamp_to_seconds(str(prev.get("start_time", "00:00:00.000"))),
                _parse_timestamp_to_seconds(str(prev.get("end_time", "00:00:00.000"))),
            )
        if idx + 1 < len(clips):
            nxt = clips[idx + 1]
            next_overlap = _clip_overlap_ratio(
                start,
                end,
                _parse_timestamp_to_seconds(str(nxt.get("start_time", "00:00:00.000"))),
                _parse_timestamp_to_seconds(str(nxt.get("end_time", "00:00:00.000"))),
            )
        clip["selection_diagnostics"] = {
            "phase": phase,
            "overlap_prev_ratio": round(prev_overlap, 3),
            "overlap_next_ratio": round(next_overlap, 3),
            "overlap_flag": bool(prev_overlap > 0.12 or next_overlap > 0.12),
        }
    vals = list(counts.values())
    imbalance_flag = (max(vals) - min(vals)) > 1 if vals else False
    return {"counts": counts, "imbalance_flag": bool(imbalance_flag)}


def _enforce_phase_diversity(
    clips: list[dict],
    candidates_by_id: dict[int, CandidateMoment],
    output_count: int,
    score_key: str = "editor_score",
) -> list[dict]:
    if not clips:
        return clips

    max_t = max(_clip_start_seconds(c, candidates_by_id) for c in clips) + 1e-6
    phase_a = max_t / 3.0
    phase_b = 2.0 * max_t / 3.0

    def _phase(c: dict) -> str:
        t = _clip_start_seconds(c, candidates_by_id)
        if t < phase_a:
            return "early"
        if t < phase_b:
            return "mid"
        return "late"

    target_per_phase = 2 if output_count >= 9 else 1
    scored = sorted(clips, key=lambda c: float(c.get(score_key, c.get("editor_score", 0))), reverse=True)
    selected: list[dict] = []
    selected_ids: set[int] = set()

    for phase_name in ("early", "mid", "late"):
        phase_candidates = [c for c in scored if _phase(c) == phase_name]
        take = min(target_per_phase, len(phase_candidates))
        for c in phase_candidates[:take]:
            cid = int(c.get("candidate_id", 0) or 0)
            if cid and cid in selected_ids:
                continue
            selected.append(c)
            if cid:
                selected_ids.add(cid)

    for c in scored:
        if len(selected) >= output_count:
            break
        cid = int(c.get("candidate_id", 0) or 0)
        if cid and cid in selected_ids:
            continue
        selected.append(c)
        if cid:
            selected_ids.add(cid)

    selected = selected[: max(1, output_count)]
    selected.sort(key=lambda c: _clip_start_seconds(c, candidates_by_id))
    return selected


def rank_sermon_moments(
    asset_dir: Path,
    model: str = "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    candidate_limit: int = 20,
    output_count: int = 10,
    host: str = "http://127.0.0.1:11434",
    logger: Callable[[str], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    asset_dir = asset_dir.resolve()
    if logger:
        logger(f"Loading analysis files from: {asset_dir}")
    if progress:
        progress(1, 7, "Loading analysis files")

    bundle = load_analysis_bundle(asset_dir)
    words = list(bundle["words"].get("words", []) or [])
    if not words:
        raise RankerError("No words found in words.json.")
    duration = float(bundle["words"].get("media", {}).get("duration_sec", 0.0) or 0.0)
    if duration <= 0:
        duration = float(words[-1].get("e", 0.0) or 0.0)

    if progress:
        progress(2, 7, "Pass 1: Building context candidates")
    pass1_target = max(20, min(30, candidate_limit))
    pass1_candidates, pass1_used_fallback, pass1_parse_error = _build_context_candidates(
        words=words,
        model=model,
        host=host,
        candidate_target=pass1_target,
        duration=duration,
        logger=logger,
    )
    if logger:
        logger(f"Pass 1 candidates: {len(pass1_candidates)}")

    if progress:
        progress(3, 7, "Scoring candidate delivery features")
    candidates = _score_candidates_from_features(bundle, pass1_candidates)
    if progress:
        progress(4, 7, "Pass 1.5: Scoring hook quality")
    hook_results = _score_hooks_with_llm(candidates, model=model, host=host, logger=logger)
    candidates = _apply_hook_scores(candidates, hook_results)
    if logger:
        logger(f"Loaded {len(candidates)} candidate moments.")
        if pass1_used_fallback:
            logger(f"Pass 1 fallback used. reason: {pass1_parse_error}")
    if progress:
        progress(5, 7, "Pass 2: Building rerank prompt")

    prompt = build_prompt(candidates, output_count=max(1, output_count))
    if progress:
        progress(6, 7, "Pass 2: Querying model")

    candidates_by_id = {c.candidate_id: c for c in candidates}

    clips: list[dict]
    pass2_used_fallback = False
    pass2_parse_error = None
    reasoning_minor_count = 0
    reasoning_severe_count = 0
    pass2_on_chunk, pass2_flush = _make_stream_logger(logger, "Pass2 stream:")
    try:
        clips = []
        for attempt_idx in range(2):
            llm_raw = call_llm_generate(
                model=model,
                prompt=(
                    prompt
                    if attempt_idx == 0
                    else (
                        f"{prompt}\n\n"
                        "Retry constraint: prior output had severe score/reason conflicts. "
                        "Return fresh JSON that references only persisted metrics accurately."
                    )
                ),
                host=host,
                on_chunk=pass2_on_chunk,
            )
            pass2_flush()
            parsed = _extract_json_object(llm_raw)
            raw_clips = parsed.get("clips", [])
            if not isinstance(raw_clips, list) or not raw_clips:
                raise RankerError("Model returned no clips list.")
            clips = [_normalize_clip(item, candidates_by_id) for item in raw_clips]
            clips, severe_count, minor_count = _finalize_clip_reasoning_and_evidence(clips)
            reasoning_minor_count = int(minor_count)
            reasoning_severe_count = int(severe_count)
            if severe_count > 0 and attempt_idx == 0:
                if logger:
                    logger(
                        f"Pass2 reasoning validator found {severe_count} severe conflicts; regenerating once."
                    )
                continue
            break
    except Exception as exc:  # noqa: BLE001
        pass2_flush()
        pass2_used_fallback = True
        pass2_parse_error = str(exc)
        clips = fallback_ranked_clips(candidates, output_count=max(1, output_count))
        clips, reasoning_severe_count, reasoning_minor_count = _finalize_clip_reasoning_and_evidence(clips)

    preference_profile = _load_preference_profile()
    personal_fit_applied = _apply_personalized_scores(clips, preference_profile)
    if logger:
        if personal_fit_applied:
            logger(
                f"Applied personalization profile v{int(preference_profile.get('version', 0))} "
                f"(cv={float(preference_profile.get('accuracy_cv', 0.0)):.3f})."
            )
        else:
            logger("Personalization profile not applied (missing/invalid/low-confidence).")

    clips = _enforce_phase_diversity(
        clips=clips,
        candidates_by_id=candidates_by_id,
        output_count=max(1, output_count),
        score_key="final_rank_score" if personal_fit_applied else "editor_score",
    )
    phase_balance = _add_overlap_and_phase_diagnostics(clips, candidates_by_id)
    for idx, clip in enumerate(clips, start=1):
        clip["clip_number"] = idx

    if progress:
        progress(7, 7, "Finalizing ranking output")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "host": host,
        "asset_dir": str(asset_dir),
        "candidate_count": len(candidates),
        "requested_output_count": output_count,
        "pass1_used_fallback": pass1_used_fallback,
        "pass1_parse_error": pass1_parse_error,
        "pass2_used_fallback": pass2_used_fallback,
        "pass2_parse_error": pass2_parse_error,
        "used_fallback": pass1_used_fallback or pass2_used_fallback,
        "parse_error": pass2_parse_error or pass1_parse_error,
        "reasoning_validation": {
            "minor_mismatch_count": int(reasoning_minor_count),
            "severe_mismatch_count": int(reasoning_severe_count),
        },
        "personalization_applied": bool(personal_fit_applied),
        "preference_profile_version": int(preference_profile.get("version", 0))
        if personal_fit_applied and isinstance(preference_profile, dict)
        else None,
        "phase_balance_diagnostics": phase_balance,
        "clips": clips,
    }
    return result


def rank_sermon_moments_from_payloads(
    words_payload: dict,
    energy_payload: dict,
    cadence_payload: dict,
    model: str = "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    output_count: int = 12,
    candidate_limit: int = 24,
    host: str = "http://127.0.0.1:11434",
    logger: Callable[[str], None] | None = None,
) -> dict:
    bundle = {"words": words_payload, "energy": energy_payload, "cadence": cadence_payload}
    words = list(bundle["words"].get("words", []) or [])
    if not words:
        raise RankerError("No words found in words payload.")
    duration = float(bundle["words"].get("media", {}).get("duration_sec", 0.0) or 0.0)
    if duration <= 0:
        duration = float(words[-1].get("e", 0.0) or 0.0)

    pass1_target = max(20, min(30, candidate_limit))
    pass1_candidates, pass1_used_fallback, pass1_parse_error = _build_context_candidates(
        words=words,
        model=model,
        host=host,
        candidate_target=pass1_target,
        duration=duration,
        logger=logger,
    )
    candidates = _score_candidates_from_features(bundle, pass1_candidates)
    hook_results = _score_hooks_with_llm(candidates, model=model, host=host, logger=logger)
    candidates = _apply_hook_scores(candidates, hook_results)
    prompt = build_prompt(candidates, output_count=max(1, output_count))
    candidates_by_id = {c.candidate_id: c for c in candidates}
    pass2_used_fallback = False
    pass2_parse_error = None
    reasoning_minor_count = 0
    reasoning_severe_count = 0
    pass2_on_chunk, pass2_flush = _make_stream_logger(logger, "Pass2 stream:")
    try:
        clips = []
        for attempt_idx in range(2):
            llm_raw = call_llm_generate(
                model=model,
                prompt=(
                    prompt
                    if attempt_idx == 0
                    else (
                        f"{prompt}\n\n"
                        "Retry constraint: prior output had severe score/reason conflicts. "
                        "Return fresh JSON that references only persisted metrics accurately."
                    )
                ),
                host=host,
                on_chunk=pass2_on_chunk,
            )
            pass2_flush()
            parsed = _extract_json_object(llm_raw)
            raw_clips = parsed.get("clips", [])
            if not isinstance(raw_clips, list) or not raw_clips:
                raise RankerError("Model returned no clips list.")
            clips = [_normalize_clip(item, candidates_by_id) for item in raw_clips]
            clips, severe_count, minor_count = _finalize_clip_reasoning_and_evidence(clips)
            reasoning_minor_count = int(minor_count)
            reasoning_severe_count = int(severe_count)
            if severe_count > 0 and attempt_idx == 0:
                if logger:
                    logger(
                        f"Pass2 reasoning validator found {severe_count} severe conflicts; regenerating once."
                    )
                continue
            break
    except Exception as exc:  # noqa: BLE001
        pass2_flush()
        pass2_used_fallback = True
        pass2_parse_error = str(exc)
        clips = fallback_ranked_clips(candidates, output_count=max(1, output_count))
        clips, reasoning_severe_count, reasoning_minor_count = _finalize_clip_reasoning_and_evidence(clips)
    preference_profile = _load_preference_profile()
    personal_fit_applied = _apply_personalized_scores(clips, preference_profile)
    if logger:
        if personal_fit_applied:
            logger(
                f"Applied personalization profile v{int(preference_profile.get('version', 0))} "
                f"(cv={float(preference_profile.get('accuracy_cv', 0.0)):.3f})."
            )
        else:
            logger("Personalization profile not applied (missing/invalid/low-confidence).")
    clips = _enforce_phase_diversity(
        clips=clips,
        candidates_by_id=candidates_by_id,
        output_count=max(1, output_count),
        score_key="final_rank_score" if personal_fit_applied else "editor_score",
    )
    phase_balance = _add_overlap_and_phase_diagnostics(clips, candidates_by_id)
    for idx, clip in enumerate(clips, start=1):
        clip["clip_number"] = idx
    if logger and (pass1_used_fallback or pass2_used_fallback):
        logger(
            "Fallback flags - pass1: %s (%s), pass2: %s (%s)"
            % (pass1_used_fallback, pass1_parse_error, pass2_used_fallback, pass2_parse_error)
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "host": host,
        "candidate_count": len(candidates),
        "requested_output_count": output_count,
        "pass1_used_fallback": pass1_used_fallback,
        "pass1_parse_error": pass1_parse_error,
        "pass2_used_fallback": pass2_used_fallback,
        "pass2_parse_error": pass2_parse_error,
        "used_fallback": pass1_used_fallback or pass2_used_fallback,
        "parse_error": pass2_parse_error or pass1_parse_error,
        "reasoning_validation": {
            "minor_mismatch_count": int(reasoning_minor_count),
            "severe_mismatch_count": int(reasoning_severe_count),
        },
        "personalization_applied": bool(personal_fit_applied),
        "preference_profile_version": int(preference_profile.get("version", 0))
        if personal_fit_applied and isinstance(preference_profile, dict)
        else None,
        "phase_balance_diagnostics": phase_balance,
        "clips": clips,
    }


def rank_result_to_moments_payload(rank_result: dict) -> dict:
    moments = []
    for clip in rank_result.get("clips", []) or []:
        editorial = _editorial_scores(clip)
        cadence = int(editorial.get("cadence", clip.get("editor_score", 0)))
        energy = int(editorial.get("emotion", clip.get("editor_score", 0)))
        contrast = int(round((int(editorial.get("editor", clip.get("editor_score", 0))) + cadence) / 2))
        reasons = [str(clip.get("cadence_marker", "")).lower().replace(" ", "_")]
        moments.append(
            {
                "start": round(_parse_timestamp_to_seconds(str(clip.get("start_time", "00:00:00.000"))), 3),
                "end": round(_parse_timestamp_to_seconds(str(clip.get("end_time", "00:00:00.000"))), 3),
                "hook": str(clip.get("opening_hook", "")).strip(),
                "cadence_score": cadence,
                "energy_score": energy,
                "contrast_score": contrast,
                "overall": int(editorial.get("editor", clip.get("editor_score", 0))),
                "reasons": sorted(set([r for r in reasons if r])),
            }
        )
    return {"moments": moments}


def format_ranked_clips_text(rank_result: dict) -> str:
    clips = rank_result.get("clips", []) or []
    if not clips:
        return "No ranked clips generated."

    lines = []
    if rank_result.get("pass1_used_fallback") or rank_result.get("pass2_used_fallback"):
        lines.append("Fallback Mode Triggered")
        lines.append(f"Pass1 fallback: {bool(rank_result.get('pass1_used_fallback'))}")
        if rank_result.get("pass1_parse_error"):
            lines.append(f"Pass1 reason: {rank_result.get('pass1_parse_error')}")
        lines.append(f"Pass2 fallback: {bool(rank_result.get('pass2_used_fallback'))}")
        if rank_result.get("pass2_parse_error"):
            lines.append(f"Pass2 reason: {rank_result.get('pass2_parse_error')}")
        lines.append("")
    for clip in clips:
        lines.append(f"Clip {clip['clip_number']}")
        lines.append(
            f"Timestamp: {clip['start_time']} - {clip['end_time']} "
            f"({clip['duration_sec']:.1f}s)"
        )
        lines.append(f"Opening Hook: {clip['opening_hook']}")
        lines.append(
            f"Type: {clip['clip_type']} | Cadence Marker: {clip['cadence_marker']} | "
            f"Editor Score: {clip['editor_score']}"
        )
        editorial = _editorial_scores(clip)
        if isinstance(editorial, dict):
            lines.append(
                "Editorial scores: "
                f"Editor {int(editorial.get('editor', 0))}, "
                f"Hook {int(editorial.get('hook', 0))}, "
                f"Cadence {int(editorial.get('cadence', 0))}, "
                f"Standalone {int(editorial.get('standalone', 0))}, "
                f"Emotion {int(editorial.get('emotion', 0))}"
            )
        feature = clip.get("feature_scores", {})
        if isinstance(feature, dict):
            lines.append(
                "Feature scores: "
                f"Overall {int(feature.get('overall_candidate', 0))}, "
                f"Cadence {int(feature.get('cadence', 0))}, "
                f"Energy {int(feature.get('energy', 0))}, "
                f"Contrast {int(feature.get('contrast', 0))}"
            )
        lines.append(f"Editor Reason: {clip['editor_reason']}")
        lines.append(
            f"Scroll-Stopping: {clip['scroll_stopping_strength']} | "
            f"Best Platform: {clip['best_platform_fit']}"
        )
        lines.append("")
    return "\n".join(lines).strip() + "\n"

