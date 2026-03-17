#!/usr/bin/env python3
"""Build prompt helpers for YouTube copy and thumbnail prompt variants."""

from __future__ import annotations

import json
import re
from typing import Any


THUMBNAIL_VARIANT_LABELS = ("A", "B", "C")
THUMBNAIL_DEFAULT_POSITIONS = ("center", "left", "right")
THUMBNAIL_ALLOWED_POSITIONS = {"center", "left", "right", "lower_third"}
THUMBNAIL_ALLOWED_LIGHTING = (
    "warm sunrise light",
    "soft window light",
    "dramatic storm lighting",
    "golden hour sunlight",
    "cool evening light",
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "god",
    "how",
    "in",
    "into",
    "is",
    "of",
    "on",
    "the",
    "through",
    "to",
    "what",
    "why",
    "with",
    "you",
    "your",
}


def _ensure_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if "```" in stripped:
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", stripped)
        if match:
            return json.loads(match.group(1))
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)
    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        raise ValueError("Response did not contain a JSON object.")
    return json.loads(match.group(0))


def _safe_excerpt(text: str, limit: int) -> str:
    clean = " ".join((text or "").split()).strip()
    if len(clean) <= limit:
        return clean
    trimmed = clean[:limit].rsplit(" ", 1)[0].strip()
    return trimmed or clean[:limit].strip()


def _pick_sermon_theme(
    youtube_title: str,
    youtube_description: str,
    sermon_metadata: dict[str, Any] | None = None,
) -> str:
    metadata = sermon_metadata if isinstance(sermon_metadata, dict) else {}
    for key in ("topics", "tags", "mainPoints", "teachingStatements"):
        values = metadata.get(key)
        if isinstance(values, list):
            for item in values:
                text = _ensure_str(item)
                if text:
                    return text
    for candidate in (youtube_title, youtube_description):
        text = _ensure_str(candidate)
        if text:
            return text
    return "hope in hardship"


def _pick_sermon_summary(
    youtube_description: str,
    sermon_metadata: dict[str, Any] | None = None,
) -> str:
    metadata = sermon_metadata if isinstance(sermon_metadata, dict) else {}
    description = _ensure_str(metadata.get("description"))
    if description:
        return _safe_excerpt(description, 220)
    text = _ensure_str(youtube_description)
    if not text:
        return ""
    first_block = text.split("\n\n", 1)[0].strip()
    return _safe_excerpt(first_block or text, 220)


def _keyword_profile(theme_text: str) -> tuple[str, list[str], list[str]]:
    text = theme_text.lower()
    profiles = [
        (
            ("battle", "storm", "warfare", "attack", "struggle"),
            "spiritual battle",
            [
                "person standing in a storm with wind and rain",
                "person climbing a rocky ridge against heavy wind",
                "person walking forward through dark clouds with resolve",
            ],
            ["dramatic storm lighting", "cool evening light", "golden hour sunlight"],
        ),
        (
            ("hope", "healing", "future", "light", "restoration"),
            "hope",
            [
                "person watching sunrise from a hillside",
                "person praying near window light at dawn",
                "person standing in an open field as morning light breaks through clouds",
            ],
            ["warm sunrise light", "golden hour sunlight", "soft window light"],
        ),
        (
            ("reflect", "reflection", "truth", "search", "honest", "repent"),
            "reflection",
            [
                "person looking into a mirror in a softly lit room",
                "person sitting alone in thoughtful silence near a window",
                "person reading a Bible alone in a quiet forest clearing",
            ],
            ["soft window light", "cool evening light", "golden hour sunlight"],
        ),
        (
            ("teach", "teaching", "wisdom", "scripture", "word", "truth"),
            "teaching",
            [
                "hands holding an open Bible in dramatic light",
                "person studying scripture at a wooden table",
                "person reading a Bible in a quiet outdoor setting",
            ],
            ["soft window light", "golden hour sunlight", "warm sunrise light"],
        ),
        (
            ("trust", "faith", "persever", "trial", "wait", "endure"),
            "perseverance",
            [
                "person sitting on a hillside watching sunrise after a long night",
                "person standing on a rocky overlook with determined posture",
                "person kneeling in prayer as early morning light fills the scene",
            ],
            ["warm sunrise light", "golden hour sunlight", "dramatic storm lighting"],
        ),
        (
            ("freedom", "breakthrough", "deliver", "chains", "victory"),
            "breakthrough",
            [
                "person breaking chains in dramatic light",
                "person stepping out of shadow into bright light",
                "person standing on a mountain ridge in a victorious pose",
            ],
            ["dramatic storm lighting", "golden hour sunlight", "warm sunrise light"],
        ),
    ]
    for keywords, label, scenes, lighting in profiles:
        if any(keyword in text for keyword in keywords):
            return label, scenes, lighting
    return (
        "hope",
        [
            "person watching sunrise from a hillside",
            "person praying near window light",
            "person standing in an open landscape with hopeful posture",
        ],
        ["warm sunrise light", "soft window light", "golden hour sunlight"],
    )


def _clean_thumbnail_phrase(phrase: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", _ensure_str(phrase))
    cleaned = " ".join(words[:3]).upper().strip()
    return cleaned or "HOLD ON"


def _pick_thumbnail_phrase(
    youtube_title: str,
    youtube_description: str,
    sermon_metadata: dict[str, Any] | None = None,
) -> str:
    theme_text = " ".join(
        part
        for part in (
            _pick_sermon_theme(youtube_title, youtube_description, sermon_metadata),
            _ensure_str(youtube_title),
        )
        if part
    ).lower()
    phrase_map = [
        (("battle", "storm", "warfare", "struggle"), "STAND FIRM"),
        (("hope", "future", "light", "healing"), "HOLD ON"),
        (("truth", "reflect", "repent", "honest"), "FACE THE TRUTH"),
        (("teach", "scripture", "word", "wisdom"), "FIND ANSWERS"),
        (("trust", "faith", "persever", "trial", "wait", "endure"), "DON'T QUIT"),
        (("freedom", "breakthrough", "chains", "victory", "deliver"), "BREAK FREE"),
        (("prayer", "pray"), "KEEP PRAYING"),
    ]
    for keywords, phrase in phrase_map:
        if any(keyword in theme_text for keyword in keywords):
            return phrase

    title_words = [
        word.upper()
        for word in re.findall(r"[A-Za-z0-9']+", youtube_title)
        if word.lower() not in _STOPWORDS and len(word) > 2
    ]
    if title_words:
        return _clean_thumbnail_phrase(" ".join(title_words[:3]))
    return "HOLD ON"


def render_thumbnail_prompt(variant: dict[str, str]) -> str:
    sermon_title = _ensure_str(variant.get("sermon_title"))
    sermon_summary = _ensure_str(variant.get("sermon_summary"))
    sermon_theme = _ensure_str(variant.get("sermon_theme"))
    thumbnail_phrase = _clean_thumbnail_phrase(variant.get("thumbnail_phrase", ""))
    scene_concept = _ensure_str(variant.get("scene_concept")) or "person watching sunrise from a hillside"
    text_position = _ensure_str(variant.get("text_position")).lower()
    if text_position not in THUMBNAIL_ALLOWED_POSITIONS:
        text_position = "center"
    lighting_description = _ensure_str(variant.get("lighting_description")) or "warm sunrise light"

    return (
        "Create a cinematic YouTube sermon thumbnail.\n\n"
        f"Message context:\nTitle: {sermon_title or 'Untitled sermon'}\n"
        f"Theme: {sermon_theme or 'hope'}\n"
        f"Summary: {sermon_summary or 'A message of faith, hope, and perseverance.'}\n\n"
        f"Scene:\n{scene_concept}\n\n"
        "Composition:\n"
        "A human subject is in the foreground.\n"
        f'Large bold text "{thumbnail_phrase}" sits in the {text_position} area in the middle depth layer.\n'
        "The environment is in the background.\n"
        "The foreground subject must partially overlap at least one letter of the text to create natural depth.\n\n"
        "Visual hierarchy:\n"
        "SUBJECT (foreground)\n"
        "TEXT (middle layer)\n"
        "ENVIRONMENT (background)\n\n"
        f"Lighting:\n{lighting_description}\n\n"
        "Style:\n"
        "Realistic photography, cinematic lighting, shallow depth of field, subtle film grain, "
        "bold composition, emotionally aligned with the message, clean YouTube thumbnail readability.\n\n"
        "Important constraints:\n"
        "Use a visual metaphor for the sermon message, not a literal church service scene.\n"
        f'Only visible text in the image should be "{thumbnail_phrase}".\n'
        "Keep the text large, bold, and easy to read at small sizes.\n"
        "The subject must partially occlude the text."
    )


def fallback_thumbnail_prompt_variants(
    youtube_title: str,
    youtube_description: str,
    sermon_metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    sermon_theme = _pick_sermon_theme(youtube_title, youtube_description, sermon_metadata)
    sermon_summary = _pick_sermon_summary(youtube_description, sermon_metadata)
    _profile_label, scenes, lighting_options = _keyword_profile(sermon_theme)
    base_phrase = _pick_thumbnail_phrase(youtube_title, youtube_description, sermon_metadata)
    variants: list[dict[str, str]] = []
    for index, label in enumerate(THUMBNAIL_VARIANT_LABELS):
        phrase = base_phrase
        if index == 1 and base_phrase == "HOLD ON":
            phrase = "DON'T QUIT"
        elif index == 2 and base_phrase in {"HOLD ON", "DON'T QUIT"}:
            phrase = "STAND FIRM"
        variant = {
            "label": label,
            "sermon_title": _ensure_str(youtube_title),
            "sermon_summary": sermon_summary,
            "sermon_theme": sermon_theme,
            "thumbnail_phrase": phrase,
            "scene_concept": scenes[index % len(scenes)],
            "text_position": THUMBNAIL_DEFAULT_POSITIONS[index],
            "lighting_description": lighting_options[index % len(lighting_options)],
        }
        variant["prompt"] = render_thumbnail_prompt(variant)
        variants.append(variant)
    return variants


def build_thumbnail_prompt_planner(
    transcript: str,
    youtube_title: str,
    youtube_description: str,
    preacher_name: str = "",
    date_preached: str = "",
    sermon_metadata: dict[str, Any] | None = None,
) -> str:
    metadata = sermon_metadata if isinstance(sermon_metadata, dict) else {}
    metadata_block = json.dumps(
        {
            "topics": metadata.get("topics") or [],
            "tags": metadata.get("tags") or [],
            "mainPoints": metadata.get("mainPoints") or [],
            "teachingStatements": metadata.get("teachingStatements") or [],
            "description": metadata.get("description") or "",
        },
        ensure_ascii=False,
        indent=2,
    )
    context_parts = []
    if preacher_name:
        context_parts.append(f"Speaker: {preacher_name.strip()}")
    if date_preached:
        context_parts.append(f"Date: {date_preached.strip()}")
    context_block = "\n".join(context_parts)
    transcript_excerpt = _safe_excerpt(transcript, 2200)
    description_excerpt = _safe_excerpt(youtube_description, 500)

    return (
        "You are planning 3 YouTube thumbnail prompt variants for a church sermon video. "
        "Use the sermon title, description, sermon metadata, and transcript excerpt as your only source.\n\n"
        "GOAL:\n"
        "Create 3 strong thumbnail prompt concepts that feel visually consistent, emotionally aligned, and varied enough to test.\n\n"
        "HARD RULES:\n"
        "- The composition must be layered as subject in foreground, text in middle layer, environment in background.\n"
        "- The foreground subject must partially overlap the text.\n"
        "- Use metaphorical imagery, not a literal church service scene.\n"
        "- The thumbnail phrase must be 1-3 words and emotionally strong.\n"
        "- Create exactly 3 variants.\n"
        "- Use these text positions in order: A=center, B=left, C=right.\n"
        "- Choose lighting from this list only: "
        + ", ".join(THUMBNAIL_ALLOWED_LIGHTING)
        + ".\n\n"
        "OUTPUT FORMAT:\n"
        "Return only valid JSON with this shape:\n"
        "{\n"
        '  "variants": [\n'
        '    {\n'
        '      "label": "A",\n'
        '      "sermon_theme": "...",\n'
        '      "sermon_summary": "...",\n'
        '      "thumbnail_phrase": "...",\n'
        '      "scene_concept": "...",\n'
        '      "text_position": "center",\n'
        '      "lighting_description": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        + (f"Context:\n{context_block}\n\n" if context_block else "")
        + f"YouTube title:\n{youtube_title.strip()}\n\n"
        + f"YouTube description:\n{description_excerpt}\n\n"
        + f"Sermon metadata:\n{metadata_block}\n\n"
        + f"Transcript excerpt:\n{transcript_excerpt}\n\n"
        "Return exactly 3 variants in JSON now."
    )


def parse_thumbnail_prompt_variants(
    raw: str,
    youtube_title: str,
    youtube_description: str,
    sermon_metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    try:
        data = _extract_json_object(raw)
    except Exception:
        return fallback_thumbnail_prompt_variants(youtube_title, youtube_description, sermon_metadata)

    raw_variants = data.get("variants")
    if not isinstance(raw_variants, list):
        return fallback_thumbnail_prompt_variants(youtube_title, youtube_description, sermon_metadata)

    fallback = fallback_thumbnail_prompt_variants(youtube_title, youtube_description, sermon_metadata)
    variants: list[dict[str, str]] = []
    for index, default_variant in enumerate(fallback):
        source = raw_variants[index] if index < len(raw_variants) and isinstance(raw_variants[index], dict) else {}
        variant = dict(default_variant)
        if source:
            variant["label"] = THUMBNAIL_VARIANT_LABELS[index]
            variant["sermon_title"] = _ensure_str(youtube_title)
            variant["sermon_theme"] = _ensure_str(source.get("sermon_theme")) or default_variant["sermon_theme"]
            variant["sermon_summary"] = _safe_excerpt(
                _ensure_str(source.get("sermon_summary")) or default_variant["sermon_summary"],
                220,
            )
            variant["thumbnail_phrase"] = _clean_thumbnail_phrase(
                _ensure_str(source.get("thumbnail_phrase")) or default_variant["thumbnail_phrase"]
            )
            variant["scene_concept"] = _ensure_str(source.get("scene_concept")) or default_variant["scene_concept"]
            text_position = _ensure_str(source.get("text_position")).lower()
            variant["text_position"] = (
                text_position if text_position in THUMBNAIL_ALLOWED_POSITIONS else default_variant["text_position"]
            )
            lighting_description = _ensure_str(source.get("lighting_description"))
            variant["lighting_description"] = lighting_description or default_variant["lighting_description"]
        variant["prompt"] = render_thumbnail_prompt(variant)
        variants.append(variant)
    return variants


def _srt_time_to_seconds(time_str: str) -> float | None:
    """Parse SRT time (HH:MM:SS,mmm or HH:MM:SS.mmm) to seconds. Returns None if not matched."""
    time_str = time_str.strip()
    m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})$", time_str)
    if m:
        h, m_, s, ms = m.groups()
        return int(h) * 3600 + int(m_) * 60 + int(s) + int(ms) / 1000.0
    return None


def parse_srt_to_chapters(srt_text: str) -> list[tuple[float, str]]:
    """Parse SRT content into a list of (start_seconds, cue_text) for chapters.
    Only uses actual timestamps from the SRT; labels are the cue text (truncated).
    Returns empty list if input doesn't look like SRT or has no valid cues."""
    chapters: list[tuple[float, str]] = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        if len(lines) < 2:
            continue
        # Find the line with " --> " (timestamp range); parse first timestamp
        start_sec = None
        text_lines: list[str] = []
        for i, line in enumerate(lines):
            if " --> " in line:
                left = line.split(" --> ", 1)[0].strip()
                start_sec = _srt_time_to_seconds(left)
                text_lines = [ln.strip() for ln in lines[i + 1 :] if ln.strip()]
                break
        if start_sec is not None and text_lines:
            label = " ".join(text_lines).strip()
            if label:
                chapters.append((start_sec, label))
    # Sort by time and de-dupe by time (keep first at each start)
    if not chapters:
        return []
    chapters.sort(key=lambda x: x[0])
    seen: set[float] = set()
    unique: list[tuple[float, str]] = []
    for start, label in chapters:
        if start not in seen:
            seen.add(start)
            unique.append((start, label))
    return unique


def seconds_to_youtube_time(total_seconds: float) -> str:
    """Format seconds as YouTube chapter time: M:SS or H:MM:SS (no leading zero on first component)."""
    if total_seconds < 0:
        total_seconds = 0.0
    total_seconds = round(total_seconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def get_chapter_segments(
    chapters: list[tuple[float, str]],
    interval_sec: float = 300.0,
) -> list[tuple[float, str]]:
    """Group cues into interval_sec windows (e.g. 5 min). Returns one (start_sec, combined_text) per window.
    start_sec is the first cue time in that window; combined_text is all cue text in that window for summarization."""
    if not chapters or interval_sec <= 0:
        return []
    buckets: dict[int, list[tuple[float, str]]] = {}
    for start_sec, text in chapters:
        bucket = int(start_sec / interval_sec)
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append((start_sec, text))
    # One entry per bucket: (first cue's start_sec, all text in bucket joined)
    segments: list[tuple[float, str]] = []
    for bucket in sorted(buckets.keys()):
        cues = buckets[bucket]
        first_start = cues[0][0]
        combined = " ".join(t for _, t in cues).strip()
        if combined:
            segments.append((first_start, combined))
    return segments


def format_youtube_chapters(chapters: list[tuple[float, str]], max_label_len: int = 50) -> str:
    """Format chapters for YouTube description. Each line: "M:SS Label" (label truncated, one line)."""
    if not chapters:
        return ""
    lines = []
    for start_sec, label in chapters:
        time_str = seconds_to_youtube_time(start_sec)
        # Single line, no newlines; truncate; strip problematic chars for YouTube
        clean = " ".join(label.split())[:max_label_len].strip()
        if not clean:
            clean = f"Chapter {time_str}"
        lines.append(f"{time_str} {clean}")
    return "Chapters:\n" + "\n".join(lines)


def srt_to_plain_text(srt_text: str) -> str:
    """Strip SRT to plain transcript (cue text only, in order) for LLM prompt."""
    chapters = parse_srt_to_chapters(srt_text)
    if not chapters:
        return srt_text.strip()
    return "\n".join(label for _, label in chapters).strip()


def build_youtube_prompt(
    transcript: str,
    preacher_name: str = "",
    date_preached: str = "",
) -> str:
    """Build prompt for generating a YouTube title and description. Output: first line = title, then blank line, then description."""
    context_parts = []
    if preacher_name:
        context_parts.append(f"Speaker: {preacher_name.strip()}")
    if date_preached:
        context_parts.append(f"Date: {date_preached.strip()}")
    context_block = "\n".join(context_parts) + "\n\n" if context_parts else ""

    return (
        "You are writing a YouTube video title and description for a church sermon. "
        "Use the sermon transcript below as your only source. Do not invent content.\n\n"
        "RULES:\n"
        "- Title: One line, under 100 characters. Catchy, clear, and search-friendly. No clickbait. "
        "Capture the main idea or a key phrase from the sermon. No quotes around the title.\n"
        "- Description: 2–4 short paragraphs. Open with a hook that summarizes the message in 1–2 sentences. "
        "Then 1–2 paragraphs expanding on the main takeaway. End with a line inviting viewers to like/subscribe or visit the church. "
        "You may include a line like 'From [Speaker]’s sermon on [Date].' at the start of the description. "
        "Keep it under 300 words. Do NOT include timestamps or a chapter list in the description—we add an accurate chapter list separately from the transcript.\n\n"
        + (f"Context:\n{context_block}" if context_block.strip() else "")
        + "OUTPUT FORMAT: Output only plain text. First line is the title (no prefix). Then a blank line. Then the full description. No labels like 'Title:' or 'Description:'.\n\n"
        "Sermon transcript:\n"
        "---\n"
        f"{transcript}\n"
        "---\n\n"
        "Write the YouTube title (first line) and description (after a blank line) now."
    )


def build_youtube_prompt_with_chapters(
    transcript: str,
    segments: list[tuple[float, str]],
    preacher_name: str = "",
    date_preached: str = "",
) -> str:
    """Single prompt for title + description + chapter titles (one per segment). Output ends with ---CHAPTERS--- then one title per line."""
    context_parts = []
    if preacher_name:
        context_parts.append(f"Speaker: {preacher_name.strip()}")
    if date_preached:
        context_parts.append(f"Date: {date_preached.strip()}")
    context_block = "\n".join(context_parts) + "\n\n" if context_parts else ""

    segments_block = ""
    for i, (_, text) in enumerate(segments, 1):
        excerpt = (text[:1500] + "...") if len(text) > 1500 else text
        segments_block += f"Segment {i} (~5 min):\n{excerpt}\n\n"

    return (
        "You are writing a YouTube video title, description, and chapter titles for a church sermon. "
        "Use the sermon content below as your only source. Do not invent content.\n\n"
        "RULES:\n"
        "- Title: One line, under 100 characters. Catchy, clear, and search-friendly. No clickbait.\n"
        "- Description: 2–4 short paragraphs after the title. Open with a hook, then main takeaway, end with like/subscribe. "
        "You may include a line like 'From [Speaker]'s sermon on [Date].' Do NOT include timestamps or a chapter list in the description.\n"
        "- Chapter titles: The sermon is split into "
        + str(len(segments))
        + " segments below (each ~5 minutes). You will output exactly one short chapter title (3–8 words) per segment, in the same order, summarizing that segment.\n\n"
        + (f"Context:\n{context_block}" if context_block.strip() else "")
        + "OUTPUT FORMAT (plain text only):\n"
        "1. First line: the video title.\n"
        "2. A blank line.\n"
        "3. The description (multiple paragraphs).\n"
        "4. A line that says exactly: ---CHAPTERS---\n"
        "5. Then exactly "
        + str(len(segments))
        + " lines: each line is the chapter title for Segment 1, Segment 2, ... in order. No numbers or timestamps, just the title text.\n\n"
        "Sermon segments:\n"
        "---\n"
        f"{segments_block}"
        "---\n\n"
        "Write the title, description, then ---CHAPTERS---, then the chapter titles (one per segment) now."
    )


def parse_youtube_response(raw: str, num_segments: int = 0) -> tuple[str, str, list[str]]:
    """Parse single-prompt response: title, description, and optional chapter titles after ---CHAPTERS---.
    Returns (title, description, chapter_titles). chapter_titles is empty if num_segments is 0 or parse fails."""
    raw = (raw or "").strip()
    chapter_titles: list[str] = []
    if num_segments > 0 and "---CHAPTERS---" in raw:
        before, _, after = raw.partition("---CHAPTERS---")
        raw = before.strip()
        lines = [ln.strip() for ln in after.strip().split("\n") if ln.strip()]
        chapter_titles = lines[:num_segments]
        if len(chapter_titles) < num_segments:
            chapter_titles = []
    lines = raw.split("\n")
    title = lines[0].strip() if lines else ""
    description = "\n".join(lines[1:]).lstrip() if len(lines) > 1 else ""
    return title, description, chapter_titles
