#!/usr/bin/env python3
"""Build prompt for YouTube video title and description from sermon transcript.
Also parses SRT to produce an accurate chapter list (timestamps only from SRT)."""

from __future__ import annotations

import re


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
