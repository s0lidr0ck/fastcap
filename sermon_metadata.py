#!/usr/bin/env python3
"""Sermon metadata extraction: SCRIBE prompt (strict JSON) and parser/validator."""

from __future__ import annotations

import json
import re
from typing import Any

# Tag options for SECTION 5 (from SCRIBE prompt)
TAG_OPTIONS = (
    "Genesis, Exodus, Leviticus, Numbers, Deuteronomy, Joshua, Judges, Ruth, "
    "1 Samuel, 2 Samuel, 1 Kings, 2 Kings, 1 Chronicles, 2 Chronicles, Ezra, "
    "Nehemiah, Esther, Job, Psalms, Proverbs, Ecclesiastes, Song of Solomon, "
    "Isaiah, Jeremiah, Lamentations, Ezekiel, Daniel, Hosea, Joel, Amos, Obadiah, "
    "Jonah, Micah, Nahum, Habakkuk, Zephaniah, Haggai, Zechariah, Malachi, "
    "Matthew, Mark, Luke, John, Acts, Romans, 1 Corinthians, 2 Corinthians, "
    "Galatians, Ephesians, Philippians, Colossians, 1 Thessalonians, "
    "2 Thessalonians, 1 Timothy, 2 Timothy, Titus, Philemon, Hebrews, James, "
    "1 Peter, 2 Peter, 1 John, 2 John, 3 John, Jude, Revelation, "
    "Holy Spirit, Pentecost, Gifts of the Spirit, Speaking in Tongues, Prophecy, "
    "Miracles, Healing, Deliverance, Spiritual Warfare, Angelology, Demonology, "
    "Signs and Wonders, Second Coming, Revival, Anointing, Baptism in the Spirit, "
    "Fasting and Prayer, Faith Healing, Divine Intervention, Apostolic, "
    "Prophetic Ministry, Spiritual Gifts, Authority of the Believer, "
    "Supernatural Encounters, Visions and Dreams, Intercessory Prayer, "
    "Power of God, Glory of God, Kingdom of God, Government, Nations, Politics, "
    "Law, Freedom, Justice, Social Issues, Environment, Conservation, "
    "Mountain, River, Ocean, Forest, Earth, World, Heart, Fire, Wind, Earth, Water, "
    "Joy, Peace, Love, Anxiety, Depression, Anger, Fear, Hope, Despair, "
    "Contentment, Grief, Sadness, Loneliness, Guilt, Shame, Optimism, Pessimism, "
    "Stress, Tranquility, Gratitude, Empathy, Compassion, Frustration, Elation, "
    "Envy, Jealousy, Confidence, Insecurity, Resilience, Vulnerability, Nostalgia, "
    "Salvation, Prayer, Forgiveness, Sin, Repentance, Trust, Worship, Praise, "
    "Spiritual Growth, Leadership, Discipleship, Community, Outreach, Evangelism, "
    "Missions, Family, Marriage, Parenting, Youth, Children, Men's Issues, "
    "Women's Issues, Senior's Ministry, Bible Study, Church Growth, Church History, "
    "Apostles, Victory, Thanksgiving, Stewardship, Money, Prosperity, Suffering, "
    "Perseverance, Trials, Temptation, Holiness, Righteousness, Ethics, Culture, "
    "Worldview, Creation, Eschatological Events, Fellowship, Unity, Controversy, "
    "Doctrine, Faithfulness, Obedience, Redemption, Sanctification, Justification, "
    "Conviction, Inspiration, Transformation, Consecration, Accountability, "
    "Mentoring, Humility, Patience, Wisdom, Discernment, Fear of God, "
    "Sovereignty of God, Majesty of God"
)

REQUIRED_KEYS = (
    "title",
    "description",
    "scriptures",
    "mainPoints",
    "tags",
    "propheticStatements",
    "keyMoments",
    "topics",
    "teachingStatements",
)

KEY_MOMENT_KEYS = ("timestamp", "quote", "explanation")


def build_scribe_prompt(
    transcript: str,
    preacher_name: str = "",
    date_preached: str = "",
) -> str:
    """Build the SCRIBE prompt that demands strict JSON output only."""
    context_lines: list[str] = []
    if preacher_name or date_preached:
        context_lines.append("Sermon context (use this in your summary; do not say \"the preacher\" when a name is given):")
        if preacher_name:
            context_lines.append(f"- Preacher/speaker: {preacher_name.strip()}")
        if date_preached:
            context_lines.append(f"- Date preached: {date_preached.strip()}")
        context_lines.append("")
    context_block = "\n".join(context_lines) if context_lines else ""

    return (
        "You are SCRIBE, a digital assistant trained to analyze Pentecostal sermon transcripts. "
        "Your task is to read a sermon transcript and output a single JSON object (and nothing else) "
        "with the exact keys described below.\n\n"
        "IMPORTANT: Before producing output, read the ENTIRE transcript. "
        "Do not invent scriptures or statements not present. "
        "Output ONLY valid JSON—no markdown code fences, no commentary before or after.\n\n"
        + (f"{context_block}" if context_block else "")
        + "Required JSON shape (use these exact key names):\n"
        "- title (string): strong sermon title from the main message\n"
        "- description (string): 2–4 paragraph summary (main teaching, theme, challenge/encouragement). "
        "When sermon context includes a preacher name, refer to them by name in the description, not as \"the preacher\".\n"
        "- scriptures (array of strings): each item like \"Reference — how it was used\", e.g. \"Isaiah 46:10 — Used to illustrate...\"\n"
        "- mainPoints (array of strings): key teaching points\n"
        "- tags (array of strings): 3–10 tags from the tag list below only\n"
        "- propheticStatements (array of strings): verbatim declarations (Thus saith the Lord, etc.); empty array if none\n"
        "- keyMoments (array of objects): each { \"timestamp\": \"...\", \"quote\": \"...\", \"explanation\": \"...\" }\n"
        "- topics (array of strings): natural-language theme phrases\n"
        "- teachingStatements (array of strings): clear doctrinal truth statements (complete sentences)\n\n"
        "Tag options (choose only from this list):\n"
        f"{TAG_OPTIONS}\n\n"
        "Transcript:\n"
        "---\n"
        f"{transcript}\n"
        "---\n\n"
        "Respond with only the JSON object, no other text."
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract first JSON object from response (handles surrounding text)."""
    stripped = text.strip()
    # Remove optional markdown code block
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


def _ensure_list(value: Any, field: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def _ensure_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_key_moment(obj: Any) -> dict[str, str] | None:
    if not isinstance(obj, dict):
        return None
    ts = _ensure_str(obj.get("timestamp"))
    quote = _ensure_str(obj.get("quote"))
    expl = _ensure_str(obj.get("explanation"))
    if not quote and not expl:
        return None
    return {"timestamp": ts or "Unknown", "quote": quote, "explanation": expl}


def parse_sermon_metadata(raw: str) -> tuple[dict[str, Any], list[str]]:
    """
    Parse and validate raw model output into normalized sermon metadata.
    Returns (payload, warnings). Raises ValueError on missing required fields or invalid structure.
    """
    warnings: list[str] = []
    data = _extract_json_object(raw)

    # Required keys
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")

    # Normalize
    title = _ensure_str(data["title"])
    if not title:
        raise ValueError("Required field 'title' is empty.")

    description = _ensure_str(data["description"])
    scriptures = _ensure_list(data["scriptures"], "scriptures")
    scriptures = [str(s).strip() for s in scriptures if str(s).strip()]

    main_points = _ensure_list(data["mainPoints"], "mainPoints")
    main_points = [str(p).strip() for p in main_points if str(p).strip()]

    tags_raw = _ensure_list(data["tags"], "tags")
    tags = []
    for t in tags_raw:
        s = str(t).strip()
        if s and s not in tags:
            tags.append(s)

    prophetic = _ensure_list(data["propheticStatements"], "propheticStatements")
    prophetic = [str(p).strip() for p in prophetic if str(p).strip()]

    key_moments_raw = _ensure_list(data["keyMoments"], "keyMoments")
    key_moments: list[dict[str, str]] = []
    for i, km in enumerate(key_moments_raw):
        normalized = _normalize_key_moment(km)
        if normalized:
            key_moments.append(normalized)
        elif km:
            warnings.append(f"keyMoments[{i}] missing quote/explanation, skipped")

    topics_raw = _ensure_list(data["topics"], "topics")
    topics = list(dict.fromkeys(str(t).strip() for t in topics_raw if str(t).strip()))

    teaching = _ensure_list(data["teachingStatements"], "teachingStatements")
    teaching = [str(t).strip() for t in teaching if str(t).strip()]

    payload: dict[str, Any] = {
        "title": title,
        "description": description,
        "scriptures": scriptures,
        "mainPoints": main_points,
        "tags": tags,
        "propheticStatements": prophetic,
        "keyMoments": key_moments,
        "topics": topics,
        "teachingStatements": teaching,
    }
    return payload, warnings
