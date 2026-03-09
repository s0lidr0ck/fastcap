#!/usr/bin/env python3
"""Build prompt for NLC-style blog post from sermon transcript."""

from __future__ import annotations

# Short NLC-style excerpts for tone/structure (from thisisnlc.com examples)
NLC_STYLE_EXCERPT_1 = """## **The Power of Stewarding God's Promises**

Every believer has received promises from God - whether grand or modest in scope. But receiving a promise is only the beginning. How you steward that promise during seasons of waiting often determines whether it will come to fruition or wither away under the weight of doubt and impatience.

Think of God's promises like seeds planted in the soil of your faith. These seeds require proper care, consistent watering with prayer, and protection from the elements of doubt that threaten to choke out their growth."""

NLC_STYLE_EXCERPT_2 = """## **Practical Steps for Breaking Free**

1. **Make a Clean Break**: Just as Elisha burned his farming equipment, identify and eliminate the "backup plans" that tempt you to return to your old life.
2. **Refuse to Tarry**: When voices (internal or external) urge you to remain where you are, respond with Elisha's determination: "I will not stay here."
3. **Embrace Your New Identity**: The moment you're saved, you receive a new name and a new nature.

Remember: God never lets His people walk through mud - He provides dry ground when you're willing to move forward."""


def build_blog_post_prompt(
    transcript: str,
    preacher_name: str = "",
    date_preached: str = "",
) -> str:
    """Build prompt for generating one NLC-style blog post from full transcript. Output: markdown (title line then body)."""
    context_lines: list[str] = []
    if preacher_name or date_preached:
        context_lines.append("Sermon context (use when relevant; refer to speaker by name, not \"the preacher\"):")
        if preacher_name:
            context_lines.append(f"- Preacher/speaker: {preacher_name.strip()}")
        if date_preached:
            context_lines.append(f"- Date preached: {date_preached.strip()}")
        context_lines.append("")
    context_block = "\n".join(context_lines) if context_lines else ""

    return (
        "You are writing a blog post for a church (NLC). Your post must match the tone, structure, and length of the example style below. "
        "Use the ENTIRE sermon transcript as your only source; do not invent content. Base every point and illustration on what the speaker actually said.\n\n"
        "STYLE RULES:\n"
        "- Tone: First person, speaking directly to the reader. Use \"you\" and \"your\". Do NOT recap the sermon by saying \"Pastor X said\" or \"the speaker taught that\"—write as if you are teaching the reader yourself. Apply the sermon's truths to the reader's life without attributing every point to the preacher.\n"
        "- Length: About 3 min read (several hundred words, 4–6 main sections).\n"
        "- Structure: One clear title line, then body with ## section headings, short paragraphs (2–4 sentences), bullet or numbered lists where helpful. "
        "Include one \"Remember:\" or a short Scripture quote, and end with a brief call to action.\n"
        "- Content: Biblical references and stories from the sermon, applied to the reader's life. No filler.\n\n"
        "EXAMPLE STYLE (match this tone and format):\n"
        "---\n"
        f"{NLC_STYLE_EXCERPT_1}\n\n"
        f"{NLC_STYLE_EXCERPT_2}\n"
        "---\n\n"
        + (f"{context_block}" if context_block else "")
        + "OUTPUT FORMAT: Output only markdown. First line is the post title (no # prefix). Then a blank line. Then the full post body with ## for main sections. No commentary before or after.\n\n"
        "Sermon transcript:\n"
        "---\n"
        f"{transcript}\n"
        "---\n\n"
        "Write the blog post in markdown now (title on first line, then body)."
    )
