#!/usr/bin/env python3
"""Build prompt for a Facebook post derived from the blog post (shorter, fresh reiteration)."""

from __future__ import annotations


def build_facebook_post_prompt(blog_post_markdown: str) -> str:
    """Build prompt for generating a Facebook post from the full blog post. Output: single post text, ~half length."""
    return (
        "You are writing a Facebook post for a church. The post is based on the blog post below, which was adapted from a sermon. "
        "Your job is to write a SHORTER version that feels new and fresh—not a cut-and-paste summary.\n\n"
        "RULES:\n"
        "- Length: About half the length of the blog post (roughly 1–2 min read). Fewer paragraphs, no section headers.\n"
        "- Tone: Direct, warm, conversational. Same key message and biblical grounding, but rephrase so it feels like a standalone reflection, not a recap.\n"
        "- Content: Reiterate what the sermon was about—main idea, one or two supporting points, and a clear takeaway. No filler or generic church phrases.\n"
        "- Do NOT say things like 'In this sermon...' or 'The blog post above...'. Write as if you are speaking to someone scrolling their feed.\n"
        "- End with a short line that invites engagement (e.g. a question or a single-sentence call to action). No hashtag spam.\n\n"
        "OUTPUT FORMAT: Output only the Facebook post text. No title, no labels, no markdown. Plain paragraphs only.\n\n"
        "Blog post (markdown):\n"
        "---\n"
        f"{blog_post_markdown}\n"
        "---\n\n"
        "Write the Facebook post now."
    )
