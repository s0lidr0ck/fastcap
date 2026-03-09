#!/usr/bin/env python3
"""Wix Blog: create draft post (Ricos rich content) and dashboard URL."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from urllib import error, request

def _get_blog_config() -> tuple[str, str, str, str]:
    """Return (token, site_id, base, member_id). member_id may be empty."""
    from wix_cms import get_wix_config
    token, _collection_id, base, site_id = get_wix_config()
    member_id = (os.environ.get("WIX_BLOG_MEMBER_ID") or "").strip()
    return token, site_id, base, member_id


def _text_node(text: str, id_prefix: str = "t", italic: bool = False) -> dict:
    decorations = []
    if italic and text:
        # Ricos range-based decoration for italic (whole segment)
        decorations.append({"type": "ITALIC", "startIndex": 0, "endIndex": len(text)})
    return {
        "type": "TEXT",
        "id": "" if not text else (id_prefix + "_" + str(uuid.uuid4())[:8]),
        "nodes": [],
        "textData": {"text": text, "decorations": decorations},
    }


def _paragraph_node(text: str, id_prefix: str = "p", italic: bool = False) -> dict:
    return {
        "type": "PARAGRAPH",
        "id": id_prefix + "_" + str(uuid.uuid4())[:8],
        "nodes": [_text_node(text, "t", italic=italic)],
        "paragraphData": {},
    }


def _heading_node(text: str, level: int = 2, id_prefix: str = "h") -> dict:
    return {
        "type": "HEADING",
        "id": id_prefix + "_" + str(uuid.uuid4())[:8],
        "nodes": [_text_node(text.strip(), "ht")],
        "headingData": {"level": min(6, max(1, level))},
    }


def _append_paragraph(nodes: list[dict], line: str) -> None:
    """Append a paragraph node; if line is *...* (single asterisks), render as italic."""
    s = line.strip()
    if len(s) >= 2 and s.startswith("*") and s.endswith("*") and not s.startswith("**"):
        inner = s[1:-1].strip()
        nodes.append(_paragraph_node(inner, italic=True))
    else:
        nodes.append(_paragraph_node(line))


def markdown_to_ricos(markdown: str) -> dict:
    """
    Convert minimal markdown (## headings, paragraphs) to a Ricos document with nodes.
    Returns {"nodes": [...]} for richContent.
    """
    nodes: list[dict] = []
    if not markdown or not markdown.strip():
        nodes.append(_paragraph_node(""))
        return {"nodes": nodes}

    # Split by ## but keep the heading text (strip ## and optional bold **)
    parts = re.split(r"\n(?=##\s)", markdown.strip())
    for block in parts:
        block = block.strip()
        if not block:
            continue
        if block.startswith("##"):
            # First line is heading
            first_line, _, rest = block.partition("\n")
            heading_text = first_line.lstrip("#").strip()
            if heading_text.startswith("**") and heading_text.endswith("**"):
                heading_text = heading_text[2:-2].strip()
            nodes.append(_heading_node(heading_text, 2))
            # Rest as paragraphs
            for para in re.split(r"\n\s*\n", rest.strip()):
                line = para.strip()
                if line:
                    _append_paragraph(nodes, line)
        else:
            for para in re.split(r"\n\s*\n", block):
                line = para.strip()
                if line:
                    _append_paragraph(nodes, line)

    if not nodes:
        nodes.append(_paragraph_node(""))
    return {"nodes": nodes}


def _slugify(title: str) -> str:
    """Derive a URL-safe slug from the post title (lowercase, spaces to hyphens, alphanumeric + hyphens)."""
    if not title or not title.strip():
        return ""
    s = title.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "-", s)
    return s.strip("-") or "post"


def create_blog_draft(
    title: str,
    markdown_body: str,
    excerpt: str = "",
    member_id: str = "",
    timeout_sec: int = 30,
) -> dict:
    """
    Create a Wix Blog draft post. Uses markdown_body converted to Ricos.
    Returns API response (includes draftPost.id). Raises ValueError for config, RuntimeError for API errors.
    """
    token, site_id, base, env_member_id = _get_blog_config()
    mid = (member_id or env_member_id).strip()
    if not mid:
        raise ValueError(
            "Wix Blog requires a post owner. Add WIX_BLOG_MEMBER_ID to your .env with the site member GUID "
            "(e.g. site owner from Wix Dashboard → Members). Restart the app after changing .env."
        )
    # Wix Blog API expects a 36-character GUID; dashboard sometimes shows a longer ID (e.g. with instance suffix)
    if len(mid) > 36:
        mid = mid[:36]
    rich = markdown_to_ricos(markdown_body)
    title_clean = (title or "Untitled").strip()
    draft = {
        "title": title_clean,
        "richContent": rich,
        "language": "en",
        "memberId": mid,
        "seoSlug": _slugify(title_clean),
    }
    if excerpt:
        draft["excerpt"] = excerpt.strip()
    # SEO: meta description (excerpt or truncated body); Wix uses seoData.tags with type "meta"
    meta_desc = (excerpt or "").strip()
    if not meta_desc and markdown_body:
        # Use first ~155 chars of plain text as fallback for meta description
        plain = re.sub(r"\s+", " ", re.sub(r"[#*_\[\]]", "", markdown_body)).strip()
        meta_desc = plain[:155].rsplit(" ", 1)[0] if len(plain) > 155 else plain
    # SEO: title tag + meta description
    seo_tags = [{"type": "title", "children": title_clean}]
    if meta_desc:
        seo_tags.append({
            "type": "meta",
            "props": {"name": "description", "content": meta_desc[:160]},
        })
    draft["seoData"] = {"tags": seo_tags}

    url = f"{base}/blog/v3/draft-posts"
    body = json.dumps({"draftPost": draft}, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": token if token.startswith("Bearer ") else f"Bearer {token}",
        "wix-site-id": site_id,
    }
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            err = json.loads(err_body) if err_body.strip() else {}
        except Exception:
            err = {"message": e.reason or str(e)}
        code = e.code
        if code == 401:
            raise ValueError("Wix authentication failed (401). Check WIX_BEARER_TOKEN.") from e
        if code == 403:
            raise ValueError(
                "Wix permission denied (403). Blog requires scope SCOPE.DC-BLOG.MANAGE-BLOG. "
                "If the error mentions memberId, set WIX_BLOG_MEMBER_ID in .env (site owner/member GUID)."
            ) from e
        msg = err.get("message") or err.get("error") or str(err)
        if code == 400:
            msg_lower = (msg or "").lower()
            if "not a valid guid" in msg_lower:
                raise ValueError(
                    "WIX_BLOG_MEMBER_ID must be a 36-character GUID. If your dashboard shows a longer ID, use only the first 36 characters (e.g. xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)."
                ) from e
            if "post owner" in msg_lower or "memberid" in msg_lower or ("member" in msg_lower and "do not exist" in msg_lower):
                raise ValueError(
                    "Wix Blog post owner invalid or not found. Set WIX_BLOG_MEMBER_ID to the 36-char member GUID for a member of this site (Wix Dashboard → Members). Error: " + (msg or "")
                ) from e
        raise RuntimeError(f"Wix Blog API error ({code}): {msg}") from e
    except error.URLError as e:
        raise RuntimeError(f"Wix request failed: {e.reason}") from e


def get_wix_dashboard_url() -> str:
    """Return the Wix Blog > Posts (Drafts) dashboard URL."""
    return (
        "https://manage.wix.com/dashboard/694ca233-e795-40aa-893e-8ee9df637b2e/blog/posts"
        "?status=%5B%7B%22id%22%3A%22UNPUBLISHED%22%2C%22name%22%3A%22Drafts%22%7D%5D"
        "&selectedColumns=col-thumbnail%2Ccol-post%2Ccol-edited%2Ccol-published%2Ccol-views+false"
        "%2Ccol-comments+false%2Ccol-likes+false%2Ccol-categories%2Ccol-tags%2Ccol-spacer"
    )
