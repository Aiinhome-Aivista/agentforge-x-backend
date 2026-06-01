"""Blog service — CRUD over the ``blog_posts`` table.

Public reads (only published posts). Admin-only writes. The route layer is
responsible for enforcing ``is_admin``; this module is intentionally
agnostic so it can be re-used (e.g. by a CLI seeding script).
"""

from __future__ import annotations

import re
import logging
import secrets
from typing import Optional, List, Dict, Any

from app.db.db_connection import get_mysql_connection

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    s = _SLUG_RE.sub("-", (title or "").lower()).strip("-")
    return (s[:160] or f"post-{secrets.token_hex(4)}")


def _row_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the heavy `content` field for list responses."""
    if not row:
        return row
    out = dict(row)
    out.pop("content", None)
    return out


# ── Public reads ────────────────────────────────────────────────────────────
def list_posts(*, include_unpublished: bool = False,
               limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    where = "" if include_unpublished else "WHERE published=1"
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            f"""
            SELECT id, slug, title, excerpt, image_url, tags,
                   author_id, author_name, published, content,
                   created_at, updated_at
              FROM blog_posts
            {where}
             ORDER BY created_at DESC, id DESC
             LIMIT %s OFFSET %s
            """,
            (max(1, min(int(limit or 50), 200)), max(0, int(offset or 0))),
        )
        return cur.fetchall() or []
    finally:
        cur.close()
        conn.close()


def get_post(*, post_id: Optional[int] = None,
             slug: Optional[str] = None,
             include_unpublished: bool = False) -> Optional[Dict[str, Any]]:
    if not post_id and not slug:
        return None
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        if post_id:
            cur.execute("SELECT * FROM blog_posts WHERE id=%s LIMIT 1", (post_id,))
        else:
            cur.execute("SELECT * FROM blog_posts WHERE slug=%s LIMIT 1", (slug,))
        row = cur.fetchone()
        if not row:
            return None
        if not include_unpublished and not row.get("published"):
            return None
        return row
    finally:
        cur.close()
        conn.close()


# ── Admin writes ────────────────────────────────────────────────────────────
def _unique_slug(cur, base: str, *, ignore_id: Optional[int] = None) -> str:
    candidate = base
    suffix = 1
    while True:
        if ignore_id:
            cur.execute(
                "SELECT id FROM blog_posts WHERE slug=%s AND id<>%s LIMIT 1",
                (candidate, ignore_id),
            )
        else:
            cur.execute("SELECT id FROM blog_posts WHERE slug=%s LIMIT 1", (candidate,))
        if not cur.fetchone():
            return candidate
        suffix += 1
        candidate = f"{base}-{suffix}"


def create_post(
    *,
    author_id: int,
    author_name: Optional[str],
    title: str,
    content: str,
    excerpt: Optional[str] = None,
    image_url: Optional[str] = None,
    tags: Optional[str] = None,
    slug: Optional[str] = None,
    published: bool = True,

    # ✅ NEW
    meta_title: Optional[str] = None,
    meta_description: Optional[str] = None,
    meta_keywords: Optional[str] = None,
    canonical_url: Optional[str] = None,
    index_status: str = "index",
) -> Dict[str, Any]:
    title = (title or "").strip()
    content = (content or "").strip()
    if len(title) < 3:
        raise ValueError("Title must be at least 3 characters")
    if len(content) < 10:
        raise ValueError("Content must be at least 10 characters")

    base_slug = slugify(slug) if slug else slugify(title)
    excerpt = (excerpt or "").strip()[:500] or None
    image_url = (image_url or "").strip()[:512] or None
    tags = (tags or "").strip()[:255] or None

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        unique = _unique_slug(cur, base_slug)
        cur.execute(
            """
            INSERT INTO blog_posts
            (slug, title, excerpt, content, image_url, tags,
            author_id, author_name, published,
            meta_title, meta_description, meta_keywords,
            canonical_url, index_status)
            VALUES (%s, %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s)
            """,
            (
                unique, title, excerpt, content, image_url, tags,
                author_id, author_name, 1 if published else 0,

                meta_title, meta_description, meta_keywords,
                canonical_url, index_status
            ),
        )
        conn.commit()
        new_id = cur.lastrowid
        cur.execute("SELECT * FROM blog_posts WHERE id=%s", (new_id,))
        return cur.fetchone()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def update_post(post_id: int, **fields) -> Optional[Dict[str, Any]]:
    allowed = (
        "title", "excerpt", "content", "image_url", "tags",
        "slug", "published",
        "meta_title", "meta_description", "meta_keywords",
        "canonical_url", "index_status"
    )
    sets: List[str] = []
    args: List[Any] = []

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        # If slug is changing, dedupe.
        if "slug" in fields and fields["slug"]:
            base = slugify(fields["slug"])
            fields["slug"] = _unique_slug(cur, base, ignore_id=post_id)

        if "published" in fields:
            fields["published"] = 1 if fields["published"] else 0

        for k in allowed:
            if k in fields:
                sets.append(f"`{k}`=%s")
                args.append(fields[k])

        if not sets:
            cur.execute("SELECT * FROM blog_posts WHERE id=%s", (post_id,))
            return cur.fetchone()

        args.append(post_id)
        cur.execute(
            f"UPDATE blog_posts SET {', '.join(sets)} WHERE id=%s",
            tuple(args),
        )
        conn.commit()
        cur.execute("SELECT * FROM blog_posts WHERE id=%s", (post_id,))
        return cur.fetchone()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def delete_post(post_id: int) -> bool:
    conn = get_mysql_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM blog_posts WHERE id=%s", (post_id,))
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
