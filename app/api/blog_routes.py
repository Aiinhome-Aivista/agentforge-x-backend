"""Blog routes.

Mounted under /api/blog via the main api blueprint. Public endpoints don't
require auth. Admin endpoints require ``is_admin=1`` on the user row.
"""

import logging
import base64
import os
import uuid

from flask import Blueprint, request, jsonify, g

from app.services import blog_service
from app.services.auth_helpers import require_auth
from app.services.admin_helpers import require_admin

logger = logging.getLogger(__name__)
blog_bp = Blueprint("blog", __name__, url_prefix="/blog")

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "blog"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _ok(message="OK", **extra):
    return jsonify({"status": True, "statuscode": 200, "message": message, **extra})


def _err(message, code=400, **extra):
    return jsonify({"status": False, "statuscode": code, "message": message, **extra}), code


# ── Public: list posts ──────────────────────────────────────────────────────
@blog_bp.get("")
def list_blog_posts():
    try:
        limit  = int(request.args.get("limit") or 50)
        offset = int(request.args.get("offset") or 0)
    except ValueError:
        return _err("limit and offset must be integers")
    try:
        items = blog_service.list_posts(limit=limit, offset=offset)
        # Use request.host_url which includes scheme and host (e.g. http://127.0.0.1:3004/)
        base_url = request.host_url
        
        for item in items:
            if item.get("image_url") and str(item["image_url"]).startswith("/"):
                # Use a simple string concatenation or urljoin
                path = item["image_url"].lstrip("/")
                item["image_url"] = f"{base_url}{path}"
            
    except Exception as e:
        logger.error("list_blog_posts failed: %s", e, exc_info=True)
        return _err("Could not load blog posts", 500)
    return _ok("OK", data=items)


# ── Public: get one (by slug) ───────────────────────────────────────────────
@blog_bp.get("/<slug>")
def get_blog_post(slug):
    if (slug or "").isdigit():
        post = blog_service.get_post(post_id=int(slug))
    else:
        post = blog_service.get_post(slug=slug)
    if not post:
        return _err("Post not found", 404)
        
    if post.get("image_url") and str(post["image_url"]).startswith("/"):
        base_url = request.host_url
        path = post["image_url"].lstrip("/")
        post["image_url"] = f"{base_url}{path}"
        
    return _ok("OK", data=post)


# ── Admin: create ───────────────────────────────────────────────────────────
# @blog_bp.post("")
# @require_auth
# @require_admin
# def create_blog_post():
#     body = request.get_json(silent=True) or {}
#     try:
#         index_status = body.get("index_status")
#         if index_status not in ["index", "noindex"]:
#             index_status = "index"

#         image_data = body.get("image_binary")
#         image = base64.b64decode(image_data) if image_data else None
#         post = blog_service.create_post(
#             author_id        = g.user["uid"],
#             author_name      = g.user.get("email"),

#             title            = body.get("title", ""),
#             content          = body.get("content", ""),
#             excerpt          = body.get("excerpt"),

#             cover_url        = body.get("cover_url"), 

#             tags             = ",".join(body.get("tags", [])) if isinstance(body.get("tags"), list) else body.get("tags"),

#             slug             = body.get("slug"),
#             published        = bool(body.get("published", True)),

#             meta_title       = body.get("meta_title"),
#             meta_description = body.get("meta_description"),
#             meta_keywords    = body.get("meta_keywords"),
#             canonical_url    = body.get("canonical_url"),
#             index_status     = index_status,
#             image            = image 
#         )
#     except ValueError as ve:
#         return _err(str(ve), 400)
#     except Exception as e:
#         logger.error("create_blog_post failed: %s", e, exc_info=True)
#         return _err("Could not create blog post", 500)
#     return _ok("Blog post created", data=post)

@blog_bp.post("")
@require_auth
@require_admin
def create_blog_post():
    try:
        form = request.form
        files = request.files

        # ✅ index_status validation
        index_status = form.get("index_status")
        if index_status not in ["index", "noindex"]:
            index_status = "index"

        # ✅ robust image extraction
        image_url = form.get("image_url") or form.get("image")

        # 1. Check request.files
        image_file = files.get("image") or files.get("image_url") or files.get("file")
        if not image_file and files:
            image_file = next(iter(files.values()))

        if image_file and image_file.filename:
            filename = image_file.filename
            
            ext = os.path.splitext(filename)[1] or ".png"
            unique_name = f"{uuid.uuid4().hex}{ext}"
            save_path = os.path.join(UPLOAD_FOLDER, unique_name)
            
            # Use absolute path to ensure we are saving in the right place
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Read bytes and write manually to avoid any stream issues
            image_bytes = image_file.read()
            if not image_bytes:
                logger.error("Uploaded image file is empty")
            else:
                with open(save_path, "wb") as f:
                    f.write(image_bytes)
                
                if os.path.exists(save_path):
                    logger.info(f"Successfully saved image to {save_path}")
                    image_url = f"/uploads/blog/{unique_name}"
                else:
                    logger.error(f"File system failed to persist {save_path}")

        # ✅ tags handling
        tags = request.form.getlist("tags")
        if not tags and form.get("tags"):
            tags = [form.get("tags")]
        tags_str = ",".join(tags) if tags else None

        post = blog_service.create_post(
            author_id        = g.user["uid"],
            author_name      = g.user.get("name"),

            title            = form.get("title", ""),
            content          = form.get("content", ""),
            excerpt          = form.get("excerpt"),

            image_url        = image_url,

            tags             = tags_str,

            slug             = form.get("slug"),
            published        = form.get("published", "1") in ["1", "true", "True"],

            meta_title       = form.get("meta_title"),
            meta_description = form.get("meta_description"),
            meta_keywords    = form.get("meta_keywords"),
            canonical_url    = form.get("canonical_url"),
            index_status     = index_status
        )

    except Exception as e:
        logger.error("create_blog_post failed: %s", e, exc_info=True)
        return _err("Could not create blog post", 500)

    if post and post.get("image_url") and post["image_url"].startswith("/"):
        base_url = request.host_url
        path = post["image_url"].lstrip("/")
        post["image_url"] = f"{base_url}{path}"

    return _ok("Blog post created", data=post)


# ── Admin: update ───────────────────────────────────────────────────────────
@blog_bp.patch("/<int:post_id>")
@require_auth
@require_admin
def update_blog_post(post_id):
    try:
        form = request.form
        files = request.files

        body = {}

        # ✅ index_status validation
        index_status = form.get("index_status")
        if index_status in ["index", "noindex"]:
            body["index_status"] = index_status

        # ✅ basic fields
        for field in [
            "title", "content", "excerpt", "slug",
            "meta_title", "meta_description",
            "meta_keywords", "canonical_url"
        ]:
            if form.get(field) is not None:
                body[field] = form.get(field)

        # ✅ published handling
        if form.get("published") is not None:
            body["published"] = form.get("published") in ["1", "true", "True"]

        # ✅ tags handling (same as create)
        tags = form.getlist("tags")
        if not tags and form.get("tags"):
            tags = [form.get("tags")]
        if tags:
            body["tags"] = ",".join(tags)

        # ✅ image handling (same logic as create)
        image_url = form.get("image_url") or form.get("image")

        image_file = files.get("image") or files.get("image_url") or files.get("file")
        if not image_file and files:
            image_file = next(iter(files.values()))

        if image_file and image_file.filename:
            filename = image_file.filename
            ext = os.path.splitext(filename)[1] or ".png"
            unique_name = f"{uuid.uuid4().hex}{ext}"

            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            save_path = os.path.join(UPLOAD_FOLDER, unique_name)

            image_bytes = image_file.read()
            if not image_bytes:
                logger.error("Uploaded image file is empty")
            else:
                with open(save_path, "wb") as f:
                    f.write(image_bytes)

                if os.path.exists(save_path):
                    logger.info(f"Saved image: {save_path}")
                    image_url = f"/uploads/blog/{unique_name}"
                else:
                    logger.error(f"Failed saving: {save_path}")

        if image_url:
            body["image_url"] = image_url

        # ✅ call service
        post = blog_service.update_post(post_id, **body)

    except Exception as e:
        logger.error("update_blog_post failed: %s", e, exc_info=True)
        return _err("Could not update blog post", 500)

    if not post:
        return _err("Post not found", 404)

    # ✅ convert relative → absolute URL
    if post.get("image_url") and post["image_url"].startswith("/"):
        base_url = request.host_url
        post["image_url"] = f"{base_url}{post['image_url'].lstrip('/')}"

    return _ok("Blog post updated", data=post)


# ── Admin: delete ───────────────────────────────────────────────────────────
@blog_bp.delete("/<int:post_id>")
@require_auth
@require_admin
def delete_blog_post(post_id):
    try:
        ok = blog_service.delete_post(post_id)
    except Exception as e:
        logger.error("delete_blog_post failed: %s", e, exc_info=True)
        return _err("Could not delete blog post", 500)
    if not ok:
        return _err("Post not found", 404)
    return _ok("Blog post deleted", data={"id": post_id})
