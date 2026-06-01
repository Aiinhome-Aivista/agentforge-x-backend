"""Captcha service — stateless, signed math challenges.

Design goals:
  * No third-party dependency (uses only stdlib + the existing PyJWT secret).
  * No DB row per challenge — the challenge ID is an HMAC-signed token that
    encodes the answer + expiry. This means the service scales horizontally
    and survives Flask worker restarts without losing state.
  * No image rendering — we render the prompt as **text** ("What is 7 + 4?")
    so that we don't introduce Pillow / Cairo as a dependency. The frontend
    formats it nicely with CSS.
  * Per-token single-use: once verified, the token is added to a tiny
    in-process replay-protection set (capped). This is best-effort; the
    primary defence is the short TTL.

Token shape (URL-safe base64):
    base64( "<answer>|<expires_unix>".encode() ).decode().rstrip("=")
    "." + base64(hmac_sha256(secret, body).digest()).rstrip("=")

The frontend never sees the answer — only the prompt text and the token.
On submit, the user sends both the typed answer and the token; the server
recomputes the HMAC, checks expiry + replay, and compares answers.
"""

from __future__ import annotations

import os
import hmac
import time
import base64
import hashlib
import logging
import secrets
from collections import deque
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

CAPTCHA_TTL_SECONDS = int(os.getenv("CAPTCHA_TTL_SECONDS", "300"))   # 5 min
CAPTCHA_REPLAY_CACHE = 4096                                          # last N tokens
CAPTCHA_SECRET_ENV = "CAPTCHA_SECRET"                                # falls back to JWT secret

# In-process replay-protection ring buffer. Best-effort only.
_replay: deque = deque(maxlen=CAPTCHA_REPLAY_CACHE)
_replay_set: set = set()


def _secret() -> bytes:
    """Pick a secret key for HMAC. Prefers CAPTCHA_SECRET, falls back to the
    same JWT secret the rest of the app uses, finally falls back to a
    process-stable random one (warns on use)."""
    s = os.getenv(CAPTCHA_SECRET_ENV) or os.getenv("FLASK_SECRET_KEY") or os.getenv("JWT_SECRET")
    if not s:
        global _runtime_secret
        try:
            return _runtime_secret  # type: ignore[name-defined]
        except NameError:
            logger.warning(
                "No CAPTCHA_SECRET / FLASK_SECRET_KEY / JWT_SECRET set — "
                "generating a process-local fallback. Set one in env for prod."
            )
            _runtime_secret = secrets.token_bytes(32)
            return _runtime_secret
    return s.encode("utf-8") if isinstance(s, str) else s


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _b64u_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign(body: bytes) -> str:
    sig = hmac.new(_secret(), body, hashlib.sha256).digest()
    return _b64u(sig)


# ── Public API ──────────────────────────────────────────────────────────────
def issue() -> Dict[str, str]:
    """Generate a fresh challenge.

    Returns:
        { "prompt": "What is 7 + 4?", "token": "<signed token>",
          "ttl_seconds": 300 }
    """
    a = secrets.randbelow(9) + 1   # 1..9
    b = secrets.randbelow(9) + 1
    op = secrets.choice(["+", "-", "×"])
    if op == "+":
        ans = a + b
    elif op == "-":
        # Make sure the result is non-negative for friendliness
        if a < b:
            a, b = b, a
        ans = a - b
    else:  # ×
        ans = a * b

    expires = int(time.time()) + CAPTCHA_TTL_SECONDS
    body = f"{ans}|{expires}".encode("utf-8")
    token = f"{_b64u(body)}.{_sign(body)}"
    return {
        "prompt":      f"What is {a} {op} {b}?",
        "token":       token,
        "ttl_seconds": CAPTCHA_TTL_SECONDS,
    }


class CaptchaError(Exception):
    """Raised by ``verify`` (when ``raise_on_failure=True``) for any
    invalid / expired / wrong-answer token."""


def verify(token: str, answer: str, *, raise_on_failure: bool = False) -> bool:
    """Validate a (token, answer) pair. Returns True/False (or raises when
    raise_on_failure=True). Also marks the token as consumed on success
    so it cannot be replayed within the in-process cache window."""

    def _fail(reason: str) -> bool:
        if raise_on_failure:
            raise CaptchaError(reason)
        return False

    if not token or not answer:
        return _fail("Captcha is required")
    answer = str(answer).strip()
    if not answer:
        return _fail("Captcha answer is required")

    # Disabled-mode escape hatch for QA / unit tests. Off by default.
    if os.getenv("CAPTCHA_DISABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        return True

    if token in _replay_set:
        return _fail("This captcha has already been used. Please reload.")

    try:
        body_b64, sig = token.split(".", 1)
    except ValueError:
        return _fail("Malformed captcha token")

    expected_sig = _sign(_b64u_decode(body_b64))
    if not hmac.compare_digest(expected_sig, sig):
        return _fail("Captcha verification failed")

    try:
        body = _b64u_decode(body_b64).decode("utf-8")
        ans_str, exp_str = body.split("|", 1)
        expires = int(exp_str)
    except (ValueError, UnicodeDecodeError):
        return _fail("Malformed captcha payload")

    if int(time.time()) > expires:
        return _fail("Captcha expired — please try again")

    if str(answer) != str(ans_str):
        return _fail("Incorrect captcha answer")

    # Mark consumed.
    if len(_replay) == _replay.maxlen:
        # Pop the oldest from the set when the deque rolls over.
        oldest = _replay[0]
        _replay_set.discard(oldest)
    _replay.append(token)
    _replay_set.add(token)
    return True


def is_disabled() -> bool:
    return os.getenv("CAPTCHA_DISABLED", "").strip().lower() in ("1", "true", "yes", "on")
