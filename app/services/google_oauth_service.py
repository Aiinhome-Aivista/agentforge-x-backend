"""Verify a Google ID token issued to the SPA.

The frontend uses @react-oauth/google. After the user signs in, Google issues
an ID token (JWT) that the SPA POSTs to /api/auth/google. We verify it here
against Google's public keys and check that the audience matches our client ID.
"""

import os
import logging
from typing import Optional

from google.oauth2 import id_token
from google.auth.transport import requests as g_requests

logger = logging.getLogger(__name__)


def verify_google_id_token(token: str) -> Optional[dict]:
    """Verify a Google-issued ID token.

    Returns a dict like:
        {sub, email, email_verified, name, picture, given_name, family_name}
    or None if verification fails.
    """
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        logger.error("GOOGLE_CLIENT_ID not configured — Google sign-in disabled")
        return None

    try:
        info = id_token.verify_oauth2_token(
            token, g_requests.Request(), client_id
        )
        # Google issuers are accounts.google.com or https://accounts.google.com
        if info.get("iss") not in ("accounts.google.com",
                                    "https://accounts.google.com"):
            logger.warning("Unexpected Google ID-token issuer: %s", info.get("iss"))
            return None
        if not info.get("email_verified"):
            logger.warning("Google account email not verified for %s",
                           info.get("email"))
            return None
        return {
            "sub":            info.get("sub"),
            "email":          info.get("email"),
            "email_verified": bool(info.get("email_verified")),
            "name":           info.get("name") or info.get("email", "").split("@")[0],
            "picture":        info.get("picture"),
            "given_name":     info.get("given_name"),
            "family_name":    info.get("family_name"),
        }
    except ValueError as e:
        logger.warning("Google token verify failed: %s", e)
        return None
    except Exception as e:
        logger.error("Google token verify error: %s", e, exc_info=True)
        return None
