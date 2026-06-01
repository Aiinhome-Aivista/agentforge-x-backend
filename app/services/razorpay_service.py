"""Razorpay integration — order creation + signature verification.

Razorpay flow:
  1. Server creates an order via Orders API → returns order_id
  2. Client opens Razorpay Checkout with that order_id
  3. On success, Razorpay returns {order_id, payment_id, signature}
  4. Server verifies the signature: HMAC-SHA256(order_id|payment_id, secret)
  5. Server marks payment paid + activates the subscription

Currency rules applied here (matching the pricing matrix):
  - country == 'IN'  → INR  (price_inr * 100 paise)
  - else             → USD  (price_usd * 100 cents)
"""

import os
import hmac
import hashlib
import logging
from typing import Optional

import razorpay

logger = logging.getLogger(__name__)


def _client() -> Optional[razorpay.Client]:
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        logger.error("Razorpay credentials missing — payments disabled")
        return None
    return razorpay.Client(auth=(key_id, key_secret))


def is_configured() -> bool:
    return bool(os.getenv("RAZORPAY_KEY_ID") and os.getenv("RAZORPAY_KEY_SECRET"))


def public_key() -> Optional[str]:
    """The key_id is safe to ship to the SPA — Checkout needs it."""
    return os.getenv("RAZORPAY_KEY_ID")


def amount_in_minor(value: float) -> int:
    """Convert e.g. 500.00 → 50000 (Razorpay uses paise/cents)."""
    return int(round(float(value) * 100))


def create_order(amount_minor: int, currency: str = "INR",
                 receipt: Optional[str] = None,
                 notes: Optional[dict] = None) -> Optional[dict]:
    """Create a Razorpay order. Returns the API response dict or None."""
    cli = _client()
    if cli is None:
        return None
    try:
        return cli.order.create({
            "amount":          amount_minor,
            "currency":        currency,
            "receipt":         receipt or "",
            "notes":           notes or {},
            "payment_capture": 1,
        })
    except Exception as e:
        logger.error("Razorpay order create failed: %s", e, exc_info=True)
        return None


def verify_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """Verify the HMAC signature returned by Razorpay Checkout."""
    secret = os.getenv("RAZORPAY_KEY_SECRET")
    if not (order_id and payment_id and signature and secret):
        return False
    try:
        body = f"{order_id}|{payment_id}".encode("utf-8")
        expected = hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
    except Exception as e:
        logger.error("Razorpay signature verify failed: %s", e)
        return False
