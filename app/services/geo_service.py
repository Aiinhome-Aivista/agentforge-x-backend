"""Detect the caller's country from their IP / request headers.

Resolution chain (first hit wins):
  1. ``Cf-Ipcountry`` — set by Cloudflare when the site is behind their proxy
  2. ``X-Country-Code`` — useful if your nginx layer is doing the lookup
  3. Environment override ``GEOIP_DEFAULT`` — for testing / fallback
  4. HTTP lookup against a free GeoIP service (``ipapi.co`` by default)
  5. Hard fallback to ``IN`` (the product's home market)

Results are cached in-process for ``GEOIP_CACHE_TTL`` seconds (default 6h)
so we don't hammer the free tier of the lookup service.
"""

import os
import time
import logging
import ipaddress
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CACHE_TTL = int(os.getenv("GEOIP_CACHE_TTL", str(6 * 3600)))
LOOKUP_URL = os.getenv("GEOIP_LOOKUP_URL", "https://ipapi.co/{ip}/country/")
LOOKUP_TIMEOUT = float(os.getenv("GEOIP_LOOKUP_TIMEOUT", "2.0"))

_cache: dict[str, tuple[str, float]] = {}


def _is_private(ip: str) -> bool:
    """Treat localhost / RFC1918 / link-local as 'no public IP'."""
    try:
        a = ipaddress.ip_address(ip)
        return a.is_private or a.is_loopback or a.is_link_local or a.is_reserved
    except ValueError:
        return True


def get_client_ip(request) -> Optional[str]:
    """Best-effort extraction of the public client IP from a Flask request."""
    # Cloudflare puts the real IP here
    cf = request.headers.get("CF-Connecting-IP")
    if cf:
        return cf.strip()
    # nginx proxy chain — first entry is the client
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    real = request.headers.get("X-Real-IP")
    if real:
        return real.strip()
    return request.remote_addr


def _lookup(ip: str) -> Optional[str]:
    """Hit the geo provider for an IP and return a 2-letter country code."""
    now = time.time()
    cached = _cache.get(ip)
    if cached and (now - cached[1]) < CACHE_TTL:
        return cached[0]

    try:
        url = LOOKUP_URL.format(ip=ip)
        resp = requests.get(url, timeout=LOOKUP_TIMEOUT,
                            headers={"User-Agent": "AgentForgeX-Geo/1.0"})
        if resp.ok:
            code = (resp.text or "").strip().upper()
            # ipapi.co returns the 2-letter code as plain text on the
            # "/{ip}/country/" path; some providers return JSON like
            # {"country":"IN"} — handle both.
            if code.startswith("{"):
                try:
                    j = resp.json()
                    code = (j.get("country") or j.get("country_code") or "").upper()
                except Exception:
                    code = ""
            if len(code) == 2 and code.isalpha():
                _cache[ip] = (code, now)
                return code
            logger.debug("Geo lookup returned non-code body for %s: %r", ip, resp.text[:80])
        else:
            logger.debug("Geo lookup HTTP %s for %s", resp.status_code, ip)
    except Exception as e:
        logger.debug("Geo lookup failed for %s: %s", ip, e)
    return None


def detect_country(request, default: str = "IN") -> str:
    """Return a 2-letter ISO country code for the caller of ``request``."""
    # 1. Cloudflare header
    cf = request.headers.get("Cf-Ipcountry") or request.headers.get("CF-IPCountry")
    if cf and len(cf) == 2 and cf.isalpha():
        return cf.upper()
    # 2. nginx-injected header
    nx = request.headers.get("X-Country-Code")
    if nx and len(nx) == 2 and nx.isalpha():
        return nx.upper()
    # 3. Static override (useful in dev)
    env_default = os.getenv("GEOIP_DEFAULT")
    if env_default and len(env_default) == 2:
        return env_default.upper()
    # 4. HTTP lookup
    ip = get_client_ip(request)
    if ip and not _is_private(ip):
        code = _lookup(ip)
        if code:
            return code
    # 5. Hard fallback
    return default.upper()


def currency_for(country: str) -> str:
    """Currency rule that matches the pricing matrix (IN → INR, else USD)."""
    return "INR" if (country or "").upper() == "IN" else "USD"
