"""Email OTP delivery via SMTP.

Configure via .env:
    SMTP_HOST           e.g. smtp.gmail.com
    SMTP_PORT           e.g. 587
    SMTP_USERNAME       full email address
    SMTP_PASSWORD       app password
    SMTP_USE_TLS        true|false  (default true)
    SMTP_FROM_NAME      sender display name
    SMTP_FROM_EMAIL     defaults to SMTP_USERNAME

In DEV (no SMTP creds) the OTP is printed to the log and the API response
includes ``debug_otp`` so testing is unblocked. NEVER ship without SMTP.
"""

import os
import smtplib
import logging
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def _bool(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _smtp_configured() -> bool:
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_USERNAME")
                and os.getenv("SMTP_PASSWORD"))


def send_otp_email(to_email: str, code: str, purpose: str = "signup") -> bool:
    """Send a 6-digit OTP. Returns True on success.

    On failure (or when SMTP is not configured) returns False but does NOT
    raise — the caller decides what to do (typically: log and continue with
    a debug_otp echo in non-prod).
    """
    subject_map = {
        "signup": "Verify your AgentForgeX account",
        "login":  "AgentForgeX sign-in code",
        "reset":  "Reset your AgentForgeX password",
    }
    subject = subject_map.get(purpose, "Your AgentForgeX verification code")

    text_body = (
        f"Your AgentForgeX verification code is: {code}\n\n"
        f"This code expires in 10 minutes. If you didn't request it, "
        f"ignore this email.\n\n— AgentForgeX"
    )
    html_body = f"""<!doctype html>
<html><body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;
background:#0a0a0a;color:#fff;margin:0;padding:40px 0;">
  <div style="max-width:480px;margin:0 auto;background:#171717;border:1px solid #262626;
              border-radius:16px;padding:32px;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:24px;">
      <div style="width:32px;height:32px;background:#10b981;border-radius:8px;
                  display:flex;align-items:center;justify-content:center;
                  font-weight:900;color:#000;">A</div>
      <div style="font-weight:800;letter-spacing:-.01em;">AgentForgeX</div>
    </div>
    <h2 style="margin:0 0 12px;font-size:20px;">Your verification code</h2>
    <p style="color:#a3a3a3;margin:0 0 24px;font-size:14px;">
      Enter the code below to {('verify your account' if purpose=='signup'
                                 else 'finish signing in')}.
    </p>
    <div style="font-family:JetBrains Mono,Menlo,monospace;font-size:34px;
                font-weight:700;letter-spacing:8px;color:#10b981;text-align:center;
                background:#0a0a0a;border:1px solid #262626;border-radius:12px;
                padding:18px 0;margin:0 0 24px;">{code}</div>
    <p style="color:#737373;font-size:12px;margin:0;">
      This code expires in 10 minutes. If you didn't request it, you can ignore
      this email safely.
    </p>
  </div>
</body></html>"""

    if not _smtp_configured():
        # Dev fallback — log the code so the developer can proceed.
        logger.warning("[DEV-OTP] %s code for %s: %s (purpose=%s)",
                       purpose, to_email, code, purpose)
        return False

    msg = EmailMessage()
    from_email = os.getenv("SMTP_FROM_EMAIL") or os.getenv("SMTP_USERNAME")
    from_name  = os.getenv("SMTP_FROM_NAME", "AgentForgeX")
    msg["From"]    = f"{from_name} <{from_email}>"
    msg["To"]      = to_email
    msg["Subject"] = subject
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    use_tls = _bool(os.getenv("SMTP_USE_TLS", "true"), True)

    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=20) as s:
                s.login(os.getenv("SMTP_USERNAME"), os.getenv("SMTP_PASSWORD"))
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.ehlo()
                if use_tls:
                    s.starttls()
                    s.ehlo()
                s.login(os.getenv("SMTP_USERNAME"), os.getenv("SMTP_PASSWORD"))
                s.send_message(msg)
        logger.info("OTP email dispatched to %s (purpose=%s)", to_email, purpose)
        return True
    except Exception as e:
        logger.error("SMTP send failed for %s: %s", to_email, e)
        return False


def is_dev_mode() -> bool:
    """True when SMTP is not configured — caller may echo the OTP back."""
    return False
