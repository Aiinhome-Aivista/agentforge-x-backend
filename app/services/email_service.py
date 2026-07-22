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

def send_welcome_email(to_email: str, name: str, password: str, file_limit_mb: int) -> bool:
    """Send a welcome email to a newly created user from the admin panel."""
    subject = "Welcome to AgentForgeX"
    import time
    unique_id = str(time.time())
    
    text_body = (
        f"Welcome to AgentForgeX, {name}!\n\n"
        f"Your account has been successfully created. Here are your details:\n"
        f"Email: {to_email}\n"
        f"Password: {password}\n"
        f"File Limit: {file_limit_mb}MB\n\n"
        f"How to login:\n"
        f"1. Go to https://agentforge.services/agentforgex/\n"
        f"2. Enter your email and password.\n"
        f"3. Change your password for better security.\n\n"
        f"— AgentForgeX\nRef: {unique_id}"
    )
    html_body = f"""<!doctype html>
<html><body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:#0a0a0a;color:#fff;margin:0;padding:40px 20px;">
  <div style="max-width:540px;margin:0 auto;background:#131313;border:1px solid #262626;
              border-radius:16px;padding:40px;box-shadow: 0 10px 30px rgba(0,0,0,0.8);">
    
    <!-- Logo area matching login page -->
    <div style="text-align:center;margin-bottom:32px;">
      <div style="width:60px;height:45px;padding-top:15px;background:#10d08c;border-radius:16px;
                  text-align:center;margin:0 auto 24px;box-shadow: 0 8px 20px rgba(16, 208, 140, 0.35);">
        <img src="https://api.iconify.design/ph:lightning-fill.svg?color=black" width="30" height="30" alt="Zap" style="display:inline-block;margin:0 auto;" />
      </div>
      
      <h2 style="margin:0 0 8px;font-size:28px;font-weight:900;color:#fff;letter-spacing:-.01em;">
        Welcome to <span style="color:#10b981;">AgentForgeX</span>
      </h2>
      <p style="color:#a3a3a3;margin:0;font-size:14px;line-height:1.6;">
        Hey {name} 🎉, your account has been successfully created and your workspace is ready.
      </p>
    </div>

    <!-- Credentials Block -->
    <div style="background:#0a0a0a;border:1px solid #262626;border-radius:12px;padding:24px;margin-bottom:32px;">
        <h3 style="margin:0 0 20px;font-size:12px;text-transform:uppercase;letter-spacing:1.5px;color:#737373;font-weight:700;">Account Credentials</h3>
        
        <div style="margin-bottom:20px;">
            <div style="font-size:13px;color:#a3a3a3;margin-bottom:6px;">Email Address</div>
            <div style="background:#ffffff;border:1px solid #e5e5e5;border-radius:8px;padding:14px 16px;
                        font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
                        font-size:15px;color:#000000;user-select:all;font-weight:600;">
                {to_email}
            </div>
        </div>

        <div style="margin-bottom:20px;">
            <div style="font-size:13px;color:#a3a3a3;margin-bottom:6px;">Password</div>
            <div style="background:#ffffff;border:1px solid #10b981;border-radius:8px;padding:14px 16px;
                        font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
                        font-size:16px;color:#000000;font-weight:700;letter-spacing:1px;user-select:all;">
                {password}
            </div>
            <div style="font-size:12px;color:#737373;margin-top:8px;">* Tip: You can easily highlight and copy the details above.</div>
        </div>

        <div style="border-top:1px solid #262626;padding-top:16px;display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:14px;color:#a3a3a3;">Assigned File Limit</span>
            <span style="background:#10b98120;color:#10b981;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:700;">{file_limit_mb} MB</span>
        </div>
    </div>

    <!-- Instructions -->
    <div style="margin-bottom:36px;">
        <h3 style="margin:0 0 16px;font-size:16px;color:#fff;font-weight:600;">Next Steps:</h3>
        <ul style="color:#a3a3a3;margin:0;padding-left:24px;font-size:15px;line-height:1.7;">
            <li style="margin-bottom:8px;">Click the login button below to access the portal.</li>
            <li style="margin-bottom:8px;">Enter your email and the password.</li>
            <li style="margin-bottom:8px;">We recommend updating your password after logging in.</li>
        </ul>
    </div>

    <!-- CTA Button -->
    <a href="https://agentforge.services/agentforgex/" 
       style="display:block;width:100%;text-align:center;background:#10b981;color:#000;
              padding:16px 0;border-radius:10px;font-weight:800;text-decoration:none;
              font-size:16px;letter-spacing:0.5px;box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);">
        Access Your Workspace
    </a>
  </div>
  
  <div style="text-align:center;margin-top:32px;color:#525252;font-size:12px;">
    &copy; 2026 AgentForgeX. All rights reserved.<br>
    If you did not request this account, please ignore this email.
    <div style="display:none;color:transparent;opacity:0;font-size:0px;line-height:0;">Ref: {unique_id}</div>
  </div>
</body></html>"""

    if not _smtp_configured():
        logger.warning("[DEV-EMAIL] Welcome email for %s", to_email)
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
        logger.info("Welcome email dispatched to %s", to_email)
        return True
    except Exception as e:
        logger.error("SMTP send failed for %s: %s", to_email, e)
        return False
