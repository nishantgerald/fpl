"""Outbound email, or a no-op when none is configured.

Deliberately pluggable and deliberately silent by default. Most deployments of
this app — including every developer machine and every test run — have no SMTP
credentials, and an unconfigured mailer must not be an error condition: it
should record what it would have sent and carry on. The alternative is a
scheduled job that dies on the first machine that isn't production.

No third-party SDK. SMTP over TLS is in the standard library, works with Gmail
app passwords, Fastmail, Postmark, SES and anything else, and adds no dependency
to a project that is otherwise four packages deep.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import NamedTuple


class Delivery(NamedTuple):
    """What happened to one message. ``sent=False`` is routine, not a failure."""

    sent: bool
    reason: str


def is_configured() -> bool:
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_FROM"))


def _config() -> dict:
    return {
        "host": os.getenv("SMTP_HOST", "").strip(),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER", "").strip(),
        "password": os.getenv("SMTP_PASSWORD", "").strip(),
        "sender": os.getenv("SMTP_FROM", "").strip(),
        "reply_to": os.getenv("SMTP_REPLY_TO", "").strip(),
    }


def send(to: str, subject: str, text: str, html: str | None = None) -> Delivery:
    """Send one message. Never raises — the caller is usually a scheduled job.

    A send failure must not abort a run partway through a mailing list, leaving
    half the users notified and no record of which half.
    """
    if not is_configured():
        return Delivery(False, "smtp_not_configured")
    if not to or "@" not in to:
        return Delivery(False, "invalid_recipient")

    cfg = _config()
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = cfg["sender"]
    message["To"] = to
    if cfg["reply_to"]:
        message["Reply-To"] = cfg["reply_to"]
    message.set_content(text)
    if html:
        message.add_alternative(html, subtype="html")

    try:
        context = ssl.create_default_context()
        if cfg["port"] == 465:
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=context,
                                  timeout=20) as server:
                if cfg["user"]:
                    server.login(cfg["user"], cfg["password"])
                server.send_message(message)
        else:
            with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as server:
                server.starttls(context=context)
                if cfg["user"]:
                    server.login(cfg["user"], cfg["password"])
                server.send_message(message)
    except Exception as error:  # noqa: BLE001 - reported, never raised
        return Delivery(False, f"{type(error).__name__}: {error}")

    return Delivery(True, "sent")
