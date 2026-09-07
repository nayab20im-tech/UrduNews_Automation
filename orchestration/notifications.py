"""
orchestration/notifications.py
==============================
Implements Section 5's "Notification System" box: email / Telegram
alerts, upload-status alerts and error alerts.

Design rules (same spirit as the rest of the project):
- console/log always works, even fully offline;
- email (smtplib) and Telegram (plain urllib, no extra dependency) are
  optional and auto-disabled when unconfigured;
- a notification failure can NEVER break the pipeline -- every channel
  is wrapped and reported back as a per-channel bool.
"""

from __future__ import annotations

import json
import smtplib
import urllib.request
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

from data_acquisition.utils.logger import get_logger

from . import config as oc

logger = get_logger("orchestration.notifications")


class NotificationManager:
    """Fan-out alerts to log (always), email and Telegram (when configured)."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        on_error: Optional[bool] = None,
        on_complete: Optional[bool] = None,
    ):
        self.enabled = oc.NOTIFY_ENABLED if enabled is None else enabled
        self.on_error = oc.NOTIFY_ON_ERROR if on_error is None else on_error
        self.on_complete = oc.NOTIFY_ON_COMPLETE if on_complete is None else on_complete

    # -- channel availability ------------------------------------------------
    def email_configured(self) -> bool:
        return bool(oc.NOTIFY_EMAIL_HOST) and bool(oc.NOTIFY_EMAIL_TO)

    def telegram_configured(self) -> bool:
        from distribution import config as dc

        return dc.is_telegram_configured()

    # -- core send -------------------------------------------------------------
    def send(self, subject: str, message: str, level: str = "info") -> Dict[str, bool]:
        """Deliver one notification; returns per-channel success flags."""
        results = {"logged": True, "email": False, "telegram": False}

        if str(level).upper() == "ERROR":
            logger.error("%s | %s", subject, message)
        elif str(level).upper() == "WARNING":
            logger.warning("%s | %s", subject, message)
        else:
            logger.info("%s | %s", subject, message)

        if not self.enabled:
            return results

        if self.email_configured():
            results["email"] = self._send_email(subject, message)
        if self.telegram_configured():
            results["telegram"] = self._send_telegram(subject, message)
        return results

    # -- convenience alerts -------------------------------------------------------
    def notify_error(self, where: str, error: Any) -> Dict[str, bool]:
        if not self.on_error:
            return {"logged": True, "email": False, "telegram": False}
        return self.send(
            f"[UrduNewsAI] Error in {where}",
            str(error),
            level="error",
        )

    def notify_upload_status(self, job: Dict[str, Any]) -> Dict[str, bool]:
        """Alert for a Section 4 distribution job (4.1 upload status)."""
        headline = job.get("headline") or f"article {job.get('article_id')}"
        status = job.get("youtube_status") or job.get("status") or "unknown"
        return self.send(
            f"[UrduNewsAI] Upload {status}: {headline[:80]}",
            json.dumps(
                {k: job.get(k) for k in (
                    "article_id", "youtube_status", "facebook_status",
                    "twitter_status", "telegram_status", "whatsapp_status",
                )},
                ensure_ascii=False,
            ),
            level="info" if status in ("published", "mock_published") else "warning",
        )

    def notify_run_report(self, report: Dict[str, Any]) -> Dict[str, bool]:
        if not self.on_complete:
            return {"logged": True, "email": False, "telegram": False}
        return self.send(
            "[UrduNewsAI] Pipeline run complete",
            json.dumps(report, ensure_ascii=False, default=str),
            level="info",
        )

    # -- channels -------------------------------------------------------------
    def _send_email(self, subject: str, message: str) -> bool:
        try:
            msg = MIMEText(message, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = oc.NOTIFY_EMAIL_FROM
            msg["To"] = ", ".join(oc.NOTIFY_EMAIL_TO)
            with smtplib.SMTP(oc.NOTIFY_EMAIL_HOST, oc.NOTIFY_EMAIL_PORT, timeout=10) as server:
                if oc.NOTIFY_EMAIL_USER:
                    server.starttls()
                    server.login(oc.NOTIFY_EMAIL_USER, oc.NOTIFY_EMAIL_PASSWORD)
                server.sendmail(oc.NOTIFY_EMAIL_FROM, oc.NOTIFY_EMAIL_TO, msg.as_string())
            logger.info("Email notification sent to %s", oc.NOTIFY_EMAIL_TO)
            return True
        except Exception as exc:
            logger.warning("Email notification failed: %s", exc)
            return False

    def _send_telegram(self, subject: str, message: str) -> bool:
        from distribution import config as dc

        try:
            url = f"https://api.telegram.org/bot{dc.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = json.dumps(
                {"chat_id": dc.TELEGRAM_CHAT_ID, "text": f"{subject}\n{message}"}
            ).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as exc:
            logger.warning("Telegram notification failed: %s", exc)
            return False
