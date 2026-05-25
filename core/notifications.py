"""Notification system for immune system alerts.

Supports Slack webhook and generic webhook notifications
for escalation events and benchmark results.
"""

import json
import os
from typing import Any

import requests

from core.config import get as cfg
from core.logger import setup_logger

logger = setup_logger("notifications")

# ---------------------------------------------------------------------------
# Slack notifier
# ---------------------------------------------------------------------------
class SlackNotifier:
    """Send messages to Slack via Incoming Webhook."""

    def __init__(self, webhook_url: str | None = None):
        self._webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")

    def _available(self) -> bool:
        return bool(self._webhook_url)

    def send_message(self, text: str, blocks: list[dict] | None = None) -> bool:
        """Send a plain text or block-kit message to Slack. Returns success."""
        if not self._available():
            logger.debug("Slack notifier skipped: no webhook URL configured")
            return False
        payload: dict[str, Any] = {"text": text}
        if blocks:
            payload["blocks"] = blocks
        try:
            resp = requests.post(
                self._webhook_url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            logger.info("Slack notification sent successfully")
            return True
        except requests.RequestException as e:
            logger.warning("Slack notification failed: %s", e)
            return False

    def send_escalation(self, report: dict) -> bool:
        """Send an escalation report as a formatted Slack message."""
        if not self._available():
            return False
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🚨 Immune System Escalation"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Consecutive Failures:*\n{report.get('consecutive_failures', '?')}"},
                    {"type": "mrkdwn", "text": f"*Threshold:*\n{report.get('threshold', '?')}"},
                ],
            },
            {"type": "divider"},
        ]
        for entry in report.get("history", []):
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Query:* {entry.get('query', 'N/A')[:100]}\n*Anomaly:* {entry.get('anomaly', 'N/A')[:100]}"},
            })
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"Generated: {report.get('generated_at', 'unknown')}"}],
        })
        return self.send_message("Immune System Escalation Notice", blocks=blocks)

    def send_benchmark_report(self, stats: dict) -> bool:
        """Send benchmark results as a Slack message."""
        if not self._available():
            return False
        s = stats.get("stats", stats)
        detection = stats.get("detection_rate_pct", 0)
        antibody_rate = stats.get("antibody_rate_pct", 0)
        text = (
            f"📊 *Adversarial Benchmark Complete*\n"
            f"• Detection Rate: {detection}%\n"
            f"• Antibody Rate: {antibody_rate}%\n"
            f"• Total Tests: {s.get('total', 0)}\n"
            f"• Immune Activated: {s.get('immune_activated', 0)}"
        )
        return self.send_message(text)


# ---------------------------------------------------------------------------
# Generic webhook notifier
# ---------------------------------------------------------------------------
class WebhookNotifier:
    """Send JSON payloads to any HTTP endpoint."""

    def __init__(self, url: str | None = None):
        self._url = url or os.getenv("NOTIFICATION_WEBHOOK_URL")

    def _available(self) -> bool:
        return bool(self._url)

    def send(self, payload: dict) -> bool:
        if not self._available():
            logger.debug("Webhook notifier skipped: no URL configured")
            return False
        try:
            resp = requests.post(self._url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info("Webhook notification sent")
            return True
        except requests.RequestException as e:
            logger.warning("Webhook notification failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# Unified notification manager
# ---------------------------------------------------------------------------
class NotificationManager:
    """Route notifications to all configured channels."""

    def __init__(self):
        self._slack = SlackNotifier()
        self._webhook = WebhookNotifier()

    @property
    def slack(self) -> SlackNotifier:
        return self._slack

    @property
    def webhook(self) -> WebhookNotifier:
        return self._webhook

    def notify_escalation(self, report: dict) -> None:
        """Send escalation to all configured channels."""
        self._slack.send_escalation(report)
        self._webhook.send({
            "event": "escalation",
            "severity": "critical",
            "report": report,
        })

    def notify_benchmark(self, report: dict) -> None:
        """Send benchmark results to all configured channels."""
        self._slack.send_benchmark_report(report)
        self._webhook.send({
            "event": "benchmark",
            "data": report,
        })

    def notify(self, title: str, message: str, severity: str = "info") -> None:
        """Send a generic notification to all channels."""
        self._slack.send_message(f"[{severity.upper()}] {title}\n{message}")
        self._webhook.send({
            "event": "notification",
            "title": title,
            "message": message,
            "severity": severity,
        })


# Global singleton
notifier = NotificationManager()
