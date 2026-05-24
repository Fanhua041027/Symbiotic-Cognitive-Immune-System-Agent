"""Human feedback escalation system.

Tracks consecutive immune system failures and generates escalation
reports when the system cannot autonomously resolve anomalies.
"""

import glob
import json
import os
import time
from datetime import datetime, timezone

from core.config import get as cfg
from core.logger import setup_logger

logger = setup_logger("escalation")

ESCALATION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "escalations",
)


class EscalationTracker:
    """Tracks immune failures and triggers human escalation."""

    def __init__(self):
        self._consecutive_failures = 0
        self._history: list[dict] = []

    def record_failure(
        self,
        query: str,
        anomaly_reason: str,
        antibodies_generated: int,
    ) -> str | None:
        """
        Record an immune response failure.
        Returns escalation report path if threshold exceeded.
        """
        self._consecutive_failures += 1
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query[:200],
            "anomaly": anomaly_reason[:200],
            "antibodies_generated": antibodies_generated,
            "consecutive_failures": self._consecutive_failures,
        }
        self._history.append(entry)
        logger.warning(
            "Immune failure #%d: %s",
            self._consecutive_failures,
            anomaly_reason[:80],
        )

        threshold = cfg("ESCALATION_THRESHOLD", 3)
        if self._consecutive_failures >= threshold:
            return self._generate_report(threshold)
        return None

    def record_success(self):
        """Reset the failure counter on successful recovery."""
        if self._consecutive_failures > 0:
            logger.info(
                "Immune system recovered after %d failures",
                self._consecutive_failures,
            )
        self._consecutive_failures = 0

    def _generate_report(self, threshold: int = 3) -> str:
        """Write an escalation report to disk."""
        os.makedirs(ESCALATION_DIR, exist_ok=True)
        filename = (
            f"escalation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        path = os.path.join(ESCALATION_DIR, filename)

        report = {
            "title": "Immune System Escalation Notice",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "consecutive_failures": self._consecutive_failures,
            "threshold": threshold,
            "history": self._history[-threshold:],
            "action_required": (
                "The immune system has failed to autonomously resolve anomalies "
                "after multiple attempts. Manual intervention is required."
            ),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.error(
            "ESCALATION: %d consecutive failures. Report saved to %s",
            self._consecutive_failures,
            path,
        )
        self._consecutive_failures = 0
        return path

    def reset(self) -> None:
        """Reset consecutive failure counter and history.

        Call this between independent queries to prevent cross-query bleeding.
        """
        self._consecutive_failures = 0
        self._history.clear()
        logger.debug("Escalation tracker reset")

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @staticmethod
    def cleanup_old_reports(max_age_days: int = 30) -> int:
        """Remove escalation reports older than max_age_days. Returns count."""
        cutoff = time.time() - max_age_days * 86400
        pattern = os.path.join(ESCALATION_DIR, "*.json")
        removed = 0
        for fpath in glob.glob(pattern):
            try:
                if os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
                    removed += 1
            except OSError:
                pass
        if removed:
            logger.info("Cleaned up %d old escalation reports", removed)
        return removed


# Global singleton
escalation = EscalationTracker()
# Clean up stale escalation reports from previous sessions
EscalationTracker.cleanup_old_reports(max_age_days=30)
