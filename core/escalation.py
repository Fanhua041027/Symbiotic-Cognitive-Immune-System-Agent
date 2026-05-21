"""Human feedback escalation system.

Tracks consecutive immune system failures and generates escalation
reports when the system cannot autonomously resolve anomalies.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from core.logger import setup_logger
from core.config import get as cfg

logger = setup_logger("escalation")

ESCALATION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "escalations"
)
MAX_FAILURES = cfg("ESCALATION_THRESHOLD", 3)


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
    ) -> Optional[str]:
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

        if self._consecutive_failures >= MAX_FAILURES:
            return self._generate_report()
        return None

    def record_success(self):
        """Reset the failure counter on successful recovery."""
        if self._consecutive_failures > 0:
            logger.info(
                "Immune system recovered after %d failures",
                self._consecutive_failures,
            )
        self._consecutive_failures = 0

    def _generate_report(self) -> str:
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
            "threshold": MAX_FAILURES,
            "history": self._history[-MAX_FAILURES:],
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

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures


# Global singleton
escalation = EscalationTracker()
