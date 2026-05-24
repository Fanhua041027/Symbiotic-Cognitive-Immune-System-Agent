"""Metrics tracking for the immune system agent.

Tracks query success/failure rates, anomaly patterns, latency statistics,
and immune response effectiveness.
"""

import atexit
import glob
import json
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from core.logger import setup_logger

logger = setup_logger("metrics")

METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metrics")


@dataclass
class QueryRecord:
    """Record of a single query execution."""
    timestamp: float
    query_prefix: str  # first 60 chars
    duration: float
    has_anomaly: bool
    anomaly_sources: list[str]
    antibody_count: int
    immune_activated: bool
    validation_status: str | None
    escalation: bool
    request_id: str
    success: bool  # whether final_output was produced


class MetricsTracker:
    """Thread-safe metrics tracking for the immune system."""

    def __init__(self, window_size: int = 1000, auto_save_every: int = 10):
        self._lock = threading.Lock()
        self._window_size = window_size
        self._records: deque[QueryRecord] = deque(maxlen=window_size)
        self._anomaly_counter: Counter[str] = Counter()
        self._session_start = time.time()
        self._total_duration = 0.0
        self._auto_save_every = auto_save_every
        self._auto_save_counter = 0
        self._last_report_path: str | None = None
        os.makedirs(METRICS_DIR, exist_ok=True)
        atexit.register(self._auto_save_on_exit)

    def record_query(self, result: dict) -> QueryRecord:
        """Record the result of a single query execution."""
        anomaly_sources = [
            a.get("source", "unknown") for a in (result.get("anomalies") or [])
        ]
        has_anomaly = len(anomaly_sources) > 0

        record = QueryRecord(
            timestamp=time.time(),
            query_prefix=(result.get("user_query", "") or "")[:60],
            duration=result.get("duration", 0.0),
            has_anomaly=has_anomaly,
            anomaly_sources=anomaly_sources,
            antibody_count=len(result.get("antibodies") or []),
            immune_activated=result.get("is_immune_active", False),
            validation_status=result.get("validation_status"),
            escalation=result.get("escalation_report") is not None,
            request_id=result.get("request_id", "") or "",
            success=result.get("final_output") is not None,
        )

        with self._lock:
            self._records.append(record)
            self._total_duration += record.duration
            for src in anomaly_sources:
                self._anomaly_counter[src] += 1

        logger.debug("Metrics recorded: anomaly=%s, immune=%s, antibodies=%d",
                      has_anomaly, record.immune_activated, record.antibody_count)

        # Auto-save every N records
        self._auto_save_counter += 1
        if self._auto_save_counter >= self._auto_save_every:
            self._auto_save()
            self._auto_save_counter = 0

        return record

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of collected metrics."""
        with self._lock:
            total = len(self._records)
            if total == 0:
                return {"status": "no_data", "records": 0}

            anomalies = sum(1 for r in self._records if r.has_anomaly)
            immune_ok = sum(1 for r in self._records if r.immune_activated)
            escalations = sum(1 for r in self._records if r.escalation)
            successes = sum(1 for r in self._records if r.success)

            durations = [r.duration for r in self._records if r.duration > 0]
            avg_duration = sum(durations) / len(durations) if durations else 0
            max_duration = max(durations) if durations else 0
            p95_duration = (
                sorted(durations)[int(len(durations) * 0.95)] if durations else 0
            )

            session_duration = time.time() - self._session_start

            anomaly_sources: Counter[str] = Counter()
            for r in self._records:
                for src in r.anomaly_sources:
                    anomaly_sources[src] += 1

            return {
                "status": "ok",
                "records": total,
                "session_duration_seconds": round(session_duration, 1),
                "success_rate": round(successes / total * 100, 1) if total > 0 else 0,
                "anomaly_rate": round(anomalies / total * 100, 1) if total > 0 else 0,
                "anomaly_breakdown": dict(anomaly_sources.most_common()),
                "immune_activation_rate": (
                    round(immune_ok / total * 100, 1) if total > 0 else 0
                ),
                "escalation_rate": (
                    round(escalations / total * 100, 1) if total > 0 else 0
                ),
                "avg_antibodies_per_query": round(
                    sum(r.antibody_count for r in self._records) / total, 2
                ),
                "latency": {
                    "avg_seconds": round(avg_duration, 2),
                    "p95_seconds": round(p95_duration, 2),
                    "max_seconds": round(max_duration, 2),
                },
                "total_llm_time_seconds": round(self._total_duration, 1),
            }

    def _auto_save(self) -> None:
        """Internal: save report with a rotating filename pattern."""
        try:
            self._last_report_path = self.save_report("_auto_recent.json")
        except Exception as e:
            logger.debug("Auto-save metrics failed: %s", e)

    def _auto_save_on_exit(self) -> None:
        """atexit hook: save final metrics report."""
        if not self._records:
            return
        try:
            self.save_report(
                f"metrics_final_{time.strftime('%Y%m%d_%H%M%S')}.json",
                quiet=True,
            )
        except Exception:
            pass

    @staticmethod
    def cleanup_old_reports(max_age_days: int = 7) -> int:
        """Remove metrics report files older than max_age_days. Returns count removed."""
        cutoff = time.time() - max_age_days * 86400
        pattern = os.path.join(METRICS_DIR, "*.json")
        removed = 0
        for fpath in glob.glob(pattern):
            try:
                if os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
                    removed += 1
            except OSError:
                pass
        if removed:
            logger.info("Cleaned up %d old metrics reports", removed)
        return removed

    @classmethod
    def load_latest_report(cls) -> dict[str, Any] | None:
        """Load the most recent auto-saved metrics report from disk."""
        pattern = os.path.join(METRICS_DIR, "*.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not files:
            return None
        try:
            with open(files[0], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug("Failed to load metrics report %s: %s", files[0], e)
            return None

    def save_report(self, filename: str | None = None, quiet: bool = False) -> str:
        """Save a metrics report to disk as JSON."""
        os.makedirs(METRICS_DIR, exist_ok=True)
        if filename is None:
            filename = f"metrics_{time.strftime('%Y%m%d_%H%M%S')}.json"

        path = os.path.join(METRICS_DIR, filename)
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": self.get_summary(),
            "recent_queries": [
                {
                    "time": time.strftime("%H:%M:%S", time.localtime(r.timestamp)),
                    "query": r.query_prefix,
                    "duration": round(r.duration, 2),
                    "anomaly": r.has_anomaly,
                    "immune": r.immune_activated,
                    "success": r.success,
                }
                for r in (list(self._records)[-20:] if self._records else [])
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        if not quiet:
            logger.info("Metrics report saved: %s", path)
        return path


# Global singleton
metrics = MetricsTracker()
# Clean up stale report artifacts from previous sessions
MetricsTracker.cleanup_old_reports(max_age_days=7)
