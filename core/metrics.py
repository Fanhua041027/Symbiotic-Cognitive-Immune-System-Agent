"""Metrics tracking for the immune system agent.

Tracks query success/failure rates, anomaly patterns, latency statistics,
and immune response effectiveness.
"""

import json
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
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
    success: bool  # whether final_output was produced


class MetricsTracker:
    """Thread-safe metrics tracking for the immune system."""

    def __init__(self, window_size: int = 1000):
        self._lock = threading.Lock()
        self._window_size = window_size
        self._records: list[QueryRecord] = []
        self._anomaly_counter: Counter[str] = Counter()
        self. _session_start = time.time()
        self._total_duration = 0.0

    def record_query(self, result: dict) -> QueryRecord:
        """Record the result of a single query execution."""
        anomaly_sources = [
            a.get("source", "unknown") for a in (result.get("anomalies") or [])
        ]
        has_anomaly = len(anomaly_sources) > 0

        record = QueryRecord(
            timestamp=time.time(),
            query_prefix=(result.get("user_query", "") or "")[:60],
            duration=0.0,  # filled externally
            has_anomaly=has_anomaly,
            anomaly_sources=anomaly_sources,
            antibody_count=len(result.get("antibodies") or []),
            immune_activated=result.get("is_immune_active", False),
            validation_status=result.get("validation_status"),
            escalation=result.get("escalation_report") is not None,
            success=result.get("final_output") is not None,
        )

        with self._lock:
            self._records.append(record)
            if len(self._records) > self._window_size:
                self._records.pop(0)
            self._total_duration += record.duration
            for src in anomaly_sources:
                self._anomaly_counter[src] += 1

        logger.debug("Metrics recorded: anomaly=%s, immune=%s, antibodies=%d",
                      has_anomaly, record.immune_activated, record.antibody_count)
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
            p95_duration = sorted(durations)[int(len(durations) * 0.95)] if durations else 0

            session_duration = time.time() - self._session_start

            anomaly_sources = Counter()
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
                "immune_activation_rate": round(immune_ok / total * 100, 1) if total > 0 else 0,
                "escalation_rate": round(escalations / total * 100, 1) if total > 0 else 0,
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

    def save_report(self, filename: str | None = None) -> str:
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
                for r in (self._records[-20:] if self._records else [])
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Metrics report saved: %s", path)
        return path


# Global singleton
metrics = MetricsTracker()
