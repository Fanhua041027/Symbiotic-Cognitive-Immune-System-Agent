"""Session manager for long-running self-healing agent.

Tracks conversation history, health metrics, and auto-recovery state
across multiple queries within a continuous session.
"""

import json
import os
import threading
import time
import uuid
from collections import deque
from typing import Any

from core.logger import setup_logger

logger = setup_logger("session")

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sessions")

# ---------------------------------------------------------------------------
# Health state per query turn
# ---------------------------------------------------------------------------
class TurnRecord:
    """Record of a single turn in the session."""
    __slots__ = (
        "timestamp", "query", "had_anomaly", "anomaly_count",
        "antibody_count", "immune_activated", "success", "duration",
    )

    def __init__(self, result: dict):
        self.timestamp = time.time()
        self.query = (result.get("user_query") or "")[:80]
        anomalies = result.get("anomalies") or []
        self.had_anomaly = bool(anomalies)
        self.anomaly_count = len(anomalies)
        self.antibody_count = len(result.get("antibodies") or [])
        self.immune_activated = result.get("is_immune_active", False)
        self.success = result.get("final_output") is not None
        self.duration = result.get("duration", 0.0)

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
class AgentSession:
    """Persistent session for a long-running self-healing agent."""

    def __init__(self, session_id: str | None = None, max_turns: int = 500):
        self._lock = threading.RLock()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._max_turns = max_turns
        self._turns: deque[TurnRecord] = deque(maxlen=max_turns)
        self._consecutive_failures = 0
        self._total_recoveries = 0
        self._start_time = time.time()
        self._last_activity = self._start_time
        self._recovery_events: list[dict] = []

    # -- Turn recording -----------------------------------------------------

    def record_turn(self, result: dict) -> None:
        """Record a single query execution result."""
        turn = TurnRecord(result)
        with self._lock:
            self._turns.append(turn)
            self._last_activity = time.time()
            if turn.success and self._consecutive_failures > 0:
                self._consecutive_failures = 0
                self._total_recoveries += 1
                self._recovery_events.append({
                    "timestamp": turn.timestamp,
                    "type": "auto_recovery",
                    "detail": f"Recovered after previous failure "
                              f"(session_id={self.session_id})",
                })
            elif not turn.success:
                self._consecutive_failures += 1

    def record_recovery_event(self, event_type: str, detail: str) -> None:
        """Record a manual recovery action."""
        with self._lock:
            self._recovery_events.append({
                "timestamp": time.time(),
                "type": event_type,
                "detail": detail,
            })

    # -- Health metrics -----------------------------------------------------

    def health_score(self, window: int = 20) -> float:
        """Compute health score 0.0-1.0 over the last N turns."""
        with self._lock:
            recent = list(self._turns)[-window:]
            if not recent:
                return 1.0
            anomaly_ratio = sum(1 for t in recent if t.had_anomaly) / len(recent)
            success_ratio = sum(1 for t in recent if t.success) / len(recent)
            # Weight: 60% success rate, 40% anomaly avoidance
            return round(0.6 * success_ratio + 0.4 * (1.0 - anomaly_ratio), 3)

    def anomaly_rate(self, window: int = 20) -> float:
        """Anomaly rate over the last N turns."""
        with self._lock:
            recent = list(self._turns)[-window:]
            if not recent:
                return 0.0
            return round(sum(1 for t in recent if t.had_anomaly) / len(recent), 3)

    def summary(self) -> dict[str, Any]:
        """Return a full session summary."""
        with self._lock:
            total = len(self._turns)
            return {
                "session_id": self.session_id,
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "total_turns": total,
                "consecutive_failures": self._consecutive_failures,
                "total_recoveries": self._total_recoveries,
                "health_score": self.health_score(),
                "anomaly_rate": self.anomaly_rate(),
                "recent_recoveries": self._recovery_events[-5:],
                "last_activity": self._last_activity,
            }

    def recent_turns(self, n: int = 10) -> list[dict]:
        """Return the most recent N turns."""
        with self._lock:
            return [t.to_dict() for t in list(self._turns)[-n:]]

    # -- Persistence --------------------------------------------------------

    def save(self) -> str:
        """Persist session to disk as JSON."""
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        path = os.path.join(SESSIONS_DIR, f"session_{self.session_id}.json")
        data = {
            "session_id": self.session_id,
            "start_time": self._start_time,
            "last_activity": self._last_activity,
            "consecutive_failures": self._consecutive_failures,
            "total_recoveries": self._total_recoveries,
            "recovery_events": self._recovery_events,
            "turns": [t.to_dict() for t in self._turns],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, session_id: str) -> "AgentSession | None":
        """Load a session from disk."""
        path = os.path.join(SESSIONS_DIR, f"session_{session_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            instance = cls(session_id=session_id)
            instance._start_time = data["start_time"]
            instance._last_activity = data["last_activity"]
            instance._consecutive_failures = data["consecutive_failures"]
            instance._total_recoveries = data["total_recoveries"]
            instance._recovery_events = data.get("recovery_events", [])
            # Restore turns
            for t in data.get("turns", []):
                tr = TurnRecord.__new__(TurnRecord)
                for s in TurnRecord.__slots__:
                    setattr(tr, s, t[s])
                instance._turns.append(tr)
            return instance
        except Exception as e:
            logger.warning("Failed to load session %s: %s", session_id, e)
            return None

    @staticmethod
    def list_sessions() -> list[dict]:
        """List all saved sessions with basic metadata."""
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        sessions = []
        for fname in os.listdir(SESSIONS_DIR):
            if fname.startswith("session_") and fname.endswith(".json"):
                try:
                    with open(os.path.join(SESSIONS_DIR, fname), encoding="utf-8") as f:
                        data = json.load(f)
                    sessions.append({
                        "session_id": data["session_id"],
                        "start_time": data.get("start_time", 0),
                        "total_turns": len(data.get("turns", [])),
                        "recoveries": data.get("total_recoveries", 0),
                        "last_activity": data.get("last_activity", 0),
                    })
                except Exception:
                    pass
        return sorted(sessions, key=lambda s: s["start_time"], reverse=True)


# Global active session
_active_session: AgentSession | None = None
_active_session_lock = threading.Lock()


def get_session() -> AgentSession:
    """Get or create the global active session."""
    global _active_session
    with _active_session_lock:
        if _active_session is None:
            _active_session = AgentSession()
        return _active_session


def reset_session() -> AgentSession:
    """Reset the global session (start fresh)."""
    global _active_session
    with _active_session_lock:
        _active_session = AgentSession()
    return _active_session
