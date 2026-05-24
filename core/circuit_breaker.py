"""LLM API 断路器 — 跟踪失败次数并控制回退。"""

import threading
import time

from core.logger import setup_logger

logger = setup_logger("circuit_breaker")


class CircuitBreaker:
    """Per-circuit failure tracker with auto-recovery cooldown.

    States: closed (normal) → open (failing) → half_open (testing) → closed.
    """

    def __init__(self, threshold: int = 3, cooldown: float = 60.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures: dict[str, int] = {}
        self._last_failure: dict[str, float] = {}
        self._state: dict[str, str] = {}
        self._lock = threading.Lock()

    def _init(self, name: str) -> None:
        if name not in self._state:
            self._state[name] = "closed"
            self._failures[name] = 0
            self._last_failure[name] = 0.0

    def _transition(self, name: str, new_state: str) -> None:
        old = self._state.get(name, "closed")
        self._state[name] = new_state
        if old != new_state:
            logger.info("Circuit breaker [%s]: %s → %s", name, old, new_state)

    def record_failure(self, name: str) -> None:
        with self._lock:
            self._init(name)
            self._failures[name] += 1
            self._last_failure[name] = time.time()
            if self._failures[name] >= self.threshold and self._state[name] != "open":
                self._transition(name, "open")

    def record_success(self, name: str) -> None:
        with self._lock:
            self._init(name)
            if self._state[name] != "closed":
                self._transition(name, "closed")
            self._failures[name] = 0

    def can_execute(self, name: str) -> bool:
        """Check if a call to the named circuit is allowed."""
        with self._lock:
            self._init(name)
            if self._state[name] == "closed":
                return True
            if self._state[name] == "open":
                if time.time() - self._last_failure[name] >= self.cooldown:
                    self._transition(name, "half_open")
                    return True
                return False
            # half_open — allow probe call
            return True

    def reset(self, name: str | None = None) -> None:
        if name:
            with self._lock:
                self._init(name)
                self._state[name] = "closed"
                self._failures[name] = 0
        else:
            for n in list(self._state):
                self._init(n)
                self._state[n] = "closed"
                self._failures[n] = 0
            logger.info("Circuit breaker: all circuits reset")

    def status(self, name: str) -> dict:
        with self._lock:
            self._init(name)
            return {
                "state": self._state[name],
                "failures": self._failures[name],
                "last_failure_age": time.time() - self._last_failure[name],
            }

    def all_status(self) -> dict[str, dict]:
        return {n: self.status(n) for n in list(self._state)}


# Global instance
breaker = CircuitBreaker()
