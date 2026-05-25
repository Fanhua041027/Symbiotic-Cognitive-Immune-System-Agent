"""Simple in-process rate limiter for API endpoints.

Uses a sliding window algorithm. Not suitable for multi-worker deployments;
for distributed rate limiting, use Redis or an API gateway.
"""

import threading
import time
from typing import Protocol

from core.config import get as cfg


class RateLimitExceeded(Exception):
    """Raised when a request exceeds the rate limit."""


class RateLimiter(Protocol):
    def check(self, key: str) -> None:
        """Check if key is within rate limit. Raises RateLimitExceeded if not."""
        ...


class TokenBucketRateLimiter:
    """Sliding window per-key rate limiter.

    Each key gets a window of `window_seconds` with at most `max_requests` calls.
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Record a request for the given key. Raises RateLimitExceeded if over limit."""
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = []
            timestamps = self._buckets[key]
            # Prune old entries
            while timestamps and timestamps[0] < cutoff:
                timestamps.pop(0)
            if len(timestamps) >= self.max_requests:
                retry_after = int(timestamps[0] + self.window_seconds - now)
                raise RateLimitExceeded(
                    f"Rate limit exceeded: {self.max_requests} requests per "
                    f"{self.window_seconds}s. Retry after {retry_after}s.",
                )
            timestamps.append(now)


# Global instance — configured via RATE_LIMIT_REQUESTS and RATE_LIMIT_WINDOW env vars
_limiter: TokenBucketRateLimiter | None = None
_limiter_lock = threading.Lock()


def get_limiter() -> TokenBucketRateLimiter | None:
    """Get or create the rate limiter. Returns None if rate limiting is disabled."""
    global _limiter
    if _limiter is not None:
        return _limiter
    with _limiter_lock:
        if _limiter is not None:
            return _limiter
        limit = cfg("RATE_LIMIT_REQUESTS", None)
        if limit is None:
            logger = __import__("core.logger", fromlist=["setup_logger"]).setup_logger("ratelimit")
            logger.info("Rate limiting disabled (RATE_LIMIT_REQUESTS not set)")
            return None
        window = int(cfg("RATE_LIMIT_WINDOW", 60))  # type: ignore[arg-type]
        _limiter = TokenBucketRateLimiter(max_requests=int(limit), window_seconds=window)
        return _limiter
