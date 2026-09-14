"""
Phase 8 — request rate limiting.

Fixed-window counter per client IP. Uses Redis (INCR + EXPIRE) when
REDIS_URL is set, which is the only version of "rate limiting" that's
actually correct once you have more than one server process or
instance sharing a limit; falls back to a per-process in-memory
counter otherwise. The in-memory fallback is exactly right for this
project's actual deployment (a single free-tier Render instance) and
for local development without spinning up Redis — nobody should have
to run a Redis container just to `uvicorn src.serving.app:app`.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from threading import Lock


class InMemoryRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int, float]:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.max_requests:
                retry_after = self.window - (now - q[0])
                return False, 0, max(retry_after, 0.0)
            q.append(now)
            return True, self.max_requests - len(q), 0.0


class RedisRateLimiter:
    def __init__(self, redis_url: str, max_requests: int, window_seconds: int):
        import redis  # local import — only needed when REDIS_URL is actually set

        self.client = redis.Redis.from_url(redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)
        self.max_requests = max_requests
        self.window = window_seconds

    def allow(self, key: str) -> tuple[bool, int, float]:
        bucket = int(time.time()) // self.window
        redis_key = f"ratelimit:{key}:{bucket}"
        count = self.client.incr(redis_key)
        if count == 1:
            self.client.expire(redis_key, self.window)
        remaining = max(self.max_requests - count, 0)
        return count <= self.max_requests, remaining, float(self.window)


def build_rate_limiter(max_requests: int = 60, window_seconds: float = 60.0):
    """Picks Redis when REDIS_URL is set and reachable, otherwise the
    in-memory fallback — never raises, so a misconfigured/unreachable
    Redis degrades the service to per-process rate limiting instead of
    taking it down."""
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            limiter = RedisRateLimiter(redis_url, max_requests, int(window_seconds))
            limiter.client.ping()
            return limiter
        except Exception:
            pass
    return InMemoryRateLimiter(max_requests, window_seconds)
