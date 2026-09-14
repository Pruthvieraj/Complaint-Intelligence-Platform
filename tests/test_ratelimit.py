"""Unit tests for the in-memory rate limiter (Phase 8). The Redis-backed
path isn't exercised here since CI has no Redis instance to talk to —
that path only activates when REDIS_URL is actually set (see
build_rate_limiter in src/serving/ratelimit.py), so it never affects
these tests or a default deployment."""
from src.serving.ratelimit import InMemoryRateLimiter
import src.serving.ratelimit as ratelimit_module


def test_allows_up_to_the_limit_then_blocks():
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60.0)
    results = [limiter.allow("client-a")[0] for _ in range(4)]
    assert results == [True, True, True, False]


def test_limits_are_tracked_per_key():
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60.0)
    assert limiter.allow("client-a")[0] is True
    assert limiter.allow("client-b")[0] is True  # different key, independent budget
    assert limiter.allow("client-a")[0] is False


def test_window_expiry_allows_requests_again(monkeypatch):
    fake_time = {"t": 1000.0}
    monkeypatch.setattr(ratelimit_module.time, "time", lambda: fake_time["t"])

    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=10.0)
    assert limiter.allow("client-a")[0] is True
    assert limiter.allow("client-a")[0] is False

    fake_time["t"] += 11.0
    assert limiter.allow("client-a")[0] is True
