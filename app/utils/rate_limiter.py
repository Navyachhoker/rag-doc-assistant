"""
Simple sliding-window rate limiter. Used specifically around the Gemini
Vision captioning calls, since that's the endpoint with the tight 10 RPM
free-tier ceiling — text embeddings run on a separate, much larger quota
and don't need throttling.
"""
import time
from collections import deque

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        """
        max_calls: requests allowed per period_seconds.
        Set below the actual API limit (e.g. 9 instead of 10) to leave
        headroom for retries/other traffic on the same project.
        """
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.call_times: deque[float] = deque()

    def wait_if_needed(self) -> None:
        now = time.monotonic()

        # Drop timestamps outside the current window
        while self.call_times and now - self.call_times[0] > self.period_seconds:
            self.call_times.popleft()

        if len(self.call_times) >= self.max_calls:
            sleep_time = self.period_seconds - (now - self.call_times[0])
            if sleep_time > 0:
                logger.info("rate_limit_throttling", sleep_seconds=round(sleep_time, 1))
                time.sleep(sleep_time)

        self.call_times.append(time.monotonic())