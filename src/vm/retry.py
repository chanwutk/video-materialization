"""Retry helpers for transient Gemini API errors (503 UNAVAILABLE, 429 RESOURCE_EXHAUSTED, etc.)."""

import asyncio
import random
import time
from typing import Any, Awaitable, Callable, TypeVar

from google.genai import errors as genai_errors

T = TypeVar("T")

# Status codes worth retrying. The SDK already retries some of these but its budget
# can be exceeded during long runs (multi-hour optimization).
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        code = getattr(exc, "code", None)
        if code in _RETRY_STATUS_CODES:
            return True
    return False


def _backoff(attempt: int, base: float = 4.0, cap: float = 120.0) -> float:
    """Exponential backoff with jitter. attempt=0 -> ~4s, attempt=4 -> ~64s, capped at 2 min."""
    delay = min(cap, base * (2 ** attempt))
    return delay * (0.5 + random.random())  # 0.5x to 1.5x jitter


def with_retries(
    fn: Callable[..., T],
    *args: Any,
    max_attempts: int = 6,
    label: str = "API call",
    **kwargs: Any,
) -> T:
    """Synchronous retry wrapper for Gemini API calls."""
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except BaseException as e:
            last = e
            if not _is_retryable(e) or attempt == max_attempts - 1:
                raise
            delay = _backoff(attempt)
            print(f"  [retry] {label} failed ({type(e).__name__}: {e}); "
                  f"sleeping {delay:.1f}s (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)
    assert last is not None
    raise last


async def with_retries_async(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    max_attempts: int = 6,
    label: str = "API call",
    **kwargs: Any,
) -> T:
    """Async retry wrapper for Gemini API calls."""
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await fn(*args, **kwargs)
        except BaseException as e:
            last = e
            if not _is_retryable(e) or attempt == max_attempts - 1:
                raise
            delay = _backoff(attempt)
            print(f"  [retry] {label} failed ({type(e).__name__}: {e}); "
                  f"sleeping {delay:.1f}s (attempt {attempt + 1}/{max_attempts})")
            await asyncio.sleep(delay)
    assert last is not None
    raise last
