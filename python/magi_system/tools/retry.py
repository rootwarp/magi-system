"""Retry utility with exponential backoff for external API calls."""

import asyncio
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

# Transient errors worth retrying
TRANSIENT_ERRORS = (TimeoutError, ConnectionError, OSError)


def retry_with_backoff(
    max_retries=3, base_delay=1.0, transient_errors=TRANSIENT_ERRORS
):
    """Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts after the initial call.
        base_delay: Base delay in seconds; doubles with each retry.
        transient_errors: Tuple of exception types that trigger a retry.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_error = None
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except transient_errors as e:
                        last_error = e
                        if attempt < max_retries:
                            delay = base_delay * (2**attempt)
                            logger.warning(
                                "Retry %d/%d for %s: %s. Waiting %ss",
                                attempt + 1,
                                max_retries,
                                func.__name__,
                                e,
                                delay,
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                "Max retries (%d) exceeded for %s: %s",
                                max_retries,
                                func.__name__,
                                e,
                            )
                raise last_error

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_error = None
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except transient_errors as e:
                        last_error = e
                        if attempt < max_retries:
                            delay = base_delay * (2**attempt)
                            logger.warning(
                                "Retry %d/%d for %s: %s. Waiting %ss",
                                attempt + 1,
                                max_retries,
                                func.__name__,
                                e,
                                delay,
                            )
                            time.sleep(delay)
                        else:
                            logger.error(
                                "Max retries (%d) exceeded for %s: %s",
                                max_retries,
                                func.__name__,
                                e,
                            )
                raise last_error

            return sync_wrapper

    return decorator
