"""Measured timing for the performance monitor (feature 16).

This deliberately does nothing clever: it wraps an operation, measures how
long it actually took with time.perf_counter(), and writes that single real
number into the store's performance log. There is no estimation, no
benchmarking harness and no synthetic data — every entry the frontend shows
came from an operation that a user actually triggered.
"""

import time
from contextlib import contextmanager

from app.services.store import store


@contextmanager
def track(operation: str, meta: dict | None = None):
    """Time a block of code and log it, even if it raises.

    Usage:
        with track("Dataset Processing", meta={"dataset_id": ds_id}):
            ... do the work ...
    """
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        duration = time.perf_counter() - start
        store.log_performance(operation, duration, status=status, meta=meta)
