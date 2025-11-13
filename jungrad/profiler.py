"""Simple profiler for timing operations."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

__all__ = ["Profiler", "profile"]


class Profiler:
    """Simple profiler for tracking operation timings."""

    def __init__(self):
        """Initialize profiler."""
        self.counts: dict[str, int] = defaultdict(int)
        self.times: dict[str, float] = defaultdict(float)
        self.enabled = True

    @contextmanager
    def record(self, name: str):
        """Context manager to record timing.

        Args:
            name: Operation name.

        Yields:
            None.
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.counts[name] += 1
            self.times[name] += elapsed

    def reset(self) -> None:
        """Reset all statistics."""
        self.counts.clear()
        self.times.clear()

    def summary(self) -> str:
        """Get summary of timings.

        Returns:
            Formatted summary string.
        """
        if not self.counts:
            return "No profiling data collected."

        lines = ["Profiling Summary:"]
        lines.append(
            f"{'Operation':<30} {'Count':<10} {'Total Time (s)':<15} {'Avg Time (ms)':<15}"
        )
        lines.append("-" * 70)

        for name in sorted(self.times.keys()):
            count = self.counts[name]
            total = self.times[name]
            avg_ms = (total / count) * 1000 if count > 0 else 0
            lines.append(f"{name:<30} {count:<10} {total:<15.6f} {avg_ms:<15.4f}")

        return "\n".join(lines)


# Global profiler instance
_profiler = Profiler()


@contextmanager
def profile(name: str, profiler: Optional[Profiler] = None):
    """Context manager for profiling.

    Args:
        name: Operation name.
        profiler: Optional profiler instance (uses global if None).

    Yields:
        None.
    """
    p = profiler or _profiler
    with p.record(name):
        yield


def get_profiler() -> Profiler:
    """Get global profiler instance.

    Returns:
        Global profiler.
    """
    return _profiler
