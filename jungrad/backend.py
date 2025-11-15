"""Backend utilities for selecting NumPy or CuPy at runtime."""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp  # type: ignore

    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - cupy optional
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False

__all__ = [
    "has_cupy",
    "get_array_module",
    "to_device_array",
    "to_numpy_array",
]


def has_cupy() -> bool:
    """Return True when CuPy is installed."""

    return _CUPY_AVAILABLE


def get_array_module(device: str | None = None):
    """Return array module (np or cp) for given device."""

    if device and device.lower() in {"cuda", "gpu"} and _CUPY_AVAILABLE:
        return cp  # type: ignore[return-value]
    return np


def to_device_array(array, device: str | None = None):
    """Move array to target device (NumPy/CuPy)."""

    xp = get_array_module(device)
    if xp is np:
        if _CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)

    # xp is CuPy here
    if isinstance(array, xp.ndarray):  # type: ignore[attr-defined]
        return array
    return xp.asarray(array)


def to_numpy_array(array):
    """Convert np/cp array to NumPy."""

    if _CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)
