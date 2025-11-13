"""Utility functions for jungrad."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np

from jungrad.exceptions import NumericsError, ShapeError
from jungrad.types import Shape

_logger = None


def get_logger():
    """Lazy import to avoid circular dependencies."""
    global _logger
    if _logger is None:
        from jungrad.logging import get_logger

        _logger = get_logger(__name__)
    return _logger


def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


def asarray(x, dtype=None) -> np.ndarray:
    """Convert input to numpy array with optional dtype.

    Args:
        x: Input to convert.
        dtype: Optional dtype to use.

    Returns:
        NumPy array.
    """
    return np.asarray(x, dtype=dtype)


def check_shape_compatible(shape1: Shape, shape2: Shape) -> bool:
    """Check if two shapes are compatible for broadcasting.

    Args:
        shape1: First shape.
        shape2: Second shape.

    Returns:
        True if shapes are compatible.
    """
    try:
        np.broadcast_shapes(shape1, shape2)
        return True
    except ValueError:
        return False


def broadcast_shape(*shapes: Shape) -> Shape:
    """Compute broadcasted shape from input shapes.

    Args:
        *shapes: Input shapes to broadcast.

    Returns:
        Broadcasted shape.

    Raises:
        ShapeError: If shapes cannot be broadcast.
    """
    try:
        return np.broadcast_shapes(*shapes)
    except ValueError as e:
        raise ShapeError(f"Cannot broadcast shapes {shapes}: {e}") from e


def reduce_broadcasted_grad(grad: np.ndarray, target_shape: Shape) -> np.ndarray:
    """Reduce gradient over broadcasted axes.

    Args:
        grad: Gradient with potentially broadcasted shape.
        target_shape: Target shape to reduce to.

    Returns:
        Reduced gradient matching target_shape.
    """
    if grad.shape == target_shape:
        return grad

    # Sum over leading dimensions that were broadcast
    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)

    # Sum over dimensions where size was 1
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if g_dim != t_dim:
            if t_dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
            else:
                raise ShapeError(f"Cannot reduce grad shape {grad.shape} to {target_shape}")

    return grad.reshape(target_shape)


def check_finite(x: np.ndarray, context: str = "") -> None:
    """Check for NaN/Inf values in array.

    Args:
        x: Array to check.
        context: Additional context string for error message.

    Raises:
        NumericsError: If NaN or Inf detected.
    """
    if not np.isfinite(x).all():
        nan_count = np.isnan(x).sum()
        inf_count = np.isinf(x).sum()
        msg = f"Non-finite values detected"
        if context:
            msg += f" in {context}"
        msg += f": {nan_count} NaN, {inf_count} Inf"
        raise NumericsError(msg)


def enable_nan_check(enabled: bool = True) -> None:
    """Enable/disable automatic NaN checking.

    Args:
        enabled: Whether to enable checking.
    """
    global _nan_check_enabled
    _nan_check_enabled = enabled


_nan_check_enabled = False


def to_fp16(x: np.ndarray) -> np.ndarray:
    """Convert array to float16.

    Args:
        x: Input array.

    Returns:
        Float16 array.
    """
    return x.astype(np.float16)


def to_fp32(x: np.ndarray) -> np.ndarray:
    """Convert array to float32.

    Args:
        x: Input array.

    Returns:
        Float32 array.
    """
    return x.astype(np.float32)


def validate_shape(shape: Tuple[int, ...]) -> None:
    """Validate that shape contains only positive integers.

    Args:
        shape: Shape tuple to validate.

    Raises:
        ShapeError: If shape is invalid.
    """
    if not isinstance(shape, tuple):
        raise ShapeError(f"Shape must be tuple, got {type(shape)}")
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise ShapeError(f"Shape dimensions must be non-negative integers, got {shape}")
