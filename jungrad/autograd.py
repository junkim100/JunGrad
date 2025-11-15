"""Autograd engine for backward propagation."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - optional CuPy dependency
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - CuPy optional
    cp = None  # type: ignore

from jungrad.types import Edge, Node
from jungrad.utils import get_logger

logger = get_logger()

# Global grad mode flag
_grad_enabled = True


def set_grad_enabled(enabled: bool) -> None:
    """Enable or disable gradient computation globally.

    Args:
        enabled: Whether to enable gradients.
    """
    global _grad_enabled
    _grad_enabled = enabled


def is_grad_enabled() -> bool:
    """Check if gradient computation is enabled.

    Returns:
        True if gradients are enabled.
    """
    return _grad_enabled


class no_grad:
    """Context manager to disable gradient computation."""

    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(False)
        return self

    def __exit__(self, *args):
        set_grad_enabled(self.prev)


class enable_grad:
    """Context manager to enable gradient computation."""

    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(True)
        return self

    def __exit__(self, *args):
        set_grad_enabled(self.prev)


def toposort(outputs) -> list:
    """Topologically sort computation graph.

    Args:
        outputs: Output tensors (can be single tensor or iterable).

    Returns:
        List of tensors in topological order (leaves first, outputs last).
    """
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    visited = set()
    result = []

    def visit(tensor):
        if tensor in visited:
            return
        visited.add(tensor)
        # Visit all parents first
        for edge in tensor.parents:
            visit(edge.tensor)
        result.append(tensor)

    for output in outputs:
        visit(output)

    return result


def _get_array_module(data):
    """Return np or cp to match ``data``."""

    if cp is not None:
        if isinstance(data, cp.ndarray):
            return cp
    return np


def _ensure_array_on_xp(array, xp):
    """Ensure ``array`` lives on the array module ``xp``."""

    if xp is np:
        if cp is not None and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    # xp is CuPy here
    if isinstance(array, xp.ndarray):  # type: ignore[attr-defined]
        return array
    return xp.asarray(array)


def backward(output, grad: Optional[np.ndarray] = None) -> None:
    """Run backward pass to compute gradients.

    Args:
        output: Output tensor.
        grad: Initial gradient (defaults to ones_like if not provided).
    """
    if not output.requires_grad:
        return

    # Topologically sort the graph
    topo = toposort(output)

    # Initialize output gradient
    xp_out = _get_array_module(output.data)

    if grad is None:
        grad = xp_out.ones_like(output.data)
    else:
        grad = _ensure_array_on_xp(grad, xp_out)

    if output.grad is None:
        output.grad = xp_out.zeros_like(output.data)
    else:
        output.grad = _ensure_array_on_xp(output.grad, xp_out)
    output.grad += grad

    # Backward through graph in reverse topological order
    for tensor in reversed(topo):
        # This means the tensor is a leaf node and has no parents, so we can skip it
        if tensor.grad is None:
            continue

        # Propagate gradient to each parent
        for edge in tensor.parents:
            parent = edge.tensor
            if not parent.requires_grad:
                continue

            # Compute gradient w.r.t. parent using grad_fn
            if edge.grad_fn is not None:
                parent_grad = edge.grad_fn(tensor.grad)
            else:
                # Default: pass gradient through unchanged (for identity ops)
                parent_grad = tensor.grad

            # Initialize parent grad if needed
            xp_parent = _get_array_module(parent.data)
            parent_grad = _ensure_array_on_xp(parent_grad, xp_parent)

            if parent.grad is None:
                parent.grad = xp_parent.zeros_like(parent.data)
            else:
                parent.grad = _ensure_array_on_xp(parent.grad, xp_parent)

            parent.grad += parent_grad
