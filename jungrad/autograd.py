"""Autograd engine for backward propagation."""

from __future__ import annotations

from typing import Optional

import numpy as np

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
    if grad is None:
        grad = np.ones_like(output.data)

    if output.grad is None:
        output.grad = np.zeros_like(output.data)
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
            if parent.grad is None:
                parent.grad = np.zeros_like(parent.data)

            # Accumulate gradient in parent
            parent.grad += parent_grad
