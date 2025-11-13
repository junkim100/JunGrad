"""Hooks for tensors and modules."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from jungrad.nn.module import Module
from jungrad.tensor import Tensor

__all__ = []


# Tensor hooks
class TensorHook:
    """Hook for tensor backward pass."""

    def __init__(self, fn: Callable[[np.ndarray], Optional[np.ndarray]]):
        """Initialize hook.

        Args:
            fn: Function to call with gradient. Returns modified gradient or None.
        """
        self.fn = fn

    def __call__(self, grad: np.ndarray) -> Optional[np.ndarray]:
        """Call hook function."""
        return self.fn(grad)


def register_tensor_hook(
    tensor: Tensor, fn: Callable[[np.ndarray], Optional[np.ndarray]]
) -> TensorHook:
    """Register backward hook on tensor.

    Args:
        tensor: Tensor to register hook on.
        fn: Hook function.

    Returns:
        Hook object.
    """
    hook = TensorHook(fn)
    if not hasattr(tensor, "_hooks"):
        tensor._hooks = []
    tensor._hooks.append(hook)
    return hook


# Module hooks
class ModuleHook:
    """Base class for module hooks."""

    pass


def register_forward_hook(
    module: Module,
    fn: Callable[[Module, tuple, dict], Optional[tuple]],
) -> Callable:
    """Register forward hook on module.

    Args:
        module: Module to register hook on.
        fn: Hook function(module, input, output) -> modified output or None.

    Returns:
        Hook handle (callable to remove hook).
    """
    if not hasattr(module, "_forward_hooks"):
        module._forward_hooks = []

    def remove():
        module._forward_hooks.remove(hook)

    hook = (fn, remove)
    module._forward_hooks.append(hook)
    return remove


def register_backward_hook(
    module: Module,
    fn: Callable[[Module, np.ndarray], Optional[np.ndarray]],
) -> Callable:
    """Register backward hook on module.

    Args:
        module: Module to register hook on.
        fn: Hook function(module, grad) -> modified grad or None.

    Returns:
        Hook handle (callable to remove hook).
    """
    if not hasattr(module, "_backward_hooks"):
        module._backward_hooks = []

    def remove():
        module._backward_hooks.remove(hook)

    hook = (fn, remove)
    module._backward_hooks.append(hook)
    return remove
