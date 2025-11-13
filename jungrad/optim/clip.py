"""Gradient clipping utilities."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from jungrad.nn.module import Parameter

__all__ = ["clip_grad_norm_", "clip_grad_value_"]


def clip_grad_norm_(
    parameters: Iterator[Parameter],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Clip gradients by norm.

    Args:
        parameters: Iterator over parameters.
        max_norm: Maximum norm value.
        norm_type: Type of norm (L2 by default).

    Returns:
        Total norm before clipping.
    """
    parameters = list(parameters)

    # Compute total norm
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = np.linalg.norm(param.grad, ord=norm_type)
            total_norm += param_norm**norm_type

    total_norm = total_norm ** (1.0 / norm_type)

    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in parameters:
            if param.grad is not None:
                param.grad = param.grad * clip_coef

    return float(total_norm)


def clip_grad_value_(
    parameters: Iterator[Parameter],
    clip_value: float,
) -> None:
    """Clip gradients by value.

    Args:
        parameters: Iterator over parameters.
        clip_value: Maximum absolute value for gradients.
    """
    for param in parameters:
        if param.grad is not None:
            param.grad = np.clip(param.grad, -clip_value, clip_value)
