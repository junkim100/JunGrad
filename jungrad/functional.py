"""High-level differentiable functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jungrad.ops import add, div, exp, log, max, maximum, mul, sub, sum, squeeze
from jungrad.tensor import Tensor, tensor

if TYPE_CHECKING:
    from jungrad.tensor import Tensor

__all__ = [
    "logsumexp",
    "softmax",
    "log_softmax",
    "relu",
    "tanh",
    "sigmoid",
    "gelu",
]


def logsumexp(
    a: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Tensor:
    """Stable log-sum-exp computation.

    Uses the max-subtraction trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

    Args:
        a: Input tensor.
        axis: Axis or axes to reduce over. None reduces all.
        keepdims: Whether to keep reduced dimensions.

    Returns:
        Result tensor.
    """
    # Compute max for numerical stability (as Tensor)
    max_val = max(a, axis=axis, keepdims=True)

    # Subtract max, then compute exp, sum, log
    shifted = sub(a, max_val)
    exp_shifted = exp(shifted)
    sum_exp = sum(exp_shifted, axis=axis, keepdims=True)
    log_sum = log(sum_exp)

    # Add max back
    if not keepdims and axis is not None:
        # Squeeze max_val to match
        if isinstance(axis, int):
            max_val_squeezed = squeeze(max_val, dim=axis)
        else:
            # Multiple axes - need to squeeze each
            max_val_squeezed = max_val
            for ax in sorted(axis, reverse=True):
                max_val_squeezed = squeeze(max_val_squeezed, dim=ax)
        return add(log_sum, max_val_squeezed)
    else:
        return add(log_sum, max_val)


def log_softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Log-softmax function (numerically stable).

    Args:
        a: Input tensor.
        axis: Axis to compute log-softmax over.

    Returns:
        Result tensor with log-softmax applied.
    """
    # log_softmax(x) = x - logsumexp(x)
    lse = logsumexp(a, axis=axis, keepdims=True)
    return a - lse


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Softmax function.

    Args:
        a: Input tensor.
        axis: Axis to compute softmax over.

    Returns:
        Result tensor with softmax applied.
    """
    # Compute log_softmax then exp
    log_smax = log_softmax(a, axis=axis)
    return exp(log_smax)


def relu(a: Tensor) -> Tensor:
    """Rectified Linear Unit activation.

    Args:
        a: Input tensor.

    Returns:
        Result tensor with ReLU applied.
    """
    zero = tensor(0.0, dtype=a.dtype)
    result = maximum(a, zero)
    result.op = "relu"
    return result


def tanh(a: Tensor) -> Tensor:
    """Hyperbolic tangent activation.

    Args:
        a: Input tensor.

    Returns:
        Result tensor with tanh applied.
    """
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # For stability: use tanh = sinh/cosh = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    exp_pos = exp(a)
    exp_neg = exp(-a)

    numerator = sub(exp_pos, exp_neg)
    denominator = add(exp_pos, exp_neg)
    return div(numerator, denominator)


def sigmoid(a: Tensor) -> Tensor:
    """Sigmoid activation (numerically stable).

    Args:
        a: Input tensor.

    Returns:
        Result tensor with sigmoid applied.
    """
    # Stable sigmoid: clip input to avoid overflow
    # Clamp input for numerical stability
    clamped = np.clip(a.data, -500, 500)  # Clip to avoid overflow
    clamped_tensor = tensor(clamped, requires_grad=a.requires_grad)
    exp_neg = exp(-clamped_tensor)
    one = tensor(1.0, dtype=a.dtype)
    return div(one, add(one, exp_neg))


def gelu(a: Tensor) -> Tensor:
    """Gaussian Error Linear Unit activation (approximation).

    Uses the approximate form: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        a: Input tensor.

    Returns:
        Result tensor with GELU applied.
    """
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    coeff = 0.044715

    # x^3
    x3 = pow(a, 3)
    # sqrt(2/pi) * (x + 0.044715 * x^3)
    inner = add(a, mul(x3, tensor(coeff, dtype=a.dtype)))
    scaled = mul(inner, tensor(sqrt_2_over_pi, dtype=a.dtype))
    # tanh(...)
    tanh_val = tanh(scaled)
    # 0.5 * (1 + tanh(...))
    one_half = tensor(0.5, dtype=a.dtype)
    one = tensor(1.0, dtype=a.dtype)
    factor = mul(one_half, add(one, tanh_val))
    # x * factor
    return mul(a, factor)
