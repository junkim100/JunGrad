"""Weight initialization functions."""

from __future__ import annotations

import numpy as np

from jungrad.nn.module import Parameter
from jungrad.tensor import Tensor

__all__ = ["xavier_uniform_", "kaiming_normal_", "orthogonal_", "constant_"]


def xavier_uniform_(param: Parameter | Tensor, gain: float = 1.0) -> None:
    """Xavier (Glorot) uniform initialization.

    Args:
        param: Parameter tensor to initialize.
        gain: Gain factor.
    """
    fan_in = param.shape[0] if len(param.shape) >= 2 else param.size
    fan_out = param.shape[1] if len(param.shape) >= 2 else param.size

    if len(param.shape) >= 2:
        fan_in = np.prod(param.shape[1:])
        fan_out = param.shape[0]

    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    param.data = np.random.uniform(-limit, limit, size=param.shape).astype(param.dtype)


def kaiming_normal_(param: Parameter | Tensor, a: float = 0.0, mode: str = "fan_in") -> None:
    """Kaiming (He) normal initialization.

    Args:
        param: Parameter tensor to initialize.
        a: Negative slope of ReLU (0 for linear).
        mode: 'fan_in' or 'fan_out'.
    """
    fan_in = param.shape[0] if len(param.shape) >= 2 else param.size
    fan_out = param.shape[1] if len(param.shape) >= 2 else param.size

    if len(param.shape) >= 2:
        fan_in = np.prod(param.shape[1:])
        fan_out = param.shape[0]

    fan = fan_in if mode == "fan_in" else fan_out
    gain = np.sqrt(2.0 / (1 + a ** 2))
    std = gain / np.sqrt(fan)

    param.data = np.random.normal(0, std, size=param.shape).astype(param.dtype)


def orthogonal_(param: Parameter | Tensor, gain: float = 1.0) -> None:
    """Orthogonal initialization.

    Args:
        param: Parameter tensor to initialize.
        gain: Gain factor.
    """
    if len(param.shape) < 2:
        raise ValueError("Orthogonal initialization requires at least 2 dimensions")

    # Generate orthogonal matrix
    flat_shape = (param.shape[0], np.prod(param.shape[1:]))
    a = np.random.randn(*flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(param.shape)

    param.data = (gain * q).astype(param.dtype)


def constant_(param: Parameter | Tensor, val: float) -> None:
    """Constant initialization.

    Args:
        param: Parameter tensor to initialize.
        val: value to fill with.
    """
    param.data.fill(val)

