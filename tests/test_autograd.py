"""Tests for autograd engine."""

import numpy as np
import pytest

from jungrad import no_grad, randn, tensor
from jungrad.autograd import is_grad_enabled, set_grad_enabled
from jungrad.ops import add, mul


def test_backward_simple():
    """Test simple backward pass."""
    a = tensor([2.0], requires_grad=True)
    b = tensor([3.0], requires_grad=True)
    c = mul(a, b)
    c.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert np.allclose(a.grad, [3.0])
    assert np.allclose(b.grad, [2.0])


def test_backward_chain():
    """Test backward through chain."""
    x = tensor([2.0], requires_grad=True)
    y = mul(x, x)  # x^2
    z = mul(y, x)  # x^3
    z.backward()

    assert np.allclose(x.grad, [12.0])  # 3 * x^2 = 3 * 4 = 12


def test_grad_mode():
    """Test grad mode control."""
    assert is_grad_enabled() is True

    with no_grad():
        assert is_grad_enabled() is False
        x = tensor([1.0], requires_grad=True)
        assert x.requires_grad is False

    assert is_grad_enabled() is True


def test_zero_grad():
    """Test zero_grad."""
    x = tensor([1.0], requires_grad=True)
    y = mul(x, x)
    y.backward()
    assert x.grad is not None

    x.zero_grad()
    assert np.allclose(x.grad, [0.0])


def test_retain_grad():
    """Test retain_grad."""
    x = tensor([2.0], requires_grad=True)
    y = mul(x, x)
    y.retain_grad()
    z = mul(y, x)
    z.backward()

    assert y.grad is not None
    assert np.allclose(y.grad, [2.0])
