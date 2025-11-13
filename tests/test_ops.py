"""Tests for operations."""

import numpy as np
import pytest

from jungrad import randn, tensor, Tensor
from jungrad.ops import add, div, exp, log, matmul, mul, neg, pow, sub, sum


def test_add():
    """Test addition."""
    a = tensor([1.0, 2.0], requires_grad=True)
    b = tensor([3.0, 4.0], requires_grad=True)
    c = add(a, b)
    assert np.allclose(c.data, [4.0, 6.0])


def test_sub():
    """Test subtraction."""
    a = tensor([5.0, 3.0], requires_grad=True)
    b = tensor([2.0, 1.0], requires_grad=True)
    c = sub(a, b)
    assert np.allclose(c.data, [3.0, 2.0])


def test_mul():
    """Test multiplication."""
    a = tensor([2.0, 3.0], requires_grad=True)
    b = tensor([4.0, 5.0], requires_grad=True)
    c = mul(a, b)
    assert np.allclose(c.data, [8.0, 15.0])


def test_div():
    """Test division."""
    a = tensor([8.0, 15.0], requires_grad=True)
    b = tensor([2.0, 3.0], requires_grad=True)
    c = div(a, b)
    assert np.allclose(c.data, [4.0, 5.0])


def test_neg():
    """Test negation."""
    a = tensor([1.0, -2.0], requires_grad=True)
    b = neg(a)
    assert np.allclose(b.data, [-1.0, 2.0])


def test_pow():
    """Test power."""
    a = tensor([2.0, 3.0], requires_grad=True)
    b = pow(a, 2)
    assert np.allclose(b.data, [4.0, 9.0])


def test_exp():
    """Test exponential."""
    a = tensor([0.0, 1.0], requires_grad=True)
    b = exp(a)
    assert np.allclose(b.data, [1.0, np.e], atol=1e-6)


def test_log():
    """Test logarithm."""
    a = tensor([1.0, np.e], requires_grad=True)
    b = log(a)
    assert np.allclose(b.data, [0.0, 1.0], atol=1e-6)


def test_matmul():
    """Test matrix multiplication."""
    a = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    c = matmul(a, b)
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(c.data, expected)


def test_sum():
    """Test sum reduction."""
    a = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = sum(a)
    assert np.allclose(b.data, 10.0)

    c = sum(a, axis=0)
    assert np.allclose(c.data, [4.0, 6.0])


def test_broadcasting():
    """Test broadcasting."""
    a = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = tensor([1.0, 2.0], requires_grad=True)
    c = add(a, b)
    expected = np.array([[2.0, 4.0], [4.0, 6.0]])
    assert np.allclose(c.data, expected)
