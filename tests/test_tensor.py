"""Tests for Tensor class."""

import numpy as np
import pytest

from jungrad import Tensor, randn, tensor, zeros, ones


def test_tensor_creation():
    """Test tensor creation."""
    t = tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.dtype == np.float64


def test_tensor_requires_grad():
    """Test requires_grad flag."""
    t1 = tensor([1.0], requires_grad=True)
    t2 = tensor([1.0], requires_grad=False)
    assert t1.requires_grad is True
    assert t2.requires_grad is False


def test_tensor_detach():
    """Test tensor detachment."""
    t = tensor([1.0], requires_grad=True)
    detached = t.detach()
    assert detached.requires_grad is False


def test_zeros_ones():
    """Test zeros and ones creation."""
    z = zeros((3, 4))
    assert z.shape == (3, 4)
    assert np.allclose(z.data, 0.0)

    o = ones((2, 2))
    assert o.shape == (2, 2)
    assert np.allclose(o.data, 1.0)


def test_randn():
    """Test random tensor creation."""
    r = randn(5, 5)
    assert r.shape == (5, 5)


def test_tensor_item():
    """Test item() for scalar tensors."""
    t = tensor(5.0)
    assert t.item() == 5.0


def test_tensor_numpy():
    """Test numpy() method."""
    t = tensor([1.0, 2.0])
    arr = t.numpy()
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, [1.0, 2.0])
