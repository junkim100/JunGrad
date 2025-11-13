"""Tests for functional API."""

import numpy as np
import pytest

from jungrad import randn, tensor, Tensor
from jungrad.functional import log_softmax, relu, sigmoid, softmax, tanh


def test_relu():
    """Test ReLU."""
    x = tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = relu(x)
    assert np.allclose(y.data, [0.0, 0.0, 1.0])


def test_relu_backward():
    """Test ReLU backward gradient."""
    x = tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = relu(x)
    y.sum().backward()
    assert np.allclose(x.grad, [0.0, 0.0, 1.0])


def test_tanh():
    """Test tanh."""
    x = tensor([0.0], requires_grad=True)
    y = tanh(x)
    assert np.allclose(y.data, [0.0], atol=1e-6)


def test_sigmoid():
    """Test sigmoid."""
    x = tensor([0.0], requires_grad=True)
    y = sigmoid(x)
    assert np.allclose(y.data, [0.5], atol=1e-6)


def test_softmax():
    """Test softmax."""
    x = tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = softmax(x, axis=-1)
    # Softmax should sum to 1
    assert np.allclose(y.data.sum(), 1.0, atol=1e-6)


def test_log_softmax():
    """Test log_softmax."""
    x = tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = log_softmax(x, axis=-1)
    # exp(log_softmax) should sum to 1
    assert np.allclose(np.exp(y.data).sum(), 1.0, atol=1e-6)
