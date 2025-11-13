"""Tests for loss functions."""

import numpy as np
import pytest

from jungrad import randn, tensor, Tensor
from jungrad.losses import bce_with_logits, cross_entropy, mse_loss


def test_mse_loss():
    """Test MSE loss."""
    pred = tensor([[1.0, 2.0]], requires_grad=True)
    target = tensor([[1.0, 2.0]])
    loss = mse_loss(pred, target)
    assert np.allclose(loss.data, 0.0, atol=1e-6)


def test_mse_loss_nonzero():
    """Test MSE loss with difference."""
    pred = tensor([[1.0, 2.0]], requires_grad=True)
    target = tensor([[0.0, 0.0]])
    loss = mse_loss(pred, target)
    assert loss.data > 0.0


def test_cross_entropy_index():
    """Test cross-entropy with index targets."""
    logits = tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    target = np.array([2])  # Class 2
    loss = cross_entropy(logits, target)
    assert loss.data > 0.0


def test_cross_entropy_onehot():
    """Test cross-entropy with one-hot targets."""
    logits = tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    target = np.array([[0.0, 0.0, 1.0]])  # One-hot
    loss = cross_entropy(logits, target)
    assert loss.data > 0.0


def test_bce_with_logits():
    """Test BCE with logits."""
    logits = tensor([[0.0, 1.0]], requires_grad=True)
    target = np.array([[0.0, 1.0]])
    loss = bce_with_logits(logits, target)
    assert loss.data >= 0.0
