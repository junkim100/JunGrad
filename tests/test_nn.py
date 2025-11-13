"""Tests for neural network modules."""

import numpy as np
import pytest

from jungrad import randn, Tensor
from jungrad.nn import Dropout, Embedding, LayerNorm, Linear, Sequential


def test_linear():
    """Test Linear layer."""
    linear = Linear(10, 5)
    x = randn(3, 10)
    out = linear(x)
    assert out.shape == (3, 5)


def test_sequential():
    """Test Sequential container."""
    model = Sequential(
        Linear(10, 20),
        Linear(20, 5),
    )
    x = randn(2, 10)
    out = model(x)
    assert out.shape == (2, 5)


def test_dropout():
    """Test Dropout layer."""
    dropout = Dropout(p=0.5)
    x = randn(3, 10)

    # Training mode
    dropout.train()
    out_train = dropout(x)
    assert out_train.shape == x.shape

    # Eval mode
    dropout.eval()
    out_eval = dropout(x)
    assert np.allclose(out_eval.data, x.data)  # No dropout in eval


def test_layernorm():
    """Test LayerNorm."""
    ln = LayerNorm(10, eps=1e-5)
    x = randn(2, 5, 10)
    out = ln(x)
    assert out.shape == x.shape

    # Check normalization
    mean = out.data.mean(axis=-1)
    std = out.data.std(axis=-1)
    assert np.allclose(mean, 0.0, atol=0.1)
    assert np.allclose(std, 1.0, atol=0.1)


def test_embedding():
    """Test Embedding layer."""
    from jungrad import tensor

    emb = Embedding(100, 64)
    indices = tensor(np.array([[0, 1, 2], [3, 4, 5]]), requires_grad=False)
    out = emb(indices)
    assert out.shape == (2, 3, 64)


def test_module_state_dict():
    """Test state_dict and load_state_dict."""
    model = Linear(5, 3)
    state = model.state_dict()

    assert "weight" in state
    assert "bias" in state

    # Create new model and load state
    model2 = Linear(5, 3)
    model2.load_state_dict(state)

    # Check weights match
    assert np.allclose(model._parameters["weight"].data, model2._parameters["weight"].data)
