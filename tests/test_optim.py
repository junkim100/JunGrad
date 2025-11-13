"""Tests for optimizers."""

import numpy as np
import pytest

from jungrad import randn, tensor, Tensor
from jungrad.nn import Linear
from jungrad.optim import Adam, AdamW, RMSProp, SGD


def test_sgd():
    """Test SGD optimizer."""
    model = Linear(5, 3)
    optimizer = SGD(model.parameters(), lr=0.01)

    from jungrad.ops import sum as sum_op

    x = randn(2, 5)
    out = model(x)
    loss = sum_op(out)
    loss.backward()

    # Store old weight
    old_weight = model._parameters["weight"].data.copy()

    # Check that gradient exists
    assert model._parameters["weight"].grad is not None

    optimizer.step()
    optimizer.zero_grad()

    # Weight should have changed (allow for small changes)
    weight_diff = np.abs(model._parameters["weight"].data - old_weight).max()
    assert weight_diff > 1e-6, f"Weight did not change significantly (max diff: {weight_diff})"


def test_adam():
    """Test Adam optimizer."""
    model = Linear(5, 3)
    optimizer = Adam(model.parameters(), lr=0.001)

    from jungrad.ops import sum as sum_op

    x = randn(2, 5)
    out = model(x)
    loss = sum_op(out)
    loss.backward()

    old_weight = model._parameters["weight"].data.copy()
    optimizer.step()
    optimizer.zero_grad()

    assert not np.allclose(model._parameters["weight"].data, old_weight)


def test_adamw():
    """Test AdamW optimizer."""
    model = Linear(5, 3)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    from jungrad.ops import sum as sum_op

    x = randn(2, 5)
    out = model(x)
    loss = sum_op(out)
    loss.backward()

    old_weight = model._parameters["weight"].data.copy()
    optimizer.step()
    optimizer.zero_grad()

    assert not np.allclose(model._parameters["weight"].data, old_weight)


def test_rmsprop():
    """Test RMSProp optimizer."""
    model = Linear(5, 3)
    optimizer = RMSProp(model.parameters(), lr=0.01)

    from jungrad.ops import sum as sum_op

    x = randn(2, 5)
    out = model(x)
    loss = sum_op(out)
    loss.backward()

    old_weight = model._parameters["weight"].data.copy()
    optimizer.step()
    optimizer.zero_grad()

    assert not np.allclose(model._parameters["weight"].data, old_weight)
