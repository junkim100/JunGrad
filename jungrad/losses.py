"""Loss functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jungrad.functional import log_softmax
from jungrad.ops import add, exp, log, mean, mul, neg, sub, sum, pow
from jungrad.tensor import tensor, Tensor
from jungrad.types import Edge

if TYPE_CHECKING:
    from jungrad.tensor import Tensor

__all__ = ["cross_entropy", "bce_with_logits", "mse_loss"]


def cross_entropy(
    logits: Tensor,
    target: Tensor | np.ndarray,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross-entropy loss.

    Supports both index targets and one-hot targets.

    Args:
        logits: Logit tensor (..., num_classes).
        target: Target tensor - either class indices (...,) or one-hot (..., num_classes).
        label_smoothing: Label smoothing factor (default 0.0).

    Returns:
        Loss tensor (scalar after reduction).
    """
    # Convert target to numpy if needed
    if isinstance(target, Tensor):
        target_np = target.data
    else:
        target_np = np.asarray(target)

    # Determine if target is one-hot or indices
    if target_np.shape == logits.shape:
        # One-hot encoding
        one_hot = True
        num_classes = logits.shape[-1]
    else:
        # Class indices
        one_hot = False
        num_classes = logits.shape[-1]

    # Compute log_softmax
    log_probs = log_softmax(logits, axis=-1)

    if one_hot:
        # One-hot targets
        if label_smoothing > 0:
            # Smooth labels: (1 - alpha) * y + alpha / K
            smooth_target = tensor(
                (1 - label_smoothing) * target_np + label_smoothing / num_classes,
                dtype=logits.dtype,
            )
        else:
            smooth_target = tensor(target_np, dtype=logits.dtype)

        # Cross-entropy = -sum(y * log(p))
        losses = neg(mul(log_probs, smooth_target))
        loss = mean(sum(losses, axis=-1))
    else:
        # Index targets - use advanced indexing
        # For simplicity, use numpy indexing to select then wrap
        batch_shape = logits.shape[:-1]
        flat_target = target_np.flatten().astype(np.int64)
        flat_log_probs = log_probs.data.reshape(-1, num_classes)

        # Select log_probs[i, target[i]] for each i
        selected_log_probs_np = flat_log_probs[np.arange(len(flat_target)), flat_target]

        # Apply label smoothing if needed
        if label_smoothing > 0:
            # Smooth: (1 - alpha) * log(p[y]) + alpha * mean(log(p))
            uniform_term_np = np.mean(log_probs.data, axis=-1).flatten()
            smooth_factor = 1 - label_smoothing
            uniform_factor = label_smoothing
            combined = smooth_factor * selected_log_probs_np + uniform_factor * uniform_term_np
            selected_log_probs_np = combined

        # Wrap and negate
        selected_log_probs = tensor(
            selected_log_probs_np, requires_grad=logits.requires_grad, dtype=logits.dtype
        )
        # Set up backward manually
        if logits.requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                # Gradient w.r.t. logits
                grad_out = np.zeros_like(log_probs.data)
                batch_size = len(flat_target)
                for i in range(batch_size):
                    idx = flat_target[i]
                    if label_smoothing > 0:
                        # Smooth gradient
                        grad_out.reshape(batch_size, -1)[i, idx] += grad[i] * smooth_factor
                        # Uniform part
                        grad_out.reshape(batch_size, -1)[i, :] += (
                            grad[i] * uniform_factor / num_classes
                        )
                    else:
                        grad_out.reshape(batch_size, -1)[i, idx] += grad[i]
                # Propagate through log_softmax: dL/dlogits = dL/dlog_probs * dlog_probs/dlogits
                # dlog_probs/dlogits = 1 - softmax = 1 - exp(log_softmax)
                softmax_probs = np.exp(log_probs.data)
                return grad_out - softmax_probs * np.sum(grad_out, axis=-1, keepdims=True)

            selected_log_probs.parents = (Edge(logits, grad_fn),)
            selected_log_probs.op = "cross_entropy"

        # Mean reduction
        loss = mean(neg(selected_log_probs))

    return loss


def bce_with_logits(logits: Tensor, target: Tensor | np.ndarray) -> Tensor:
    """Binary cross-entropy with logits (numerically stable).

    Uses the stable formula: max(logit, 0) - logit * target + log(1 + exp(-abs(logit)))

    Args:
        logits: Logit tensor.
        target: Target tensor (same shape, values in [0, 1]).

    Returns:
        Loss tensor (scalar after reduction).
    """
    # Convert target to tensor
    if not isinstance(target, Tensor):
        target_tensor = tensor(target, dtype=logits.dtype, requires_grad=False)
    else:
        target_tensor = target

    # Stable BCE: -[target * log(sigmoid(x)) + (1-target) * log(1-sigmoid(x))]
    # = max(x, 0) - x*target + log(1 + exp(-|x|))

    # For numerical stability, implement directly
    max_term_np = np.maximum(logits.data, 0)
    target_term_np = logits.data * target_tensor.data

    # log(1 + exp(-|x|))
    abs_logits_np = np.abs(logits.data)
    log_term_np = np.log1p(np.exp(-abs_logits_np))

    # Combine
    loss_elem_np = max_term_np - target_term_np + log_term_np
    loss_elem = tensor(loss_elem_np, requires_grad=logits.requires_grad, dtype=logits.dtype)

    # Set up backward
    if logits.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Gradient: (x > 0 ? 1 : 0) - target + (-sign(x) * exp(-|x|) / (1 + exp(-|x|)))
            # Simplified: sigmoid(x) - target
            sigmoid_x = 1.0 / (1.0 + np.exp(-logits.data))
            return grad * (sigmoid_x - target_tensor.data)

        loss_elem.parents = (Edge(logits, grad_fn),)
        loss_elem.op = "bce_with_logits"

    return mean(loss_elem)


def mse_loss(input: Tensor, target: Tensor | np.ndarray) -> Tensor:
    """Mean squared error loss.

    Args:
        input: Input tensor.
        target: Target tensor (same shape as input).

    Returns:
        Loss tensor (scalar).
    """
    # Convert target to tensor if needed
    if not isinstance(target, Tensor):
        target_tensor = tensor(target, dtype=input.dtype, requires_grad=False)
    else:
        target_tensor = target

    # MSE = mean((input - target)^2)

    diff = sub(input, target_tensor)
    squared = pow(diff, 2)
    return mean(squared)
