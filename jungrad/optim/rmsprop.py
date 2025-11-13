"""RMSProp optimizer."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from jungrad.nn.module import Parameter

__all__ = ["RMSProp"]


class RMSProp:
    """RMSProp optimizer."""

    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        """Initialize RMSProp optimizer.

        Args:
            parameters: Iterator over parameters to optimize.
            lr: Learning rate.
            alpha: Smoothing constant.
            eps: Term added to denominator.
            weight_decay: Weight decay (L2 penalty).
            momentum: Momentum factor.
        """
        self.parameters = list(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

        # State: square_avg (running average of squared gradients), momentum buffer
        self.state: dict[Parameter, dict] = {}
        for param in self.parameters:
            self.state[param] = {"square_avg": np.zeros_like(param.data)}
            if momentum > 0:
                self.state[param]["momentum_buffer"] = np.zeros_like(param.data)

    def step(self) -> None:
        """Perform optimization step."""
        for param in self.parameters:
            if param.grad is None:
                continue

            state = self.state[param]
            grad = param.grad

            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Update running average of squared gradients
            state["square_avg"] = self.alpha * state["square_avg"] + (1 - self.alpha) * (grad**2)

            # Compute update
            avg = np.sqrt(state["square_avg"]) + self.eps
            update = grad / avg

            # Apply momentum if enabled
            if self.momentum > 0:
                state["momentum_buffer"] = self.momentum * state["momentum_buffer"] + update
                update = state["momentum_buffer"]

            # Update parameter
            param.data = param.data - self.lr * update

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients."""
        for param in self.parameters:
            param.zero_grad(set_to_none)
