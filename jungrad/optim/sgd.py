"""SGD optimizer with momentum and Nesterov."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from jungrad.nn.module import Parameter

__all__ = ["SGD"]


class SGD:
    """Stochastic Gradient Descent optimizer.

    Supports momentum and Nesterov acceleration.
    """

    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ):
        """Initialize SGD optimizer.

        Args:
            parameters: Iterator over parameters to optimize.
            lr: Learning rate.
            momentum: Momentum factor.
            nesterov: Whether to use Nesterov momentum.
            weight_decay: Weight decay (L2 penalty).
        """
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        # Velocity buffers for momentum
        self.state: dict[Parameter, dict] = {}
        for param in self.parameters:
            if momentum > 0:
                self.state[param] = {"velocity": np.zeros_like(param.data)}

    def step(self) -> None:
        """Perform optimization step."""
        for param in self.parameters:
            if param.grad is None:
                continue

            grad = param.grad

            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Apply momentum
            if self.momentum > 0:
                velocity = self.state[param]["velocity"]
                velocity = self.momentum * velocity + grad
                self.state[param]["velocity"] = velocity

                if self.nesterov:
                    grad = grad + self.momentum * velocity
                else:
                    grad = velocity

            # Update parameter
            param.data = param.data - self.lr * grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients.

        Args:
            set_to_none: If True, set grad to None instead of zeros.
        """
        for param in self.parameters:
            param.zero_grad(set_to_none)
