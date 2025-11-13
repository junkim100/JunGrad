"""Adam and AdamW optimizers."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from jungrad.nn.module import Parameter

__all__ = ["Adam", "AdamW"]


class Adam:
    """Adam optimizer."""

    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """Initialize Adam optimizer.

        Args:
            parameters: Iterator over parameters to optimize.
            lr: Learning rate.
            betas: Coefficients for moving averages of gradient and its square.
            eps: Term added to denominator for numerical stability.
            weight_decay: Weight decay (L2 penalty).
        """
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # State: step, exp_avg (momentum), exp_avg_sq (variance)
        self.state: dict[Parameter, dict] = {}
        for param in self.parameters:
            self.state[param] = {
                "step": 0,
                "exp_avg": np.zeros_like(param.data),
                "exp_avg_sq": np.zeros_like(param.data),
            }

    def step(self) -> None:
        """Perform optimization step."""
        for param in self.parameters:
            if param.grad is None:
                continue

            state = self.state[param]
            state["step"] += 1
            grad = param.grad

            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # Update biased moment estimates
            state["exp_avg"] = self.beta1 * state["exp_avg"] + (1 - self.beta1) * grad
            state["exp_avg_sq"] = self.beta2 * state["exp_avg_sq"] + (1 - self.beta2) * (grad**2)

            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state["step"]
            bias_correction2 = 1 - self.beta2 ** state["step"]

            step_size = self.lr / bias_correction1

            # Update parameter
            denom = np.sqrt(state["exp_avg_sq"] / bias_correction2) + self.eps
            param.data = param.data - step_size * state["exp_avg"] / denom

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients."""
        for param in self.parameters:
            param.zero_grad(set_to_none)


class AdamW:
    """AdamW optimizer (decoupled weight decay)."""

    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """Initialize AdamW optimizer.

        Args:
            parameters: Iterator over parameters to optimize.
            lr: Learning rate.
            betas: Coefficients for moving averages.
            eps: Term added to denominator.
            weight_decay: Decoupled weight decay coefficient.
        """
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # State: step, exp_avg, exp_avg_sq
        self.state: dict[Parameter, dict] = {}
        for param in self.parameters:
            self.state[param] = {
                "step": 0,
                "exp_avg": np.zeros_like(param.data),
                "exp_avg_sq": np.zeros_like(param.data),
            }

    def step(self) -> None:
        """Perform optimization step."""
        for param in self.parameters:
            if param.grad is None:
                continue

            state = self.state[param]
            state["step"] += 1
            grad = param.grad

            # Update biased moment estimates
            state["exp_avg"] = self.beta1 * state["exp_avg"] + (1 - self.beta1) * grad
            state["exp_avg_sq"] = self.beta2 * state["exp_avg_sq"] + (1 - self.beta2) * (grad**2)

            # Bias correction
            bias_correction1 = 1 - self.beta1 ** state["step"]
            bias_correction2 = 1 - self.beta2 ** state["step"]

            step_size = self.lr / bias_correction1
            denom = np.sqrt(state["exp_avg_sq"] / bias_correction2) + self.eps

            # Update parameter (decoupled weight decay)
            param.data = param.data - step_size * (
                state["exp_avg"] / denom + self.weight_decay * param.data
            )

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients."""
        for param in self.parameters:
            param.zero_grad(set_to_none)
