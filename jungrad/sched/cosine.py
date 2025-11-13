"""Cosine annealing learning rate scheduler."""

from __future__ import annotations

import numpy as np

__all__ = ["CosineLR"]


class CosineLR:
    """Cosine annealing learning rate scheduler.

    Optionally supports warmup.
    """

    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0.0,
        warmup_steps: int = 0,
    ):
        """Initialize cosine scheduler.

        Args:
            optimizer: Optimizer to schedule.
            T_max: Maximum number of iterations.
            eta_min: Minimum learning rate.
            warmup_steps: Number of warmup steps.
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def step(self) -> None:
        """Update learning rate."""
        self.last_epoch += 1

        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.last_epoch / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps)
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))

        self.optimizer.lr = float(lr)

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr
