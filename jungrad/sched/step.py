"""Step and exponential learning rate schedulers."""

from __future__ import annotations

import numpy as np

__all__ = ["StepLR", "ExponentialLR"]


class StepLR:
    """Step learning rate scheduler.

    Decays lr by gamma every step_size epochs.
    """

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        """Initialize step scheduler.

        Args:
            optimizer: Optimizer to schedule.
            step_size: Period of learning rate decay.
            gamma: Multiplicative factor.
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self) -> None:
        """Update learning rate."""
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr


class ExponentialLR:
    """Exponential learning rate scheduler.

    Decays lr by gamma every epoch.
    """

    def __init__(self, optimizer, gamma: float):
        """Initialize exponential scheduler.

        Args:
            optimizer: Optimizer to schedule.
            gamma: Multiplicative factor.
        """
        self.optimizer = optimizer
        self.gamma = gamma
        self.last_epoch = 0

    def step(self) -> None:
        """Update learning rate."""
        self.last_epoch += 1
        self.optimizer.lr *= self.gamma

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr
