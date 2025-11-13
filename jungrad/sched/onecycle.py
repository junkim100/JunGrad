"""One-cycle learning rate scheduler."""

from __future__ import annotations

import numpy as np

__all__ = ["OneCycleLR"]


class OneCycleLR:
    """One-cycle learning rate policy.

    Increases lr from initial to max_lr, then decreases following cosine.
    """

    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
    ):
        """Initialize one-cycle scheduler.

        Args:
            optimizer: Optimizer to schedule.
            max_lr: Maximum learning rate.
            total_steps: Total number of steps.
            pct_start: Percentage of steps for warmup.
            anneal_strategy: 'cos' or 'linear'.
            div_factor: Initial lr = max_lr / div_factor.
            final_div_factor: Final lr = initial_lr / final_div_factor.
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        self.warmup_steps = int(total_steps * pct_start)
        self.last_epoch = 0

        # Set initial lr
        self.optimizer.lr = self.initial_lr

    def step(self) -> None:
        """Update learning rate."""
        self.last_epoch += 1

        if self.last_epoch <= self.warmup_steps:
            # Warmup: linear increase
            progress = self.last_epoch / self.warmup_steps
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Annealing: cosine or linear decrease
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            if self.anneal_strategy == "cos":
                lr = self.final_lr + (self.max_lr - self.final_lr) * 0.5 * (
                    1 + np.cos(np.pi * progress)
                )
            else:  # linear
                lr = self.max_lr + (self.final_lr - self.max_lr) * progress

        self.optimizer.lr = float(lr)

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr
