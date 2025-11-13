"""Utility functions for neural networks."""

from __future__ import annotations

from jungrad.nn.module import Module, Parameter

__all__ = ["count_params", "freeze", "unfreeze", "named_parameters", "summary"]


def count_params(module: Module) -> int:
    """Count number of parameters in module.

    Args:
        module: Module to count parameters for.

    Returns:
        Total number of parameters.
    """
    return sum(p.data.size for p in module.parameters())


def freeze(module: Module) -> None:
    """Freeze all parameters in module.

    Args:
        module: Module to freeze.
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module: Module) -> None:
    """Unfreeze all parameters in module.

    Args:
        module: Module to unfreeze.
    """
    for param in module.parameters():
        param.requires_grad = True


def named_parameters(module: Module, prefix: str = "", recurse: bool = True):
    """Get named parameters from module.

    Args:
        module: Module to get parameters from.
        prefix: Name prefix.
        recurse: Whether to recurse into submodules.

    Yields:
        (name, parameter) tuples.
    """
    yield from module.named_parameters(prefix=prefix, recurse=recurse)


def summary(module: Module, input_shape: tuple[int, ...] | None = None) -> str:
    """Get summary of module.

    Args:
        module: Module to summarize.
        input_shape: Optional input shape for computing output shapes.

    Returns:
        Summary string.
    """
    lines = [f"{module.__class__.__name__}"]
    lines.append(f"  Parameters: {count_params(module):,}")

    if input_shape:
        # Try to compute output shape (simplified)
        lines.append(f"  Input shape: {input_shape}")
        try:
            # This would require actually running forward - simplified for now
            lines.append(f"  Output shape: (computed)")
        except:
            pass

    return "\n".join(lines)
