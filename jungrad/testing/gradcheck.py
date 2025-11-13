"""Gradient checking via finite differences."""

from __future__ import annotations

from typing import Callable

import numpy as np

from jungrad.tensor import Tensor

__all__ = ["gradcheck"]


def gradcheck(
    func: Callable,
    inputs: tuple[Tensor, ...],
    eps: float = 1e-5,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
) -> bool:
    """Check gradients using finite differences.

    Args:
        func: Function to test.
        inputs: Input tensors.
        eps: Perturbation size for finite differences.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        raise_exception: Whether to raise on failure.

    Returns:
        True if check passes.

    Raises:
        AssertionError: If gradients don't match and raise_exception=True.
    """
    # Convert inputs to float64 for precision
    inputs_fp64 = []
    for inp in inputs:
        if inp.requires_grad:
            inputs_fp64.append(Tensor(inp.data.astype(np.float64), requires_grad=True))
        else:
            inputs_fp64.append(Tensor(inp.data.astype(np.float64), requires_grad=False))

    # Forward pass
    output = func(*inputs_fp64)
    if not isinstance(output, Tensor):
        raise ValueError("Function must return a Tensor")

    # Compute analytical gradient
    output.backward()

    # Check each input
    all_passed = True
    for i, (inp, inp_fp64) in enumerate(zip(inputs, inputs_fp64)):
        if not inp.requires_grad:
            continue

        if inp_fp64.grad is None:
            raise ValueError(f"Gradient not computed for input {i}")

        analytical_grad = inp_fp64.grad

        # Compute numerical gradient
        numerical_grad = np.zeros_like(inp_fp64.data)
        flat_input = inp_fp64.data.flatten()
        flat_grad = numerical_grad.flatten()

        for j in range(len(flat_input)):
            # Perturb forward
            flat_input[j] += eps
            output_plus = func(*inputs_fp64)
            loss_plus = float(
                output_plus.data.item() if output_plus.data.size == 1 else np.sum(output_plus.data)
            )

            # Perturb backward
            flat_input[j] -= 2 * eps
            output_minus = func(*inputs_fp64)
            loss_minus = float(
                output_minus.data.item()
                if output_minus.data.size == 1
                else np.sum(output_minus.data)
            )

            # Restore
            flat_input[j] += eps

            # Central difference
            flat_grad[j] = (loss_plus - loss_minus) / (2 * eps)

            # Zero out gradients for next iteration
            for inp in inputs_fp64:
                if inp.grad is not None:
                    inp.grad.fill(0)

        numerical_grad = flat_grad.reshape(inp_fp64.shape)

        # Compare
        diff = np.abs(analytical_grad - numerical_grad)
        max_diff = np.max(diff)
        max_analytical = np.max(np.abs(analytical_grad))
        max_numerical = np.max(np.abs(numerical_grad))

        # Relative error
        rel_error = max_diff / (atol + rtol * max(max_analytical, max_numerical))

        passed = rel_error < 1.0 or max_diff < atol

        if not passed:
            all_passed = False
            msg = (
                f"Gradient check failed for input {i}:\n"
                f"  Max absolute difference: {max_diff}\n"
                f"  Max analytical grad: {max_analytical}\n"
                f"  Max numerical grad: {max_numerical}\n"
                f"  Relative error: {rel_error}"
            )
            if raise_exception:
                raise AssertionError(msg)
            else:
                print(f"WARNING: {msg}")

    return all_passed
