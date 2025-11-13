"""Low-level primitive operations with explicit backward functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jungrad.autograd import is_grad_enabled
from jungrad.types import Edge
from jungrad.utils import reduce_broadcasted_grad

if TYPE_CHECKING:
    from jungrad.tensor import Tensor
else:
    from jungrad.tensor import Tensor

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "pow",
    "exp",
    "log",
    "maximum",
    "sum",
    "mean",
    "max",
    "min",
    "var",
    "std",
    "matmul",
    "bmm",
    "transpose",
    "permute",
    "reshape",
    "flatten",
    "concat",
    "stack",
    "squeeze",
    "unsqueeze",
    "broadcast_to",
    "gather",
    "scatter_add",
    "slice",
    "take",
]


def add(a: Tensor, b: Tensor | float | int) -> Tensor:
    """Elementwise addition with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor or scalar.

    Returns:
        Result tensor.
    """
    from jungrad.tensor import Tensor

    # Convert scalar to tensor
    if not isinstance(b, Tensor):
        b = Tensor(np.array(b, dtype=a.dtype))

    # Forward
    out_data = np.add(a.data, b.data)
    out = Tensor(out_data, requires_grad=a.requires_grad or b.requires_grad, op="add")

    # Setup backward
    if out.requires_grad:
        parents = []
        if a.requires_grad:
            a_shape = a.shape

            def grad_fn_a(grad: np.ndarray) -> np.ndarray:
                return reduce_broadcasted_grad(grad, a_shape)

            parents.append(Edge(a, grad_fn_a))

        if b.requires_grad:
            b_shape = b.shape

            def grad_fn_b(grad: np.ndarray) -> np.ndarray:
                return reduce_broadcasted_grad(grad, b_shape)

            parents.append(Edge(b, grad_fn_b))

        out.parents = tuple(parents)

    return out


def sub(a: Tensor, b: Tensor | float | int) -> Tensor:
    """Elementwise subtraction with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor or scalar.

    Returns:
        Result tensor.
    """
    from jungrad.tensor import Tensor

    # Convert scalar to tensor
    if not isinstance(b, Tensor):
        b = Tensor(np.array(b, dtype=a.dtype))

    # Forward
    out_data = np.subtract(a.data, b.data)
    out = Tensor(out_data, requires_grad=a.requires_grad or b.requires_grad, op="sub")

    # Setup backward
    if out.requires_grad:
        parents = []
        if a.requires_grad:
            a_shape = a.shape

            def grad_fn_a(grad: np.ndarray) -> np.ndarray:
                return reduce_broadcasted_grad(grad, a_shape)

            parents.append(Edge(a, grad_fn_a))

        if b.requires_grad:
            b_shape = b.shape

            def grad_fn_b(grad: np.ndarray) -> np.ndarray:
                return -reduce_broadcasted_grad(grad, b_shape)

            parents.append(Edge(b, grad_fn_b))

        out.parents = tuple(parents)

    return out


def mul(a: Tensor, b: Tensor | float | int) -> Tensor:
    """Elementwise multiplication with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor or scalar.

    Returns:
        Result tensor.
    """
    from jungrad.tensor import Tensor

    # Convert scalar to tensor
    if not isinstance(b, Tensor):
        b = Tensor(np.array(b, dtype=a.dtype))

    # Forward
    out_data = np.multiply(a.data, b.data)
    out = Tensor(out_data, requires_grad=a.requires_grad or b.requires_grad, op="mul")

    # Setup backward
    if out.requires_grad:
        parents = []
        if a.requires_grad:
            a_shape = a.shape

            def grad_fn_a(grad: np.ndarray) -> np.ndarray:
                grad = grad * b.data
                return reduce_broadcasted_grad(grad, a_shape)

            parents.append(Edge(a, grad_fn_a))

        if b.requires_grad:
            b_shape = b.shape

            def grad_fn_b(grad: np.ndarray) -> np.ndarray:
                grad = grad * a.data
                return reduce_broadcasted_grad(grad, b_shape)

            parents.append(Edge(b, grad_fn_b))

        out.parents = tuple(parents)

    return out


def div(a: Tensor, b: Tensor | float | int) -> Tensor:
    """Elementwise division with broadcasting.

    Args:
        a: Numerator tensor.
        b: Denominator tensor or scalar.

    Returns:
        Result tensor.
    """
    from jungrad.tensor import Tensor

    # Convert scalar to tensor
    if not isinstance(b, Tensor):
        b = Tensor(np.array(b, dtype=a.dtype))

    # Forward
    out_data = np.divide(a.data, b.data)
    out = Tensor(out_data, requires_grad=a.requires_grad or b.requires_grad, op="div")

    # Setup backward
    if out.requires_grad:
        parents = []
        if a.requires_grad:
            a_shape = a.shape

            def grad_fn_a(grad: np.ndarray) -> np.ndarray:
                grad = grad / b.data
                return reduce_broadcasted_grad(grad, a_shape)

            parents.append(Edge(a, grad_fn_a))

        if b.requires_grad:
            b_shape = b.shape

            def grad_fn_b(grad: np.ndarray) -> np.ndarray:
                grad = -grad * a.data / (b.data**2)
                return reduce_broadcasted_grad(grad, b_shape)

            parents.append(Edge(b, grad_fn_b))

        out.parents = tuple(parents)

    return out


def neg(a: Tensor) -> Tensor:
    """Elementwise negation.

    Args:
        a: Input tensor.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.negative(a.data)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="neg")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return -grad

        out.parents = (Edge(a, grad_fn),)

    return out


def pow(a: Tensor, b: Tensor | float | int) -> Tensor:
    """Elementwise power operation.

    Args:
        a: Base tensor.
        b: Exponent tensor or scalar.

    Returns:
        Result tensor.
    """
    from jungrad.tensor import Tensor

    # Convert scalar to tensor
    if not isinstance(b, Tensor):
        b = Tensor(np.array(b, dtype=a.dtype))

    # Forward
    out_data = np.power(a.data, b.data)
    out = Tensor(out_data, requires_grad=a.requires_grad or b.requires_grad, op="pow")

    # Setup backward
    if out.requires_grad:
        parents = []
        if a.requires_grad:
            a_shape = a.shape

            def grad_fn_a(grad: np.ndarray) -> np.ndarray:
                # d/dx (x^y) = y * x^(y-1)
                grad = grad * b.data * np.power(a.data, b.data - 1)
                return reduce_broadcasted_grad(grad, a_shape)

            parents.append(Edge(a, grad_fn_a))

        if b.requires_grad:
            b_shape = b.shape

            def grad_fn_b(grad: np.ndarray) -> np.ndarray:
                # d/dy (x^y) = x^y * log(x)
                grad = grad * out_data * np.log(np.maximum(a.data, 1e-10))  # Avoid log(0)
                return reduce_broadcasted_grad(grad, b_shape)

            parents.append(Edge(b, grad_fn_b))

        out.parents = tuple(parents)

    return out


def exp(a: Tensor) -> Tensor:
    """Elementwise exponential.

    Args:
        a: Input tensor.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.exp(a.data)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="exp")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * out_data

        out.parents = (Edge(a, grad_fn),)

    return out


def log(a: Tensor) -> Tensor:
    """Elementwise natural logarithm.

    Args:
        a: Input tensor.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.log(a.data)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="log")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad / a.data

        out.parents = (Edge(a, grad_fn),)

    return out


def maximum(a: Tensor, b: Tensor | float | int) -> Tensor:
    """Elementwise maximum with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor or scalar.

    Returns:
        Result tensor.
    """
    from jungrad.tensor import Tensor

    if not isinstance(b, Tensor):
        b = Tensor(np.array(b, dtype=a.dtype))

    out_data = np.maximum(a.data, b.data)
    requires_grad = a.requires_grad or b.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad, op="maximum")

    if requires_grad:
        parents = []
        if a.requires_grad:
            a_shape = a.shape

            def grad_fn_a(grad: np.ndarray) -> np.ndarray:
                mask = (a.data > b.data).astype(grad.dtype)
                if b.requires_grad:
                    eq_mask = (a.data == b.data).astype(grad.dtype)
                    mask += 0.5 * eq_mask
                grad_contrib = grad * mask
                return reduce_broadcasted_grad(grad_contrib, a_shape)

            parents.append(Edge(a, grad_fn_a))

        if b.requires_grad:
            b_shape = b.shape

            def grad_fn_b(grad: np.ndarray) -> np.ndarray:
                mask = (b.data > a.data).astype(grad.dtype)
                if a.requires_grad:
                    eq_mask = (a.data == b.data).astype(grad.dtype)
                    mask += 0.5 * eq_mask
                grad_contrib = grad * mask
                return reduce_broadcasted_grad(grad_contrib, b_shape)

            parents.append(Edge(b, grad_fn_b))

        out.parents = tuple(parents)

    return out


# Reduction operations
def sum(
    a: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum elements over given axes.

    Args:
        a: Input tensor.
        axis: Axis or axes to sum over. None sums all.
        keepdims: Whether to keep reduced dimensions.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.sum(a.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="sum")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Expand grad to match input shape
            if keepdims:
                return np.broadcast_to(grad, a.shape)
            else:
                # Expand and broadcast
                expanded = np.expand_dims(grad, axis=axis) if axis is not None else grad
                return np.broadcast_to(expanded, a.shape)

        out.parents = (Edge(a, grad_fn),)

    return out


def mean(
    a: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Mean of elements over given axes.

    Args:
        a: Input tensor.
        axis: Axis or axes to reduce. None reduces all.
        keepdims: Whether to keep reduced dimensions.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.mean(a.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="mean")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Gradient of mean is grad / n
            if axis is None:
                n = a.data.size
            elif isinstance(axis, int):
                n = a.data.shape[axis]
            else:
                n = np.prod([a.data.shape[ax] for ax in axis])

            if keepdims:
                return np.broadcast_to(grad / n, a.shape)
            else:
                expanded = np.expand_dims(grad / n, axis=axis) if axis is not None else grad / n
                return np.broadcast_to(expanded, a.shape)

        out.parents = (Edge(a, grad_fn),)

    return out


def max(
    a: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Maximum over given axes.

    Args:
        a: Input tensor.
        axis: Axis or axes to reduce. None reduces all.
        keepdims: Whether to keep reduced dimensions.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.max(a.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="max")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Gradient is 1 at max locations, 0 elsewhere
            # Handle multiple max values (subgradient = 1/n where n is count)
            max_mask = a.data == np.max(a.data, axis=axis, keepdims=True)
            if axis is not None:
                # Count max values along axis
                if isinstance(axis, int):
                    count = np.sum(max_mask, axis=axis, keepdims=True)
                else:
                    count = np.sum(max_mask, axis=axis, keepdims=True)
                max_mask = max_mask / np.maximum(count, 1.0)

            if keepdims:
                grad_expanded = np.broadcast_to(grad, a.shape)
            else:
                expanded = np.expand_dims(grad, axis=axis) if axis is not None else grad
                grad_expanded = np.broadcast_to(expanded, a.shape)

            return grad_expanded * max_mask

        out.parents = (Edge(a, grad_fn),)

    return out


def min(
    a: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Minimum over given axes.

    Args:
        a: Input tensor.
        axis: Axis or axes to reduce. None reduces all.
        keepdims: Whether to keep reduced dimensions.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.min(a.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="min")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Similar to max, but for min
            min_mask = a.data == np.min(a.data, axis=axis, keepdims=True)
            if axis is not None:
                if isinstance(axis, int):
                    count = np.sum(min_mask, axis=axis, keepdims=True)
                else:
                    count = np.sum(min_mask, axis=axis, keepdims=True)
                min_mask = min_mask / np.maximum(count, 1.0)

            if keepdims:
                grad_expanded = np.broadcast_to(grad, a.shape)
            else:
                expanded = np.expand_dims(grad, axis=axis) if axis is not None else grad
                grad_expanded = np.broadcast_to(expanded, a.shape)

            return grad_expanded * min_mask

        out.parents = (Edge(a, grad_fn),)

    return out


def var(
    a: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    unbiased: bool = True,
) -> Tensor:
    """Variance over given axes.

    Args:
        a: Input tensor.
        axis: Axis or axes to reduce. None reduces all.
        keepdims: Whether to keep reduced dimensions.
        unbiased: If True, use Bessel's correction (N-1 denominator).

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.var(a.data, axis=axis, keepdims=keepdims, ddof=1 if unbiased else 0)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="var")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Gradient of variance: 2 * (x - mean) / n (or n-1 for unbiased)
            mean_val = np.mean(a.data, axis=axis, keepdims=True)
            if axis is None:
                n = a.data.size
            elif isinstance(axis, int):
                n = a.data.shape[axis]
            else:
                n = np.prod([a.data.shape[ax] for ax in axis])

            denom = n - 1 if unbiased else n
            diff = a.data - mean_val

            if keepdims:
                grad_expanded = np.broadcast_to(grad, a.shape)
            else:
                expanded = np.expand_dims(grad, axis=axis) if axis is not None else grad
                grad_expanded = np.broadcast_to(expanded, a.shape)

            return grad_expanded * 2 * diff / denom

        out.parents = (Edge(a, grad_fn),)

    return out


def std(
    a: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    unbiased: bool = True,
) -> Tensor:
    """Standard deviation over given axes.

    Args:
        a: Input tensor.
        axis: Axis or axes to reduce. None reduces all.
        keepdims: Whether to keep reduced dimensions.
        unbiased: If True, use Bessel's correction.

    Returns:
        Result tensor.
    """
    # Forward: std = sqrt(var)
    var_tensor = var(a, axis=axis, keepdims=keepdims, unbiased=unbiased)
    out_data = np.sqrt(var_tensor.data)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="std")

    # Setup backward: d/dx sqrt(var) = (1/2) * (1/sqrt(var)) * d(var)/dx
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Use var's backward but scale by 1/(2*std)
            var_grad_fn = var_tensor.parents[0].grad_fn if var_tensor.parents else None
            if var_grad_fn is not None:
                var_grad = var_grad_fn(grad)
                std_val = out_data
                # Expand std to match shape for broadcasting
                if keepdims:
                    std_expanded = np.broadcast_to(std_val, a.shape)
                else:
                    expanded = np.expand_dims(std_val, axis=axis) if axis is not None else std_val
                    std_expanded = np.broadcast_to(expanded, a.shape)
                # Avoid division by zero
                return var_grad / (2 * np.maximum(std_expanded, 1e-8))
            return np.zeros_like(a.data)

        out.parents = (Edge(a, grad_fn),)

    return out


# Linear algebra operations
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication (2D).

    Args:
        a: First tensor (..., M, N).
        b: Second tensor (..., N, K).

    Returns:
        Result tensor (..., M, K).
    """
    # Forward
    out_data = np.matmul(a.data, b.data)
    out = Tensor(out_data, requires_grad=a.requires_grad or b.requires_grad, op="matmul")

    # Setup backward
    if out.requires_grad:
        parents = []
        if a.requires_grad:

            def grad_fn_a(grad: np.ndarray) -> np.ndarray:
                # dA = grad @ B^T
                return np.matmul(grad, np.swapaxes(b.data, -2, -1))

            parents.append(Edge(a, grad_fn_a))

        if b.requires_grad:

            def grad_fn_b(grad: np.ndarray) -> np.ndarray:
                # dB = A^T @ grad
                return np.matmul(np.swapaxes(a.data, -2, -1), grad)

            parents.append(Edge(b, grad_fn_b))

        out.parents = tuple(parents)

    return out


def bmm(a: Tensor, b: Tensor) -> Tensor:
    """Batched matrix multiplication.

    Args:
        a: First tensor (B, M, N).
        b: Second tensor (B, N, K).

    Returns:
        Result tensor (B, M, K).
    """
    return matmul(a, b)  # matmul already handles batched case


def transpose(a: Tensor, dim0: int = 0, dim1: int = 1) -> Tensor:
    """Transpose two dimensions.

    Args:
        a: Input tensor.
        dim0: First dimension.
        dim1: Second dimension.

    Returns:
        Result tensor.
    """
    # Forward
    axes = list(range(len(a.shape)))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    out_data = np.transpose(a.data, axes)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="transpose")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Transpose back
            return np.transpose(grad, axes)

        out.parents = (Edge(a, grad_fn),)

    return out


def permute(a: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Permute dimensions.

    Args:
        a: Input tensor.
        dims: New dimension order.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.transpose(a.data, dims)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="permute")

    # Setup backward
    if out.requires_grad:
        # Inverse permutation
        inv_dims = tuple(np.argsort(dims))

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return np.transpose(grad, inv_dims)

        out.parents = (Edge(a, grad_fn),)

    return out


def reshape(a: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reshape tensor.

    Args:
        a: Input tensor.
        shape: Target shape.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.reshape(a.data, shape)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="reshape")

    # Setup backward
    if out.requires_grad:
        original_shape = a.shape

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return np.reshape(grad, original_shape)

        out.parents = (Edge(a, grad_fn),)

    return out


def flatten(a: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten tensor dimensions.

    Args:
        a: Input tensor.
        start_dim: Start dimension (inclusive).
        end_dim: End dimension (inclusive).

    Returns:
        Result tensor.
    """
    # Forward
    if end_dim == -1:
        end_dim = len(a.shape) - 1

    new_shape = list(a.shape)
    flattened_size = 1
    for i in range(start_dim, end_dim + 1):
        flattened_size *= new_shape[i]

    new_shape = new_shape[:start_dim] + [flattened_size] + new_shape[end_dim + 1 :]
    return reshape(a, tuple(new_shape))


def concat(tensors: tuple[Tensor, ...] | list[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors along dimension.

    Args:
        tensors: Input tensors.
        dim: Dimension to concatenate along.

    Returns:
        Result tensor.
    """
    from jungrad.tensor import Tensor

    # Forward
    out_data = np.concatenate([t.data for t in tensors], axis=dim)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=requires_grad, op="concat")

    # Setup backward
    if out.requires_grad:
        parents = []
        # Compute split points
        split_points = [0]
        for t in tensors:
            split_points.append(split_points[-1] + t.shape[dim])

        for i, t in enumerate(tensors):
            if t.requires_grad:
                start = split_points[i]
                end = split_points[i + 1]

                def make_grad_fn(start_idx, end_idx, tensor_shape):
                    def grad_fn(grad: np.ndarray) -> np.ndarray:
                        # Extract slice for this tensor
                        slices = [slice(None)] * len(grad.shape)
                        slices[dim] = slice(start_idx, end_idx)
                        return grad[tuple(slices)]

                    return grad_fn

                parents.append(Edge(t, make_grad_fn(start, end, t.shape)))

        out.parents = tuple(parents)

    return out


def stack(tensors: tuple[Tensor, ...] | list[Tensor], dim: int = 0) -> Tensor:
    """Stack tensors along new dimension.

    Args:
        tensors: Input tensors (must have same shape).
        dim: New dimension index.

    Returns:
        Result tensor.
    """
    from jungrad.tensor import Tensor

    # Forward
    out_data = np.stack([t.data for t in tensors], axis=dim)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=requires_grad, op="stack")

    # Setup backward
    if out.requires_grad:
        parents = []
        for i, t in enumerate(tensors):
            if t.requires_grad:

                def make_grad_fn(tensor_idx, tensor_dim):
                    def grad_fn(grad: np.ndarray) -> np.ndarray:
                        # Unstack: extract slice along new dimension
                        slices = [slice(None)] * len(grad.shape)
                        slices[tensor_dim] = tensor_idx
                        return grad[tuple(slices)]

                    return grad_fn

                parents.append(Edge(t, make_grad_fn(i, dim)))

        out.parents = tuple(parents)

    return out


def squeeze(a: Tensor, dim: int | tuple[int, ...] | None = None) -> Tensor:
    """Remove dimensions of size 1.

    Args:
        a: Input tensor.
        dim: Dimension(s) to squeeze. None squeezes all.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.squeeze(a.data, axis=dim)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="squeeze")

    # Setup backward
    if out.requires_grad:
        original_shape = a.shape

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return np.reshape(grad, original_shape)

        out.parents = (Edge(a, grad_fn),)

    return out


def unsqueeze(a: Tensor, dim: int) -> Tensor:
    """Add dimension of size 1.

    Args:
        a: Input tensor.
        dim: Position to insert dimension.

    Returns:
        Result tensor.
    """
    # Forward
    new_shape = list(a.shape)
    new_shape.insert(dim, 1)
    return reshape(a, tuple(new_shape))


def broadcast_to(a: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Broadcast tensor to shape.

    Args:
        a: Input tensor.
        shape: Target shape.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.broadcast_to(a.data, shape)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="broadcast_to")

    # Setup backward
    if out.requires_grad:
        original_shape = a.shape

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return reduce_broadcasted_grad(grad, original_shape)

        out.parents = (Edge(a, grad_fn),)

    return out


# Indexing operations
def gather(a: Tensor, dim: int, index: Tensor) -> Tensor:
    """Gather values along dimension using indices.

    Args:
        a: Input tensor.
        dim: Dimension to gather along.
        index: Index tensor (same shape as output, with values in [0, shape[dim]-1]).

    Returns:
        Result tensor.
    """
    from jungrad.tensor import Tensor

    # Forward
    # Create index tuple for advanced indexing
    indices = []
    for i in range(len(a.shape)):
        if i == dim:
            indices.append(index.data)
        else:
            # Create meshgrid-like indices
            shape_for_broadcast = list(a.shape)
            shape_for_broadcast[i] = 1
            indices.append(
                np.broadcast_to(np.arange(a.shape[i]).reshape(shape_for_broadcast), index.shape)
            )

    out_data = a.data[tuple(indices)]
    out = Tensor(out_data, requires_grad=a.requires_grad, op="gather")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # Scatter grad back to input positions
            grad_out = np.zeros_like(a.data)
            grad_out[tuple(indices)] = grad
            return grad_out

        out.parents = (Edge(a, grad_fn),)

    return out


def scatter_add(a: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """Scatter and add values from src to positions given by index.

    Args:
        a: Input tensor.
        dim: Dimension to scatter along.
        index: Index tensor.
        src: Source tensor to scatter from.

    Returns:
        Result tensor.
    """
    from jungrad.tensor import Tensor

    # Forward
    out_data = a.data.copy()
    indices = []
    for i in range(len(a.shape)):
        if i == dim:
            indices.append(index.data)
        else:
            shape_for_broadcast = list(a.shape)
            shape_for_broadcast[i] = 1
            indices.append(
                np.broadcast_to(np.arange(a.shape[i]).reshape(shape_for_broadcast), index.shape)
            )

    np.add.at(out_data, tuple(indices), src.data)
    requires_grad = a.requires_grad or src.requires_grad
    out = Tensor(out_data, requires_grad=requires_grad, op="scatter_add")

    # Setup backward
    if out.requires_grad:
        parents = []
        if a.requires_grad:

            def grad_fn_a(grad: np.ndarray) -> np.ndarray:
                return grad  # Gradient passes through

            parents.append(Edge(a, grad_fn_a))

        if src.requires_grad:

            def grad_fn_src(grad: np.ndarray) -> np.ndarray:
                # Extract gradient at scattered positions
                return grad[tuple(indices)]

            parents.append(Edge(src, grad_fn_src))

        out.parents = tuple(parents)

    return out


def slice(a: Tensor, dim: int, start: int, end: int) -> Tensor:
    """Slice tensor along dimension.

    Args:
        a: Input tensor.
        dim: Dimension to slice.
        start: Start index.
        end: End index (exclusive).

    Returns:
        Result tensor.
    """
    # Forward
    slices = [slice(None)] * len(a.shape)
    slices[dim] = slice(start, end)
    out_data = a.data[tuple(slices)]
    out = Tensor(out_data, requires_grad=a.requires_grad, op="slice")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            grad_out = np.zeros_like(a.data)
            grad_out[tuple(slices)] = grad
            return grad_out

        out.parents = (Edge(a, grad_fn),)

    return out


def take(a: Tensor, indices: Tensor) -> Tensor:
    """Take elements using flat indices.

    Args:
        a: Input tensor.
        indices: Flat indices.

    Returns:
        Result tensor.
    """
    # Forward
    out_data = np.take(a.data, indices.data)
    out = Tensor(out_data, requires_grad=a.requires_grad, op="take")

    # Setup backward
    if out.requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            grad_out = np.zeros_like(a.data)
            np.add.at(grad_out, indices.data.flatten(), grad.flatten())
            return grad_out

        out.parents = (Edge(a, grad_fn),)

    return out
