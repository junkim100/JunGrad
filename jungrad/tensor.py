"""N-D Tensor class with autograd support."""

from __future__ import annotations

from typing import Optional

import numpy as np

from jungrad.autograd import backward as backward_fn, is_grad_enabled
from jungrad.types import Edge, NDArray
from jungrad.utils import asarray

__all__ = ["Tensor", "tensor", "zeros", "ones", "randn", "arange", "full"]


class Tensor:
    """N-D Tensor with automatic differentiation.

    Attributes:
        data: Underlying numpy array.
        grad: Gradient array (None if not computed).
        requires_grad: Whether this tensor requires gradients.
        op: Operation name that created this tensor.
        parents: Tuple of edges to parent tensors.
        name: Optional name for debugging.
        _retain_grad: Whether to retain gradient for non-leaf nodes.
    """

    def __init__(
        self,
        data: NDArray | list | float | int,
        requires_grad: bool = False,
        op: str = "",
        parents: tuple[Edge, ...] = (),
        name: Optional[str] = None,
    ):
        """Initialize tensor.

        Args:
            data: Tensor data (will be converted to numpy array.
            requires_grad: Whether gradients are needed.
            op: Operation name.
            parents: Parent edges in computation graph.
            name: Optional name.
        """
        self.data = asarray(data)
        self.grad: Optional[NDArray] = None
        self.requires_grad = requires_grad and is_grad_enabled()
        self.op = op
        self.parents = parents
        self.name = name
        self._retain_grad = False

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor shape."""
        return self.data.shape

    @property
    def dtype(self):
        """Tensor dtype."""
        return self.data.dtype

    def is_leaf(self) -> bool:
        """Check if tensor is a leaf (no parents)."""
        return len(self.parents) == 0

    def detach(self) -> Tensor:
        """Return a new tensor detached from computation graph."""
        return Tensor(self.data.copy(), requires_grad=False, name=self.name)

    def retain_grad(self) -> None:
        """Mark tensor to retain gradient during backward."""
        self._retain_grad = True

    def item(self) -> float | int | complex:
        """Return scalar value if tensor has one element."""
        return self.data.item()

    def numpy(self) -> NDArray:
        """Return underlying numpy array."""
        return self.data

    def astype(self, dtype) -> Tensor:
        """Return tensor with new dtype."""
        return Tensor(self.data.astype(dtype), requires_grad=self.requires_grad, name=self.name)

    def to(self, dtype) -> Tensor:
        """Alias for astype."""
        return self.astype(dtype)

    def backward(self, grad: Optional[NDArray] = None) -> None:
        """Run backward pass starting from this tensor.

        Args:
            grad: Initial gradient. Defaults to ones_like(data).
        """
        if not self.requires_grad:
            return
        backward_fn(self, grad)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero out gradient.

        Args:
            set_to_none: If True, set grad to None instead of zeros.
        """
        if set_to_none:
            self.grad = None
        elif self.grad is not None:
            self.grad.fill(0)

    # Operator overloads
    def __add__(self, other):
        from jungrad.ops import add

        return add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from jungrad.ops import sub

        return sub(self, other)

    def __rsub__(self, other):
        from jungrad.ops import sub

        return sub(other, self)

    def __mul__(self, other):
        from jungrad.ops import mul

        return mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from jungrad.ops import div

        return div(self, other)

    def __rtruediv__(self, other):
        from jungrad.ops import div

        return div(other, self)

    def __neg__(self):
        from jungrad.ops import neg

        return neg(self)

    def __pow__(self, other):
        from jungrad.ops import pow

        return pow(self, other)

    def __matmul__(self, other):
        from jungrad.ops import matmul

        return matmul(self, other)

    def sum(self, axis=None, keepdims=False):
        """Sum elements over given axes.

        Args:
            axis: Axis or axes to sum over. None sums all.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            Summed tensor.
        """
        from jungrad.ops import sum as sum_op

        return sum_op(self, axis=axis, keepdims=keepdims)

    def __repr__(self) -> str:
        """String representation."""
        name_str = f", name='{self.name}'" if self.name else ""
        grad_str = f", grad={self.grad is not None}" if self.requires_grad else ""
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad}{grad_str}{name_str})"


# Creation helpers
def tensor(
    data: NDArray | list | float | int,
    requires_grad: bool = False,
    dtype=None,
    name: Optional[str] = None,
) -> Tensor:
    """Create a tensor from data.

    Args:
        data: Input data.
        requires_grad: Whether gradients are needed.
        dtype: Optional dtype.
        name: Optional name.

    Returns:
        New tensor.
    """
    arr = asarray(data, dtype=dtype)
    return Tensor(arr, requires_grad=requires_grad, name=name)


def zeros(
    shape: tuple[int, ...] | int,
    requires_grad: bool = False,
    dtype=None,
    name: Optional[str] = None,
) -> Tensor:
    """Create tensor filled with zeros.

    Args:
        shape: Tensor shape.
        requires_grad: Whether gradients are needed.
        dtype: Optional dtype (defaults to float32).
        name: Optional name.

    Returns:
        New tensor.
    """
    if dtype is None:
        dtype = np.float32
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad, name=name)


def ones(
    shape: tuple[int, ...] | int,
    requires_grad: bool = False,
    dtype=None,
    name: Optional[str] = None,
) -> Tensor:
    """Create tensor filled with ones.

    Args:
        shape: Tensor shape.
        requires_grad: Whether gradients are needed.
        dtype: Optional dtype (defaults to float32).
        name: Optional name.

    Returns:
        New tensor.
    """
    if dtype is None:
        dtype = np.float32
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad, name=name)


def randn(
    *shape: int,
    requires_grad: bool = False,
    dtype=None,
    name: Optional[str] = None,
) -> Tensor:
    """Create tensor with random normal values.

    Args:
        *shape: Tensor shape dimensions.
        requires_grad: Whether gradients are needed.
        dtype: Optional dtype (defaults to float32).
        name: Optional name.

    Returns:
        New tensor.
    """
    if dtype is None:
        dtype = np.float32
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad, name=name)


def arange(
    start: int | float,
    end: int | float | None = None,
    step: int | float = 1,
    requires_grad: bool = False,
    dtype=None,
    name: Optional[str] = None,
) -> Tensor:
    """Create tensor with evenly spaced values.

    Args:
        start: Start value (or end if end is None).
        end: End value (exclusive).
        step: Step size.
        requires_grad: Whether gradients are needed.
        dtype: Optional dtype.
        name: Optional name.

    Returns:
        New tensor.
    """
    arr = np.arange(start, end, step, dtype=dtype)
    return Tensor(arr, requires_grad=requires_grad, name=name)


def full(
    shape: tuple[int, ...] | int,
    fill_value: float | int,
    requires_grad: bool = False,
    dtype=None,
    name: Optional[str] = None,
) -> Tensor:
    """Create tensor filled with a value.

    Args:
        shape: Tensor shape.
        fill_value: value to fill with.
        requires_grad: Whether gradients are needed.
        dtype: Optional dtype.
        name: Optional name.

    Returns:
        New tensor.
    """
    if isinstance(shape, int):
        shape = (shape,)
    arr = np.full(shape, fill_value, dtype=dtype)
    return Tensor(arr, requires_grad=requires_grad, name=name)
