"""Type definitions and data structures for the autograd graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, Union

import numpy as np

# Type aliases
NDArray = np.ndarray
Shape = Tuple[int, ...]
DType = np.dtype

# DType union for type hints
DTypeLike = Union[np.dtype, type, str]


@dataclass(slots=True, frozen=True)
class Edge:
    """Represents an edge in the computation graph.

    Attributes:
        tensor: The child tensor (destination of the edge)
        grad_fn: Function to compute gradient w.r.t. this edge's source
    """

    tensor: object  # Will be Tensor type once defined
    grad_fn: callable | None = None


@dataclass(slots=True)
class Node:
    """Represents a node in the computation graph.

    Attributes:
        tensor: The tensor associated with this node
        op: Operation name that produced this tensor
        parents: Tuple of edges connecting to parent nodes
    """

    tensor: object  # Will be Tensor type once defined
    op: str
    parents: tuple[Edge, ...] = ()
