"""Graphviz export for computation graphs."""

from __future__ import annotations

from typing import Optional

import numpy as np

from jungrad.autograd import toposort
from jungrad.tensor import Tensor

__all__ = ["export_graph", "to_dot"]


def export_graph(output: Tensor, filename: str) -> None:
    """Export computation graph to DOT file.

    Args:
        output: Output tensor.
        filename: Output filename.
    """
    dot_str = to_dot(output)
    with open(filename, "w") as f:
        f.write(dot_str)


def to_dot(output: Tensor, max_nodes: int = 100) -> str:
    """Convert computation graph to DOT format.

    Args:
        output: Output tensor.
        max_nodes: Maximum number of nodes to include.

    Returns:
        DOT format string.
    """
    # Get all nodes in topological order
    topo = toposort(output)

    if len(topo) > max_nodes:
        topo = topo[:max_nodes]

    lines = ["digraph computation_graph {", "  rankdir=LR;"]

    # Create nodes
    node_ids = {tensor: i for i, tensor in enumerate(topo)}

    for i, tensor in enumerate(topo):
        shape_str = "x".join(str(d) for d in tensor.shape)
        grad_str = ""
        if tensor.grad is not None:
            grad_norm = float(np.linalg.norm(tensor.grad))
            grad_str = f", grad_norm={grad_norm:.3f}"

        name = tensor.name or f"tensor_{i}"
        op = tensor.op or "leaf"

        label = f"{name}\\n{op}\\nshape={shape_str}{grad_str}"

        color = "lightblue" if tensor.is_leaf() else "lightgreen"
        lines.append(f'  node{i} [label="{label}", style=filled, fillcolor={color}];')

    # Create edges
    for i, tensor in enumerate(topo):
        for edge in tensor.parents:
            parent = edge.tensor
            if parent in node_ids:
                parent_id = node_ids[parent]
                lines.append(f"  node{parent_id} -> node{i};")

    lines.append("}")
    return "\n".join(lines)
