"""Neural network layers."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from jungrad.backend import get_array_module, to_device_array, to_numpy_array
from jungrad.nn.module import Module, Parameter
from jungrad.ops import add, matmul
from jungrad.tensor import Tensor, randn, zeros, ones

__all__ = [
    "Sequential",
    "Stack",
    "Linear",
    "Conv1d",
    "Embedding",
    "ReLU",
    "LayerNorm",
    "Dropout",
]


# ============================================================================
# Containers
# ============================================================================


class Sequential(Module):
    """Sequential container of modules."""

    def __init__(self, *modules: Module):
        """Initialize sequential container.

        Args:
            *modules: Modules to apply in sequence.
        """
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all modules.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        for module in self._modules.values():
            x = module(x)
        return x


class Stack(Module):
    """Stack container that applies modules independently and stacks outputs."""

    def __init__(self, *modules: Module):
        """Initialize stack container.

        Args:
            *modules: Modules to apply independently.
        """
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all modules and stack.

        Args:
            x: Input tensor.

        Returns:
            Stacked output tensor.
        """
        from jungrad.ops import stack

        outputs = [module(x) for module in self._modules.values()]
        return stack(outputs, dim=0)


# ============================================================================
# Layers
# ============================================================================


class Linear(Module):
    """Linear (fully connected) layer.

    Computes: output = input @ weight.T + bias
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Initialize linear layer.

        Args:
            in_features: Input feature size.
            out_features: Output feature size.
            bias: Whether to use bias.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight: (out_features, in_features)
        weight = randn(out_features, in_features, requires_grad=True)
        self.register_parameter("weight", Parameter(weight.data, name="weight"))

        if bias:
            bias_tensor = zeros(out_features, requires_grad=True)
            self.register_parameter("bias", Parameter(bias_tensor.data, name="bias"))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
        """
        from jungrad.ops import transpose

        # Use Parameter directly (it's a Tensor subclass)
        weight = self._parameters["weight"]
        weight_T = transpose(weight, dim0=0, dim1=1)
        output = matmul(x, weight_T)

        if self._parameters["bias"] is not None:
            bias = self._parameters["bias"]
            output = add(output, bias)

        return output

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self._parameters['bias'] is not None})"


class Conv1d(Module):
    """1D convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        """Initialize Conv1d layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of convolution kernel.
            stride: Stride of convolution.
            padding: Zero-padding added to both sides.
            bias: Whether to use bias.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight: (out_channels, in_channels, kernel_size)
        weight = randn(out_channels, in_channels, kernel_size, requires_grad=True)
        self.register_parameter("weight", Parameter(weight.data, name="weight"))

        if bias:
            bias_tensor = zeros(out_channels, requires_grad=True)
            self.register_parameter("bias", Parameter(bias_tensor.data, name="bias"))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, in_channels, length).

        Returns:
            Output tensor (batch, out_channels, out_length).
        """
        import numpy as np
        from jungrad.ops import add, mul
        from jungrad.tensor import tensor as tensor_fn, zeros
        from jungrad.ops import slice

        # Add padding
        if self.padding > 0:
            padded_shape = list(x.shape)
            padded_shape[-1] += 2 * self.padding
            padded = zeros(tuple(padded_shape), dtype=x.dtype)
            # Copy x into padded (simplified - would use proper slicing in production)
            # For now, use numpy operations then wrap
            padded_np = np.zeros(padded_shape, dtype=x.dtype)
            padded_np[..., self.padding : -self.padding if self.padding > 0 else None] = x.data
            padded = tensor_fn(padded_np, requires_grad=x.requires_grad)
        else:
            padded = x

        batch_size, in_channels, length = padded.shape
        out_length = (length - self.kernel_size) // self.stride + 1

        # Initialize output
        output_np = np.zeros((batch_size, self.out_channels, out_length))

        weight_data = self._parameters["weight"].data
        for b in range(batch_size):
            for o in range(self.out_channels):
                for i in range(out_length):
                    start = i * self.stride
                    end = start + self.kernel_size
                    # Extract window
                    window = padded.data[b, :, start:end]  # (in_channels, kernel_size)
                    # Convolve: sum(window * weight[o])
                    output_np[b, o, i] = np.sum(window * weight_data[o])

        output = tensor_fn(
            output_np,
            requires_grad=x.requires_grad or self._parameters["weight"].requires_grad,
        )

        # Set up backward manually (simplified - full backward would be more complex)
        if output.requires_grad:

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                # Backward through conv1d
                grad_input = np.zeros_like(x.data)
                grad_weight = np.zeros_like(weight_data)

                # Compute gradients (simplified implementation)
                for b in range(batch_size):
                    for o in range(self.out_channels):
                        for i in range(out_length):
                            start = i * self.stride
                            end = start + self.kernel_size
                            g = grad[b, o, i]

                            # Gradient w.r.t. input
                            if self.padding > 0:
                                pad_start = start - self.padding
                                pad_end = end - self.padding
                                if pad_start >= 0 and pad_end <= x.shape[-1]:
                                    grad_input[b, :, pad_start:pad_end] += g * weight_data[o]
                            else:
                                grad_input[b, :, start:end] += g * weight_data[o]

                            # Gradient w.r.t. weight
                            window = padded.data[b, :, start:end]
                            grad_weight[o] += g * window

                return grad_input

            from jungrad.types import Edge

            output.parents = (Edge(x, grad_fn),)
            output.op = "conv1d"

        # Add bias if present
        if self._parameters["bias"] is not None:
            bias_tensor = Tensor(
                self._parameters["bias"].data,
                requires_grad=self._parameters["bias"].requires_grad,
            )
            # Broadcast bias: (out_channels,) -> (batch, out_channels, out_length)
            bias_expanded = tensor_fn(
                np.expand_dims(np.expand_dims(bias_tensor.data, 0), -1),
                requires_grad=bias_tensor.requires_grad,
            )
            output = add(output, bias_expanded)

        return output

    def __repr__(self) -> str:
        return f"Conv1d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self._parameters['bias'] is not None})"


def _sinusoidal_table(max_len: int, dim: int, device: str) -> np.ndarray:
    """Create sinusoidal positional encodings."""

    xp = get_array_module(device)
    position = xp.arange(max_len, dtype=xp.float32)[:, None]
    div_term = xp.exp(xp.arange(0, dim, 2, dtype=xp.float32) * -(math.log(10000.0) / dim))
    table = xp.zeros((max_len, dim), dtype=xp.float32)
    table[:, 0::2] = xp.sin(position * div_term)
    table[:, 1::2] = xp.cos(position * div_term)
    return to_numpy_array(table)


class Embedding(Module):
    """Embedding layer.

    Maps integer indices to dense vectors.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        *,
        positional_encoding: Optional[str] = None,
        max_position_embeddings: Optional[int] = None,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        """Initialize embedding layer.

        Args:
            num_embeddings: Size of embedding dictionary.
            embedding_dim: Size of each embedding vector.
            padding_idx: If specified, pad embedding at this index.
        """
        super().__init__()
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.device = device.lower()
        self.positional_encoding = positional_encoding.lower() if positional_encoding else None
        self.max_position_embeddings = max_position_embeddings
        self._dropout = Dropout(dropout) if dropout > 0 else None

        # Initialize weights
        weight = randn(num_embeddings, embedding_dim, requires_grad=True)
        if padding_idx is not None:
            weight.data[padding_idx] = 0.0

        self.register_parameter("weight", Parameter(weight.data, name="weight"))

        # Positional encodings
        if self.positional_encoding is not None:
            if max_position_embeddings is None:
                raise ValueError(
                    "max_position_embeddings is required when positional_encoding is enabled"
                )
            if self.positional_encoding not in {"sinusoidal", "learned"}:
                raise ValueError("positional_encoding must be 'sinusoidal', 'learned', or None")
            if self.positional_encoding == "learned":
                position_weight = randn(max_position_embeddings, embedding_dim, requires_grad=True)
                self.register_parameter(
                    "position_embeddings",
                    Parameter(position_weight.data, name="position_embeddings"),
                )
            else:
                sinusoidal = _sinusoidal_table(max_position_embeddings, embedding_dim, self.device)
                self.register_buffer(
                    "position_embeddings",
                    Tensor(sinusoidal, requires_grad=False, name="position_embeddings"),
                )
        else:
            self.register_parameter("position_embeddings", None)

    def _default_position_ids(self, shape: tuple[int, ...]) -> np.ndarray:
        if len(shape) == 0:
            return np.zeros((1,), dtype=np.int64)
        position_grid = np.indices(shape, dtype=np.int64)[-1]
        return position_grid

    def forward(
        self, x: Tensor | np.ndarray, position_ids: Tensor | np.ndarray | None = None
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of indices (long tensor, any shape).

        Returns:
            Output tensor (..., embedding_dim).
        """
        # Convert to numpy if needed
        if isinstance(x, Tensor):
            indices_np = x.data.astype(np.int64)
        else:
            indices_np = np.asarray(x, dtype=np.int64)
        original_shape = indices_np.shape
        indices_flat = indices_np.flatten()

        # Lookup embeddings
        weight_data = to_device_array(self._parameters["weight"].data, self.device)
        xp = get_array_module(self.device)
        indices_device = to_device_array(indices_flat, self.device).astype(np.int64)
        embeddings = weight_data[indices_device]  # (N, embedding_dim)
        output_shape = original_shape + (self.embedding_dim,)
        output_xp = embeddings.reshape(output_shape)

        # Positional encoding
        if self.positional_encoding is not None:
            if position_ids is None:
                positions = self._default_position_ids(original_shape)
            elif isinstance(position_ids, Tensor):
                positions = position_ids.data.astype(np.int64)
            else:
                positions = np.asarray(position_ids, dtype=np.int64)
            if positions.shape != original_shape:
                raise ValueError("position_ids must have the same shape as input indices")
            max_pos = int(positions.max()) if positions.size else 0
            if max_pos >= int(self.max_position_embeddings or 0):
                raise ValueError("position_ids exceed max_position_embeddings; increase the limit")
            pos_flat = positions.reshape(-1)
            if self.positional_encoding == "learned":
                pos_table = to_device_array(
                    self._parameters["position_embeddings"].data, self.device
                )
            else:
                pos_table = to_device_array(self._buffers["position_embeddings"].data, self.device)
            pos_embeddings = pos_table[pos_flat].reshape(output_shape)
            output_xp = output_xp + pos_embeddings

        output_np = to_numpy_array(output_xp)
        output = Tensor(output_np, requires_grad=self._parameters["weight"].requires_grad)

        # Set up backward
        if output.requires_grad:
            # Get array module for device consistency
            weight_data = self._parameters["weight"].data
            xp = get_array_module(self.device)

            # Store indices on the correct device for backward pass
            indices_device = to_device_array(indices_flat, self.device).astype(np.int64)

            def grad_fn(grad: np.ndarray) -> np.ndarray:
                # Convert grad to device array if needed
                grad_device = to_device_array(grad, self.device)

                # Gradient w.r.t. weight: scatter grad to weight positions
                grad_weight = xp.zeros_like(weight_data)
                grad_flat = grad_device.reshape(-1, self.embedding_dim)

                # Use vectorized scatter operation
                # Convert indices to device array
                indices_dev = to_device_array(indices_device, self.device).astype(xp.int64)

                # Use advanced indexing to scatter gradients
                # This is more efficient than a loop and handles device consistency
                unique_indices = xp.unique(indices_dev)
                for idx in unique_indices:
                    # Create mask on the same device
                    mask = indices_dev == idx
                    # Sum gradients for this index
                    grad_sum = xp.sum(grad_flat[mask], axis=0)
                    # Ensure grad_sum is a proper array on the device
                    if not isinstance(grad_sum, xp.ndarray):
                        grad_sum = xp.asarray(grad_sum)
                    grad_sum = to_device_array(grad_sum, self.device)
                    # Convert idx to Python int for indexing
                    idx_int = int(to_numpy_array(idx).item())
                    # Perform addition on the same device
                    grad_weight[idx_int] = xp.add(grad_weight[idx_int], grad_sum)

                # Convert back to NumPy if needed for parameter update
                return to_numpy_array(grad_weight)

            from jungrad.types import Edge

            output.parents = (Edge(self._parameters["weight"], grad_fn),)
            output.op = "embedding"

        if self._dropout is not None:
            output = self._dropout(output)

        return output

    def __repr__(self) -> str:
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx})"


# ============================================================================
# Activations
# ============================================================================


class ReLU(Module):
    """ReLU activation layer."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with ReLU applied.
        """
        from jungrad.functional import relu

        return relu(x)

    def __repr__(self) -> str:
        return "ReLU()"


# ============================================================================
# Normalization & Regularization
# ============================================================================


class LayerNorm(Module):
    """Layer normalization."""

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """Initialize layer norm.

        Args:
            normalized_shape: Shape to normalize over.
            eps: Epsilon for numerical stability.
            affine: Whether to use learnable affine parameters.
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.affine = affine

        if affine:
            weight = ones(normalized_shape, requires_grad=True)
            bias_tensor = zeros(normalized_shape, requires_grad=True)
            self.register_parameter("weight", Parameter(weight.data, name="weight"))
            self.register_parameter("bias", Parameter(bias_tensor.data, name="bias"))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        from jungrad.ops import mean, var, add, mul, sub, div, pow
        from jungrad.tensor import tensor
        import numpy as np

        # Compute mean and var over last len(normalized_shape) dimensions
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean_val = mean(x, axis=dims, keepdims=True)
        var_val = var(x, axis=dims, keepdims=True, unbiased=False)

        # Normalize: (x - mean) / sqrt(var + eps)
        eps_tensor = tensor(self.eps, dtype=x.dtype)
        std_val = add(var_val, eps_tensor)

        # sqrt via pow(0.5)
        std_val = pow(std_val, 0.5)

        normalized = div(sub(x, mean_val), std_val)

        # Apply affine transformation if enabled
        if self.affine:
            weight_tensor = Tensor(
                self._parameters["weight"].data,
                requires_grad=self._parameters["weight"].requires_grad,
            )
            bias_tensor = Tensor(
                self._parameters["bias"].data,
                requires_grad=self._parameters["bias"].requires_grad,
            )
            normalized = add(mul(normalized, weight_tensor), bias_tensor)

        return normalized

    def __repr__(self) -> str:
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps}, affine={self.affine})"


class Dropout(Module):
    """Dropout layer."""

    def __init__(self, p: float = 0.5):
        """Initialize dropout.

        Args:
            p: Dropout probability.
        """
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        if not self._training or self.p == 0.0:
            return x

        # Generate dropout mask
        mask = np.random.binomial(1, 1 - self.p, size=x.shape).astype(x.dtype)
        mask = mask / (1 - self.p)  # Scale to preserve expectation

        mask_tensor = Tensor(mask, requires_grad=False)
        from jungrad.ops import mul

        return mul(x, mask_tensor)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
