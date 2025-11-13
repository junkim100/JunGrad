"""Base Module and Parameter classes."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterator, Optional

import numpy as np

from jungrad.tensor import Tensor

__all__ = ["Module", "Parameter"]


class Parameter(Tensor):
    """A parameter tensor that always requires gradients."""

    def __init__(self, data: np.ndarray | list | float | int, name: Optional[str] = None):
        """Initialize parameter.

        Args:
            data: Parameter data.
            name: Optional name.
        """
        super().__init__(data, requires_grad=True, name=name)


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        """Initialize module."""
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._buffers: OrderedDict[str, Tensor] = OrderedDict()
        self._modules: OrderedDict[str, Module] = OrderedDict()
        self._training = True

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        """Register a parameter.

        Args:
            name: Parameter name.
            param: Parameter tensor.
        """
        if param is None:
            if name in self._parameters:
                del self._parameters[name]
        else:
            self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Tensor | None) -> None:
        """Register a buffer (non-trainable tensor).

        Args:
            name: Buffer name.
            tensor: Buffer tensor.
        """
        if tensor is None:
            if name in self._buffers:
                del self._buffers[name]
        else:
            self._buffers[name] = tensor

    def add_module(self, name: str, module: Module | None) -> None:
        """Add a submodule.

        Args:
            name: Module name.
            module: Module instance.
        """
        if module is None:
            if name in self._modules:
                del self._modules[name]
        else:
            self._modules[name] = module

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Get all parameters.

        Args:
            recurse: Whether to recurse into submodules.

        Yields:
            Parameter tensors.
        """
        for param in self._parameters.values():
            yield param
        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=True)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        """Get all parameters with names.

        Args:
            prefix: Name prefix.
            recurse: Whether to recurse into submodules.

        Yields:
            (name, parameter) tuples.
        """
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield (full_name, param)
        if recurse:
            for name, module in self._modules.items():
                sub_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_parameters(prefix=sub_prefix, recurse=True)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Get all buffers.

        Args:
            recurse: Whether to recurse into submodules.

        Yields:
            Buffer tensors.
        """
        for buffer in self._buffers.values():
            yield buffer
        if recurse:
            for module in self._modules.values():
                yield from module.buffers(recurse=True)

    def state_dict(self) -> dict:
        """Get state dictionary.

        Returns:
            Dictionary mapping parameter/buffer names to tensors.
        """
        state = {}
        for name, param in self._parameters.items():
            state[name] = param.data.copy()
        for name, buffer in self._buffers.items():
            state[name] = buffer.data.copy()
        for name, module in self._modules.items():
            state.update({f"{name}.{k}": v for k, v in module.state_dict().items()})
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dictionary.

        Args:
            state_dict: State dictionary to load.
        """
        for name, param in self._parameters.items():
            if name in state_dict:
                param.data = state_dict[name].copy()
        for name, buffer in self._buffers.items():
            if name in state_dict:
                buffer.data = state_dict[name].copy()
        for name, module in self._modules.items():
            module_state = {
                k[len(name) + 1 :]: v for k, v in state_dict.items() if k.startswith(f"{name}.")
            }
            if module_state:
                module.load_state_dict(module_state)

    def train(self, mode: bool = True) -> Module:
        """Set training mode.

        Args:
            mode: Training mode (True) or eval mode (False).

        Returns:
            Self for chaining.
        """
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Module:
        """Set evaluation mode.

        Returns:
            Self for chaining.
        """
        return self.train(False)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients of all parameters.

        Args:
            set_to_none: If True, set grad to None instead of zeros.
        """
        for param in self.parameters():
            param.zero_grad(set_to_none)

    def __call__(self, *args, **kwargs):
        """Forward pass."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass (to be implemented by subclasses)."""
        raise NotImplementedError
