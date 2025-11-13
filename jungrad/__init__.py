"""JunGrad - A robust N-D autograd library.

Package name: jungrad
Project name: JunGrad
"""

from jungrad.autograd import enable_grad, is_grad_enabled, no_grad, set_grad_enabled
from jungrad.losses import bce_with_logits, cross_entropy, mse_loss
from jungrad.nn import (
    Conv1d,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    Parameter,
    ReLU,
    Sequential,
    Stack,
    init,
)
from jungrad.optim import SGD, Adam, AdamW, RMSProp, clip_grad_norm_
from jungrad.sched import CosineLR, ExponentialLR, OneCycleLR, StepLR
from jungrad.tensor import Tensor, arange, full, ones, randn, tensor, zeros

# Functional API
from jungrad import functional as F

# Testing
from jungrad.testing import gradcheck

# Graphviz
from jungrad import graphviz

# Profiler
from jungrad import profiler

__version__ = "0.1.0"

__all__ = [
    # Core
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "arange",
    "full",
    # Autograd
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    # Functional
    "F",
    # NN
    "Module",
    "Parameter",
    "Linear",
    "Embedding",
    "Conv1d",
    "LayerNorm",
    "Dropout",
    "ReLU",
    "Sequential",
    "Stack",
    "init",
    # Losses
    "cross_entropy",
    "mse_loss",
    "bce_with_logits",
    # Optimizers
    "SGD",
    "Adam",
    "AdamW",
    "RMSProp",
    "clip_grad_norm_",
    # Schedulers
    "StepLR",
    "ExponentialLR",
    "CosineLR",
    "OneCycleLR",
    # Testing
    "gradcheck",
    # Tools
    "graphviz",
    "profiler",
]
