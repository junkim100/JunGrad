# JunGrad

A robust N-D autograd library with comprehensive operations, neural network layers, and optimizers.

## Features

- **N-D Tensor** with numpy backend and automatic differentiation
- **Comprehensive Operations**:
  - Elementwise ops (add, sub, mul, div, exp, log, pow, etc.)
  - Reductions (sum, mean, max, min, var, std)
  - Linear algebra (matmul, transpose, reshape, concat, stack)
  - Indexing (gather, scatter_add, slice, take)
- **Neural Network Modules**:
  - `Linear`, `Conv1d`, `Embedding`, `LayerNorm`, `Dropout`
  - `Sequential`, `Stack` containers
  - Parameter and state_dict management
- **Optimizers**:
  - `SGD` (with momentum and Nesterov)
  - `Adam`, `AdamW` (decoupled weight decay)
  - `RMSProp`
  - Gradient clipping utilities
- **Learning Rate Schedulers**:
  - `StepLR`, `ExponentialLR`
  - `CosineLR` (with warmup)
  - `OneCycleLR`
- **Loss Functions**:
  - `cross_entropy` (with label smoothing support)
  - `mse_loss`
  - `bce_with_logits` (numerically stable)
- **Functional API**:
  - Activations: `relu`, `tanh`, `sigmoid`, `gelu`
  - Stable: `softmax`, `log_softmax`, `logsumexp`
- **Utilities**:
  - Gradient checking via finite differences
  - Profiler for timing operations
  - Graphviz export for computation graphs
  - Hooks system

## Installation

```bash
# From PyPI (once published)
pip install jungrad

# Optional extras
pip install "JunGrad[viz]"        # Graphviz rendering support
pip install "JunGrad[tutorial]"   # Dependencies for the quickstart notebook
pip install "JunGrad[sparse]"     # SciPy-backed sparse utilities

# Development installation (editable)
pip install -e .

# Development with extras
pip install -e ".[dev,sparse,viz,tutorial]"
```

## Quick Start

### Basic Tensor Operations

```python
from jungrad import tensor, randn

# Create tensors
a = tensor([1.0, 2.0, 3.0], requires_grad=True)
b = tensor([4.0, 5.0, 6.0], requires_grad=True)

# Operations
c = a + b
d = a * b
e = a ** 2

# Backward pass
e.backward()
print(f"a.grad = {a.grad}")
print(f"b.grad = {b.grad}")
```

### Neural Network

```python
from jungrad import randn, no_grad
from jungrad.nn import Linear, Sequential, ReLU, Dropout
from jungrad.optim import Adam
from jungrad.losses import cross_entropy
from jungrad.sched import CosineLR
from jungrad.optim import clip_grad_norm_

# Create model with dropout
model = Sequential(
    Linear(10, 64),
    ReLU(),
    Dropout(0.3),
    Linear(64, 32),
    ReLU(),
    Dropout(0.3),
    Linear(32, 5),  # 5 classes
)

# Training data
x = randn(32, 10)
y = [0, 1, 2, 3, 4] * 6 + [0, 1]  # class labels

# Forward pass
logits = model(x)
loss = cross_entropy(logits, y)

# Backward and optimize
optimizer = Adam(model.parameters(), lr=0.001)
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()

# Evaluation mode
model.eval()
with no_grad():
    pred = model(x)
```

### Complete Training Example

```python
from jungrad import randn, tensor, no_grad
from jungrad.nn import Linear, Sequential, ReLU, Dropout
from jungrad.nn.init import kaiming_normal_
from jungrad.losses import cross_entropy
from jungrad.optim import Adam, clip_grad_norm_
from jungrad.sched import CosineLR
import numpy as np

# Model
model = Sequential(
    Linear(10, 64), ReLU(), Dropout(0.3),
    Linear(64, 32), ReLU(), Dropout(0.3),
    Linear(32, 5),
)

# Initialize weights
for module in model.modules():
    if isinstance(module, Linear):
        kaiming_normal_(module.weight)

# Setup training
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = CosineLR(optimizer, T_max=50)

# Training loop
for epoch in range(50):
    x = randn(32, 10)
    y = np.random.randint(0, 5, size=32)

    model.train()
    logits = model(x)
    loss = cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # Validation
    if epoch % 10 == 0:
        model.eval()
        with no_grad():
            val_logits = model(randn(16, 10))
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Gradient Checking

```python
from jungrad import randn
from jungrad.ops import matmul
from jungrad.testing import gradcheck

# Verify gradients are correct
a = randn(3, 4, requires_grad=True)
b = randn(4, 5, requires_grad=True)

passed = gradcheck(lambda x, y: matmul(x, y), (a, b), eps=1e-5)
print(f"Gradients verified: {passed}")
```

## Tutorial

See `quickstart.ipynb` for a comprehensive interactive tutorial covering:
- Creating and manipulating N-D tensors
- Automatic differentiation and gradient computation
- Building neural networks with Sequential layers
- Training models with optimizers (SGD, Adam, AdamW, RMSProp)
- Advanced layers (Conv1d, LayerNorm, Embedding, Dropout)
- Activation functions and functional API
- Learning rate schedulers (StepLR, CosineLR, OneCycleLR)
- Gradient clipping for stable training
- Performance optimization with `no_grad()`
- Computation graph visualization with Graphviz
- Performance profiling
- Gradient checking for verification
- **End-to-end classification demo** with real-world data (20 Newsgroups dataset)

## Documentation

### Tensor API

```python
from jungrad import Tensor, tensor, zeros, ones, randn

# Creation
x = tensor([1.0, 2.0, 3.0], requires_grad=True)
x = zeros((3, 4))
x = ones((2, 2))
x = randn(5, 5)

# Methods
x.detach()           # Detach from graph
x.retain_grad()      # Retain gradient for non-leaf
x.item()             # Get scalar value
x.numpy()            # Get numpy array
x.backward()         # Compute gradients
x.zero_grad()        # Zero gradients
```

### Autograd

```python
from jungrad import no_grad, enable_grad

# Context managers
with no_grad():
    # Operations without gradients
    y = x * 2

with enable_grad():
    # Enable gradients
    z = x + y
```

### Operations

```python
from jungrad.ops import add, mul, matmul, sum, mean

# Elementwise
c = add(a, b)
c = mul(a, b)

# Matrix multiplication
out = matmul(x, w)

# Reductions
s = sum(x, axis=0, keepdims=True)
m = mean(x, axis=-1)
```

### Neural Network Layers

```python
from jungrad.nn import Linear, Conv1d, Embedding, LayerNorm, Dropout

# Linear layer
linear = Linear(in_features=10, out_features=5, bias=True)

# Conv1d
conv = Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# Embedding
embed = Embedding(num_embeddings=1000, embedding_dim=128)

# LayerNorm
ln = LayerNorm(normalized_shape=128, eps=1e-5, affine=True)

# Dropout
dropout = Dropout(p=0.5)
```

### Optimizers

```python
from jungrad.optim import SGD, Adam, AdamW, RMSProp, clip_grad_norm_

# SGD
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

# Adam
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (decoupled weight decay)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Gradient clipping
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Loss Functions

```python
from jungrad.losses import cross_entropy, mse_loss, bce_with_logits

# MSE loss
loss = mse_loss(pred, target)

# Cross-entropy (supports index and one-hot targets)
loss = cross_entropy(logits, target_indices)
loss = cross_entropy(logits, target_onehot, label_smoothing=0.1)

# BCE with logits
loss = bce_with_logits(logits, target)
```

## Development

### Running Tests

**Run all tests:**
```bash
pytest tests/
# or
python -m pytest tests/
```

**Run with verbose output:**
```bash
pytest tests/ -v
```

**Run a specific test file:**
```bash
pytest tests/test_tensor.py
```

**Run a specific test:**
```bash
pytest tests/test_autograd.py::test_backward_simple
```

**Run tests matching a pattern:**
```bash
pytest tests/ -k "backward"
```

**With coverage report:**
```bash
pytest tests/ --cov=jungrad --cov-report=term-missing
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

### Code Quality

```bash
# Format
black jungrad/

# Lint
ruff check jungrad/

# Type check
mypy jungrad/
```

## Architecture

- **tensor.py**: N-D Tensor class with autograd support
- **autograd.py**: Backward engine with topological sort
- **ops.py**: Low-level primitive operations (200+ operations)
- **functional.py**: High-level differentiable functions (activations, softmax)
- **losses.py**: Loss functions (MSE, cross-entropy, BCE)
- **nn/**: Neural network modules (Module, Parameter, layers, initialization, utils)
- **optim/**: Optimizers (SGD, Adam, AdamW, RMSProp) and gradient clipping
- **sched/**: Learning rate schedulers (StepLR, CosineLR, OneCycleLR, ExponentialLR)
- **testing/**: Gradient checking utilities
- **graphviz.py**: Computation graph visualization
- **profiler.py**: Performance profiling tools
- **hooks.py**: Forward and backward hooks for debugging

## Design Philosophy

- **N-D from the start**: Not scalar-only like micrograd - works with tensors of any shape
- **Separate engine**: Clean separation between tensor and autograd engine
- **Broadcast-aware**: Proper handling of broadcasting in forward and backward passes
- **Numerically stable**: Uses logsumexp for softmax, stable sigmoid, stable BCE with logits
- **Type-safe**: Full type hints throughout the codebase
- **PyTorch-inspired API**: Familiar interface for users coming from PyTorch
- **Educational**: Clear, well-documented code suitable for learning autograd internals

## Real-World Example

The `quickstart.ipynb` tutorial includes a complete end-to-end classification demo using the [20 Newsgroups dataset](https://huggingface.co/datasets/SetFit/20_newsgroups), demonstrating:
- Loading real-world text data
- TF-IDF feature extraction
- Multi-layer neural network with dropout
- Training with learning rate scheduling
- Gradient clipping for stability
- Comprehensive evaluation metrics
- Training curve visualization

This showcases how JunGrad can be used for practical machine learning tasks beyond synthetic examples.

## Changelog

### Version 0.2.0

**Major Features:**
- **GPU Support via CuPy**: Full GPU acceleration support using CuPy arrays
  - Automatic device detection and conversion between NumPy and CuPy
  - Device-aware operations throughout the library
  - Seamless fallback to CPU when CuPy is not available
  - New `backend` module with utilities: `has_cupy()`, `get_array_module()`, `to_device_array()`, `to_numpy_array()`

**Improvements:**
- All operations (`add`, `sub`, `mul`, `div`, `matmul`, `maximum`, etc.) now handle mixed NumPy/CuPy arrays
- Autograd engine is device-aware and maintains device consistency during backpropagation
- Embedding layer backward pass updated for GPU compatibility
- `asarray` utility preserves CuPy arrays instead of converting to NumPy

**Technical Details:**
- Operations automatically detect array module (NumPy or CuPy) from input tensors
- Device conversion handled transparently during tensor operations
- Gradient accumulation maintains device consistency across the computation graph

### Version 0.1.0 (Initial Release)

- Core N-D tensor implementation with autograd
- Comprehensive operations library (200+ operations)
- Neural network modules (Linear, Conv1d, Embedding, LayerNorm, Dropout)
- Optimizers (SGD, Adam, AdamW, RMSProp)
- Learning rate schedulers (StepLR, CosineLR, OneCycleLR, ExponentialLR)
- Loss functions (MSE, cross-entropy, BCE with logits)
- Functional API with activations
- Gradient checking utilities
- Computation graph visualization
- Performance profiling tools

## License

MIT License
