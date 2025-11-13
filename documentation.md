# JunGrad Codebase Guide

This document walks through every module, class, and key function in the `jungrad` package so you can reason about the system as if you authored it. JunGrad provides an N-dimensional autograd stack on top of NumPy, complete with neural network building blocks, optimizers, schedulers, profiling utilities, and testing helpers.

**Note**: For a hands-on tutorial with real-world examples, see `quickstart.ipynb`, which includes an end-to-end classification demo using the 20 Newsgroups dataset.

---

## Package Top-Level

- **`jungrad/__init__.py`** exposes the public surface area. It re-exports tensor constructors, gradient-mode helpers, functional utilities, neural-network modules, loss functions, optimizers, schedulers, testing helpers, and toolkits such as `graphviz` and `profiler`. Importing `jungrad` gives you batteries included (`Tensor`, `Module`, `SGD`, schedulers, etc.).
- **`jungrad/_version.py`** pins `__version__ = "0.1.0"`. `__init__` also mirrors that value for users.

---

## Core Types & Support Utilities

- **`jungrad/types.py`** defines shared type aliases (`NDArray`, `Shape`, `DType`) and the graph primitives:
  - `Edge`: immutable edge wrapper holding a parent tensor reference and an optional `grad_fn` used during backward propagation.
  - `Node`: container for a tensor, the op that produced it, and parent edges (useful for visualization/debugging).
- **`jungrad/exceptions.py`** houses custom exception types (`JungradError`, `AutogradError`, `ShapeError`, `NumericsError`). These help distinguish logic bugs, shape mismatches, and numerical instabilities.
- **`jungrad/logging.py`** offers a `get_logger` helper that lazily configures a stream handler with a standard formatter. Other modules import this through `jungrad.utils.get_logger` to avoid circular imports.
- **`jungrad/utils.py`** provides convenience helpers:
  - `seed_everything` seeds Python and NumPy RNGs.
  - `asarray` normalizes inputs to NumPy arrays.
  - Broadcasting utilities (`check_shape_compatible`, `broadcast_shape`, `reduce_broadcasted_grad`) manage shape inference and the reverse-broadcast logic required during backward passes.
  - Numerical guards (`check_finite`, `enable_nan_check`) allow opt-in NaN detection.
  - Dtype converters (`to_fp16`, `to_fp32`) and `validate_shape` round out low-level sanitation.

These utilities underpin the tensor core and ops library, ensuring consistent shape and error handling.

---

## Autograd Engine

- **`jungrad/tensor.py`** defines the `Tensor` class, the heart of JunGrad:
  - Stores `data` (NumPy array), optional `grad`, operation metadata (`op`, `parents` of type `Edge`), and debugging fields (`name`, `_retain_grad`).
  - Controls gradient tracking via `requires_grad` and the global flag from `autograd.is_grad_enabled()`.
  - Implements basic tensor APIs (`shape`, `dtype`, `item`, `numpy`, `astype`, `to`, `detach`, `retain_grad`, `zero_grad`).
  - Overloads arithmetic operators (`__add__`, `__sub__`, `__mul__`, `__truediv__`, `__neg__`, `__pow__`, `__matmul__`) by delegating to functions in `jungrad.ops` to keep forward/backward logic centralized.
  - Supplies reduction helper `sum` and creation factories (`tensor`, `zeros`, `ones`, `randn`, `arange`, `full`). These wrap NumPy constructors while respecting gradient mode.

- **`jungrad/autograd.py`** implements global gradient control and the reverse-mode engine:
  - `set_grad_enabled`, `is_grad_enabled`, `no_grad`, `enable_grad` mirror PyTorch-style gradient mode management.
  - `toposort(outputs)` walks the computation graph (edges stored on tensors) to produce a deterministic order for the backward pass.
  - `backward(output, grad=None)` seeds gradients (defaulting to `ones_like`), accumulates into `output.grad`, and iterates through the reversed topological order. For each parent edge, it invokes the stored `grad_fn` to propagate gradients, allocating and accumulating into parent gradients as needed.

Together, `Tensor` and `autograd.backward` orchestrate computation graph construction, gradient storage, and backpropagation.

---

## Primitive Operations (`jungrad/ops.py`)

This file contains the differentiable building blocks. Each function performs a NumPy forward computation, instantiates an output `Tensor`, and, when gradients are enabled, wires `Edge` objects with closure-based `grad_fn`s that describe how to backpropagate. Major categories:

- **Elementwise arithmetic**: `add`, `sub`, `mul`, `div`, `neg`, `pow` support broadcasting and call `reduce_broadcasted_grad` to reverse broadcasted shapes.
- **Unary transforms**: `exp`, `log`, activations via composite operations elsewhere.
- **Reductions**: `sum`, `mean`, `max`, `min`, `var`, `std` implement gradient broadcasting over reduced axes, handling keepdims semantics and unbiased variance.
- **Linear algebra**: `matmul`, `bmm` compute matrix multiplications with the standard Jacobians (`grad @ Bᵀ`, `Aᵀ @ grad`).
- **Shape ops**: `transpose`, `permute`, `reshape`, `flatten`, `squeeze`, `unsqueeze`, `broadcast_to` manipulate layout while providing inverses for gradients.
- **Tensor assembly**: `concat`, `stack` split upstream gradients along the concatenation dimension or extract slices for stacked tensors.
- **Indexing**: `gather`, `scatter_add`, `slice`, `take` implement advanced indexing semantics with corresponding gradient scatter/gather behavior.

These primitives are the only place where backward formulas live, keeping autograd mechanics declarative and reusable across higher-level APIs.

---

## Functional Helpers (`jungrad/functional.py`)

Wrapper functions layer numerical stability and ergonomic APIs atop the primitives:

- `logsumexp` performs the max-subtraction trick for stability: `log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))`. Supports axis reduction and keepdims.
- `log_softmax`, `softmax` build on `logsumexp` for numerically stable probability distributions.
- Activation families: `relu`, `tanh`, `sigmoid`, `gelu` combine primitive ops. `relu` uses `maximum` with zero; `tanh` and `sigmoid` use exp-based formulas; `gelu` uses the approximate form with tanh.

All functions return `Tensor` objects that integrate seamlessly with the autograd system. The module imports `max` from `ops` (not Python's built-in) to support axis-aware reductions.

---

## Loss Functions (`jungrad/losses.py`)

- `cross_entropy` accepts logits with either class indices or one-hot targets and optional label smoothing. It routes through `log_softmax`, supports manual gradient wiring for the indexed path, and reduces via `mean`.
- `mse_loss` wraps `(input - target)^2` followed by `mean`.
- `bce_with_logits` implements the numerically stable sigmoid-cross-entropy identity `max(x,0) - x*y + log(1 + exp(-|x|))` with a hand-coded gradient equivalent to `sigmoid(x) - target`.

These losses illustrate how to drop into NumPy for tricky parts yet still register proper `grad_fn`s so the rest of the graph differentiates automatically.

---

## Neural Network Building Blocks (`jungrad/nn/`)

### Core abstractions (`module.py`)
- `Parameter` is a `Tensor` subclass that always requires gradients.
- `Module` mirrors PyTorch’s module system: it manages ordered dictionaries of parameters, buffers, and child modules; supplies traversal (`parameters`, `named_parameters`, `buffers`), state management (`state_dict`, `load_state_dict`), mode control (`train`, `eval`), and gradient resetting (`zero_grad`). Forward passes are defined by overriding `.forward` or `__call__`.

### Layers (`layers.py`)
- `Linear` holds weight and optional bias parameters, performs `x @ Wᵀ + b` using `matmul` and `transpose`.
- `Sequential` and `Stack` compose submodules sequentially or in parallel (with stacking along a new dimension).
- `ReLU` delegates to `functional.relu` while presenting a module interface.
- `Dropout` applies inverted dropout during training by sampling a NumPy mask and multiplying through `ops.mul`.
- `LayerNorm` computes mean/variance across trailing dimensions, normalizes input, and applies optional affine weights/biases.
- `Embedding` performs table lookups, flattens indices, gathers rows from the weight matrix, reshapes, and scatters gradients back to weights.
- `Conv1d` implements a basic 1D convolution with manual padding, loop-based accumulation, and custom backward logic for both input and weights (bias is added via broadcasting).

### Initialization (`init.py`)
- `xavier_uniform_` (Glorot uniform): Samples from uniform distribution based on fan-in and fan-out.
- `kaiming_normal_` (He normal): Samples from normal distribution optimized for ReLU activations. Supports `fan_in` or `fan_out` modes.
- `orthogonal_`: Generates orthogonal matrices using SVD, useful for RNNs and transformers.
- `constant_`: Fills parameters with a constant value.

All functions mutate `Parameter.data` in-place via NumPy operations. They compute fan-in/out statistics automatically and handle reshaping necessary for orthogonal matrices.

### Utilities (`utils.py`)
- `count_params`, `freeze`, `unfreeze`, `named_parameters`, `summary` provide introspection and control over trainable parameters. `summary` gives a lightweight textual overview (with placeholder output shape unless a forward pass is attempted).

---

## Optimization Suite (`jungrad/optim/`)

- **`SGD` (`optim/sgd.py`)**: supports learning rate, momentum, optional Nesterov updates, and L2 weight decay. Maintains per-parameter velocity buffers when momentum is enabled.
- **`Adam` & `AdamW` (`optim/adam.py`)**: track biased first and second moments, apply bias correction, and update parameters; `AdamW` decouples weight decay from the moment update.
- **`RMSProp` (`optim/rmsprop.py`)**: keeps an exponential moving average of squared gradients, optional momentum, and weight decay.
- **Gradient clipping (`optim/clip.py`)**: `clip_grad_norm_` scales gradients to a maximum norm, `clip_grad_value_` clamps values elementwise.

Optimizers expect iterators of `Parameter` objects (e.g., `model.parameters()`) and mutate `param.data` in-place.

---

## Learning Rate Schedulers (`jungrad/sched/`)

- **`StepLR`** (`sched/step.py`): Decays the optimizer's `lr` by `gamma` every `step_size` epochs. Also supports `ExponentialLR` which multiplies by `gamma` each step.
- **`CosineLR`** (`sched/cosine.py`): Implements cosine annealing: `eta_min + (eta_max - eta_min) * (1 + cos(π * T_cur / T_max)) / 2`. Supports warmup phase and minimum learning rate.
- **`OneCycleLR`** (`sched/onecycle.py`): Follows the one-cycle policy with three phases:
  1. Linear warmup from `initial_lr` to `max_lr` over `pct_start * total_steps`
  2. Cosine or linear annealing from `max_lr` to `final_lr` over remaining steps
  3. Optional final decay phase

All schedulers operate by mutating the optimizer's `lr` attribute. Calling `.step()` advances the schedule, and `.get_lr()` exposes the current value. Schedulers track `last_epoch` internally for state management.

---

## Hooks, Profiling, Visualization, Testing

- **`jungrad/hooks.py`**
  - `TensorHook` and `register_tensor_hook` attach callable hooks to tensors for intercepting backward gradients (allowing modification or inspection).
  - Module-level hooks (`register_forward_hook`, `register_backward_hook`) mirror PyTorch semantics, storing callables and returning removal handles. Hooks are stored on modules as `_forward_hooks` and `_backward_hooks` lists.
- **`jungrad/profiler.py`**
  - `Profiler` collects timing statistics per named scope using a context manager (`record`). It aggregates counts and total/average times, and exposes `reset` and `summary` for reporting.
  - `profile(name)` is a convenience context manager around a global `Profiler` instance (`get_profiler`).
- **`jungrad/graphviz.py`**
  - `to_dot(output, max_nodes=100)` topologically sorts the computation graph and emits a DOT description with tensor metadata (shape, grad norm, op name).
  - `export_graph(output, filename)` writes that DOT string to disk.
- **`jungrad/testing/gradcheck.py`**
  - `gradcheck` performs finite-difference gradient verification. It clones inputs to float64, perturbs each scalar entry by `±eps`, compares analytical gradients from `backward()` to numerical estimates, and either raises or warns depending on `raise_exception`.

These utilities round out the developer experience by enabling debugging, performance profiling, visualization, and regression tests for new ops.

---

## Package Entrypoint Refresher

- Importing `jungrad` yields aliases for core components (`Tensor`, tensor factories, gradient-mode contexts), the neural-network toolkit (`Module`, layers, initializers), losses, optimizers, schedulers, testing utilities (`gradcheck`), and helper namespaces (`functional` as `jungrad.F`).
- The package-level `__all__` ensures clean tab-completion and `from jungrad import *` behavior when desired.

---

## Mental Model Recap

1. **Computation graph construction** happens implicitly: any call into `jungrad.ops` or composite functions returns a `Tensor` carrying parent `Edge` objects with closures describing local Jacobians.
2. **Backward propagation** runs via `autograd.backward`, visiting nodes in reverse topological order and invoking each edge’s `grad_fn` to accumulate gradients using NumPy operations.
3. **High-level abstractions** (functional ops, losses, modules, optimizers, schedulers) are thin layers that orchestrate the primitives, never duplicating gradient logic.
4. **Developer tooling** (hooks, profiler, graphviz, gradcheck) equips you to debug and validate new components quickly.

With this structure in mind, extending JunGrad—by authoring new ops, layers, or optimizers—boils down to implementing a NumPy forward pass and supplying the matching gradient in `ops.py` or a higher-level module. Everything else (Tensor bookkeeping, engine orchestration, utilities) is already in place.

## Real-World Usage Example

The `quickstart.ipynb` notebook demonstrates a complete end-to-end workflow:

1. **Data Loading**: Uses Hugging Face `datasets` library to load the 20 Newsgroups text classification dataset
2. **Feature Extraction**: Applies TF-IDF vectorization via scikit-learn to convert text to dense feature vectors
3. **Model Architecture**: Multi-layer feedforward network with dropout regularization
4. **Training Setup**: Adam optimizer with cosine annealing scheduler and gradient clipping
5. **Training Loop**: Full epoch-based training with train/validation splits, metrics tracking, and best model selection
6. **Evaluation**: Per-class accuracy metrics and comprehensive validation reporting
7. **Visualization**: Matplotlib plots showing training/validation loss and accuracy curves

This example showcases how all the components work together in a realistic machine learning pipeline, demonstrating that JunGrad is not just an educational tool but can handle real-world tasks.


