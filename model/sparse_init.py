"""Sparse initialization (paper Appendix F: sparsity ratio s = 90%).

Verbatim from the paper's experimental details, every environment:
  "Lastly, we used sparse initialization with a sparsity ratio s of 90%."
  (F.1 electricity, F.2 MinAtar, F.3 Atari -- all 90%.)

This is a core ingredient of the published stream-x recipe
from Stage 2. Adapted from github.com/mohmdelsayed/streaming-drl/sparse_init.py.

Applied to every Conv2d and Linear weight: draw uniform in
[-sqrt(1/fan_in), sqrt(1/fan_in)], then zero `sparsity` fraction of the
incoming weights per output unit (preserving at least some inputs to each
output). Biases zero.
"""
from __future__ import annotations
import math
import torch


def sparse_init(tensor: torch.Tensor, sparsity: float = 0.9):
    """In-place sparse init for 2D (Linear) or 4D (Conv2d) weight tensors."""
    if tensor.dim() == 2:
        fan_out, fan_in = tensor.shape
    elif tensor.dim() == 4:
        fan_out = tensor.shape[0]
        fan_in = tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
    else:
        raise ValueError(f"sparse_init: unsupported tensor dim {tensor.dim()}")
    num_zeros = int(math.ceil(sparsity * fan_in))
    with torch.no_grad():
        bound = math.sqrt(1.0 / fan_in)
        tensor.uniform_(-bound, bound)
        if tensor.dim() == 2:
            for col in range(fan_out):
                idx = torch.randperm(fan_in)[:num_zeros]
                tensor[col, idx] = 0.0
        else:  # 4D
            for oc in range(fan_out):
                flat = tensor[oc].reshape(fan_in)
                idx = torch.randperm(fan_in)[:num_zeros]
                flat[idx] = 0.0


def apply_sparse_init(module: torch.nn.Module, sparsity: float = 0.9):
    """Apply sparse init to all Conv2d/Linear weights; zero biases."""
    for m in module.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d,
                          torch.nn.modules.conv.Conv2d)):
            sparse_init(m.weight, sparsity=sparsity)
            if m.bias is not None:
                m.bias.data.zero_()
        # EventConv2d/EventLinear hold nn.Parameter weight/bias directly;
        # catch them by attribute name.
        if hasattr(m, "weight") and isinstance(getattr(m, "weight"),
                                               torch.nn.Parameter):
            w = m.weight
            if w.dim() in (2, 4):
                sparse_init(w, sparsity=sparsity)
                if hasattr(m, "bias") and m.bias is not None and \
                        isinstance(m.bias, torch.nn.Parameter):
                    m.bias.data.zero_()
