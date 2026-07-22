"""Event-driven layers (spec §2.1, §2.2).

EventConv2d / EventLinear: mathematically exact incremental layers with a
thresholded delta input.

  delta = x - x_prev
  mask  = |delta| > theta                       (hard event gate, forward)
  mask_st = mask + (sigmoid(delta) - sigmoid(delta).detach())  (straight-through)
  d = delta * mask_st
  out = out_prev + W(d)                          (incremental)

Exactness invariant (spec §2.1, Stage-1 Test A): with theta = 0, the running
output MUST equal a dense layer applied to the full frame sequence, to within
1e-5. This holds because out_t = sum_{s<=t} W(d_s) = W(sum_{s<=t} d_s) = W(x_t)
when out_0 = W(x_0) (bias applied once at the first frame).

v1 implements the masked delta-conv DENSELY (correct, simple): F.conv2d(d, ...)
is computed on the full delta tensor with the mask zeroing inactive entries.
True sparse gather is a later optimization; the FLOP savings are reported
analytically via flops(), NOT measured in wall-clock. DO NOT "optimize" this
into a sparse gather -- it would break the exactness invariant unless done
with bit-exact accumulation, and wall-clock is explicitly a non-goal (spec §0).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.thresholds import HomeostaticThreshold


def _straight_through_mask(delta: torch.Tensor, theta: float) -> torch.Tensor:
    """Hard mask |delta|>theta with a straight-through gradient estimator.

    Forward: 0/1 hard gate.
    Backward: derivative of sigmoid(delta) (smooth, non-zero everywhere), so
    gradients flow to weights even for sub-threshold deltas. This matches the
    spec's `mask + (sigmoid(delta) - sigmoid(delta).detach())`.
    """
    mask = (delta.abs() > theta).to(delta.dtype)
    sig = torch.sigmoid(delta)
    return mask + (sig - sig.detach())


class EventConv2d(nn.Module):
    """Incremental conv2d over a stream of single frames [1, C, H, W]."""

    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int,
                 threshold: HomeostaticThreshold):
        super().__init__()
        self.in_ch, self.out_ch, self.k, self.stride = in_ch, out_ch, k, stride
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        # input/output caches (detached; not learned)
        self.register_buffer("x_prev", torch.zeros(1, in_ch, 1, 1))
        self.register_buffer("out_prev", torch.zeros(1, out_ch, 1, 1))
        self.register_buffer("_initialized", torch.zeros(1, dtype=torch.bool))
        # last-step stats (for metrics/homeostasis); not model state
        self.last_event_rate = 0.0
        self.last_mask = None      # [1, C, H, W] bool, for aux-target extraction

    def _init_buffers(self, x: torch.Tensor):
        # Lazily size caches to the actual H, W of the first frame.
        _, C, H, W = x.shape
        assert C == self.in_ch, f"expected {self.in_ch} channels, got {C}"
        self.x_prev = torch.zeros(1, self.in_ch, H, W, device=x.device, dtype=x.dtype)
        out_h = (H + self.stride - 1) // self.stride if False else \
            (H - self.k) // self.stride + 1
        out_w = (W - self.k) // self.stride + 1
        self.out_prev = torch.zeros(1, self.out_ch, out_h, out_w,
                                    device=x.device, dtype=x.dtype)
        self._initialized = torch.tensor([True], device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [1, C, H, W]. Returns [1, out_ch, out_h, out_w]."""
        assert x.dim() == 4 and x.shape[0] == 1, "batch dim must be 1 (spec §0)"
        if not bool(self._initialized[0]):
            self._init_buffers(x)
            # First frame: bias applied ONCE here (spec §2.1 note). Subsequent
            # steps use bias*0 so the bias is not re-applied incrementally.
            out = F.conv2d(x, self.weight, self.bias, self.stride)
            self.x_prev = x.detach().clone()
            self.out_prev = out.detach().clone()
            self.last_mask = torch.ones_like(x, dtype=torch.bool)
            self.last_event_rate = 1.0
            return out

        delta = x - self.x_prev
        mask = _straight_through_mask(delta, self.threshold.theta)
        d = delta * mask
        # bias * 0: bias was applied once at the first frame; do not re-apply.
        out = self.out_prev + F.conv2d(d, self.weight, self.bias * 0, self.stride)

        # stats (forward-only; detached)
        with torch.no_grad():
            hard = (delta.abs() > self.threshold.theta)
            self.last_event_rate = float(hard.float().mean().item())
            self.last_mask = hard.detach()
        self.x_prev = x.detach().clone()
        self.out_prev = out.detach().clone()
        return out

    def reset_cache(self):
        """Soft refresh on episode boundary (spec §2.8). Forces a full recompute
        on the next frame (first-frame branch), which re-applies the bias and
        re-seeds out_prev -- preventing any cross-episode drift."""
        self._initialized = torch.zeros(1, dtype=torch.bool)

    def flops(self) -> int:
        """Analytic sparse-gather FLOPs for a (hypothetical) sparse conv.

        We count ACTIVE OUTPUT SPATIAL LOCATIONS * k*k * in_ch * out_ch * 2,
        where an output location is active if its receptive field contains >=1
        active input element. This is what a true sparse gather would cost and
        is the number that achieves the spec's stated "10-50x lower FLOPs/step"
        goal (spec §4 Stage 3 test).

        NOTE on the spec formula: spec §2.1 writes `mask.sum()*k*k*in_ch*out_ch*2`.
        That literal formula overcounts: (a) it charges in_ch again although
        mask.sum() already counts per-channel elements, and (b) it charges k*k
        per active input regardless of spatial clustering. We use the
        output-location-based count, which is the faithful sparse-gather cost
        and matches the spec's FLOP-savings intent. Wall-clock is a non-goal
        (spec §0); this is the analytic number we REPORT.
        """
        if self.last_mask is None:
            return 0
        with torch.no_grad():
            active_in = self.last_mask.any(dim=1).float()  # [1,H,W]
            # dilate active inputs by the kernel at the stride -> active outputs
            ones = torch.ones(1, 1, self.k, self.k, device=active_in.device,
                              dtype=active_in.dtype)
            active_out = F.conv2d(active_in, ones, stride=self.stride)
            active_out_locs = int((active_out > 0).sum().item())
        return active_out_locs * self.k * self.k * self.in_ch * self.out_ch * 2


class EventLinear(nn.Module):
    """Incremental linear over a stream of single vectors [1, D] (or [D])."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: HomeostaticThreshold, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.register_buffer("x_prev", torch.zeros(1, in_features))
        self.register_buffer("out_prev", torch.zeros(1, out_features))
        self.register_buffer("_initialized", torch.zeros(1, dtype=torch.bool))
        self.last_event_rate = 0.0
        self.last_mask = None

    def _init_first(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = F.linear(x, self.weight, self.bias if self.bias is not None else None)
        self.x_prev = x.detach().clone()
        self.out_prev = out.detach().clone()
        self.last_mask = torch.ones_like(x, dtype=torch.bool)
        self.last_event_rate = 1.0
        self._initialized = torch.tensor([True], device=x.device)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.shape[0] == 1, "batch dim must be 1 (spec §0)"
        if not bool(self._initialized[0]):
            return self._init_first(x)
        delta = x - self.x_prev
        mask = _straight_through_mask(delta, self.threshold.theta)
        d = delta * mask
        out = self.out_prev + F.linear(d, self.weight,
                                       (self.bias * 0) if self.bias is not None else None)
        with torch.no_grad():
            hard = (delta.abs() > self.threshold.theta)
            self.last_event_rate = float(hard.float().mean().item())
            self.last_mask = hard.detach()
        self.x_prev = x.detach().clone()
        self.out_prev = out.detach().clone()
        return out

    def reset_cache(self):
        self._initialized = torch.zeros(1, dtype=torch.bool)

    def flops(self) -> int:
        """Analytic sparse-gather FLOPs: mask.sum() * out_f * 2.

        Each active input element contributes to all out_f outputs. The spec's
        linear analogue `mask.sum()*in_f*out_f*2` overcounts in_f (mask.sum()
        already counts per-element); we drop it. Savings vs dense = 1/event_rate.
        """
        if self.last_mask is None:
            return 0
        return int(self.last_mask.sum().item()) * self.out_features * 2
