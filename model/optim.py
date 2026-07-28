"""ObGD — Overshooting-bounded Gradient Descent (Elsayed et al. 2024).

From "Streaming Deep Reinforcement Learning Finally Works"
(arXiv:2410.14606), the optimizer at the core of the stream-x algorithms.
Ported from the official implementation (github.com/mohmdelsayed/streaming-drl,
optim.py) with one change: the step-size bound is computed **per parameter
group**, so actor / critic / CNN groups get their own κ and normalization —
equivalent to the paper's use of separate ObGD instances per network.

Why this matters for SEAL (sample efficiency): streaming RL fails through
instability, not information scarcity (the "stream barrier"). Hard clipping
or tiny fixed learning rates waste most of each sample. ObGD instead bounds
the *effective* step size per update:

    e      <- γλ·e + ∇L                    (per-parameter eligibility trace)
    δ̄      = max(|δ|, 1)
    M      = δ̄ · ||e||₁ · lr · κ           (projected effective step size)
    step   = lr / M    if M > 1  else lr   (shrink only when overshooting)
    p      <- p − step · δ · e

This allows lr = 1.0 with zero divergence: any update that would overshoot
is auto-shrunk, so EVERY sample takes the largest stable step.

Loss convention: losses must be **δ-free** — ObGD multiplies by δ itself.
For actor-critic with TD error δ (stream AC):
    actor loss  = −log π(a)            (∇ = −∇log π; update += step·δ·∇log π)
    critic loss = −c_V · V             (update += step·c_V·δ·∇V, semi-gradient TD)
    entropy     = −c_ent · H · sign(δ) (exploration weighted by |δ|)
    CNN (e-prop): −(L_in · p)          (update += step·δ·∇(L_in·p), Eq. 36 sign)
"""
from __future__ import annotations
import torch


class ObGD(torch.optim.Optimizer):
    """Overshooting-bounded GD with per-group eligibility traces.

    Args:
        params: param groups; each may set its own "lr", "kappa", "weight_decay".
        lr: default step size α (1.0 in the paper — the bound handles safety).
            NOTE: in the normalized regime (M > 1) lr cancels; the effective
            step magnitude is then governed by κ. lr matters only when traces
            are small (e.g. right after episode resets).
        gamma: discount γ (trace decay = γλ)
        lamda: trace λ (trace decay = γλ; 0.8 in the paper)
        kappa: default scaling factor κ (2.0 value / 3.0 policy in the paper)
        weight_decay: per-group L2 pull (0 = off). The normalized update has
            near-constant size, so weights perform a random walk with drift;
            a small decay bounds ||W|| (keeps softmax entropy and V scale in
            range). Not in the stream-x paper; needed here because gradients
            from a linear readout on sparse terminal rewards are far more
            correlated than their LN'd deep-conv features.
    """
    def __init__(self, params, lr: float = 1.0, gamma: float = 0.99,
                 lamda: float = 0.8, kappa: float = 2.0, weight_decay: float = 0.0):
        # delta_cap (0 = off, stream-x default): when > 0, clamp the
        # overshooting-normalizer d_bar = max(|delta|,1) to delta_cap for THIS
        # group only. Below the cap: standard stream-x (constant-magnitude,
        # safe step). Above the cap: the update grows linearly with |delta|,
        # restoring e-prop's error-proportional kicks. Needed for the critic
        # group because SEAL's value readout is LEAKY (e-prop Eq. 11, kappa=0.95
        # -> ~20x gain, ~20-frame memory): it accumulates a DC bias that small
        # constant-magnitude steps cannot flush. The terminal reward is the
        # only signal large enough to clear it; delta_cap stops ObGD from
        # normalizing that signal away. Actor/CNN leave delta_cap=0 (their
        # heads are not leaky, so stream-x's bound is correct for them).
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa,
                        weight_decay=weight_decay, delta_cap=0.0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, delta: float, reset: bool = False):
        """One ObGD update.

        Args:
            delta: scalar TD error δ_t (gates AND scales the update)
            reset: zero the eligibility traces (episode boundary)
        """
        # ---- 1) update traces, per-group ||e||_1 ----
        z_sums = []
        for group in self.param_groups:
            z_sum = 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "eligibility_trace" not in state:
                    state["eligibility_trace"] = torch.zeros_like(p)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad)
                z_sum += e.abs().sum().item()
            z_sums.append(z_sum)

        # ---- 2) per-group overshooting-bounded step ----
        # d_bar is per-group: capped for groups with delta_cap>0 (critic),
        # uncapped (pure stream-x) otherwise.
        a_delta = abs(delta)
        for group, z_sum in zip(self.param_groups, z_sums):
            cap = group.get("delta_cap", 0.0)
            if cap > 0.0 and a_delta > cap:
                d_bar = cap                       # above cap: update ~ |delta|/cap
            else:
                d_bar = max(a_delta, 1.0)          # standard stream-x
            M = d_bar * z_sum * group["lr"] * group["kappa"]
            step_size = group["lr"] / M if M > 1.0 else group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = self.state[p]["eligibility_trace"]
                wd = group["weight_decay"]
                if wd > 0:
                    p.mul_(1.0 - step_size * wd)
                p.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()
