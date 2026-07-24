"""AdaptiveObGD — the paper's actual optimizer (verbatim from stream_q_atari.py).

Source: Elsayed, Vasan & Mahmood, "Streaming Deep Reinforcement Learning
Finally Works", arXiv:2410.14606. Official implementation:
github.com/mohmdelsayed/streaming-drl, optim.py.

AdaptiveObGD = ObGD (overshooting-bounded) + Adam-style per-parameter second-
moment normalization of the eligibility trace. This is the paper's fix for
trace explosion on long runs: instead of a hard global ‖z‖_1 clip (which
freezes the step size once hit — the ~756k-frame coma), each parameter's trace
is divided by the running RMS of (delta * trace) for that parameter. The
normalized trace has unit scale per parameter, so z_sum stays O(n_params)
regardless of how large the raw traces grow. No ceiling to hit, no coma.

=== The math (paper-faithful) ===
  e[p]    = (λ·γ)·e[p] + grad[p]                         (eligibility trace)
  v[p]    = β2·v[p] + (1-β2)·(δ·e[p])²                   (2nd moment, Adam-style)
  v_hat   = v / (1 - β2^t)                                (bias correction)
  z_sum   = Σ |e[p] / sqrt(v_hat[p] + ε)|                (normalized denominator)
  δ̄      = max(|δ|, 1)
  α_eff   = α / (κ·δ̄·z_sum)  if that product > 1 else α  (ObGD bound)
  W[p]   -= α_eff · δ · e[p] / sqrt(v_hat[p] + ε)        (per-param normalized step)

The α cancels in the bound-active regime (verified): the effective step is
governed by κ, δ̄, and the normalized z_sum — NOT by nominal α. λ is the only
true dial (controls the raw trace's temporal horizon); β2/ε are Adam-standard.
"""
from __future__ import annotations
import torch


class AdaptiveObGD:
    """Adaptive Overshooting-bounded Gradient Descent (paper, verbatim).

    Holds one eligibility trace per param + one second-moment accumulator per
    param (Adam-style v). The trace is normalized by sqrt(v_hat) in both the
    denominator and the update, keeping z_sum bounded by ~n_params.
    """
    def __init__(self, params, alpha: float = 1.0, kappa: float = 2.0,
                 lam: float = 0.8, gamma: float = 0.99,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.params = list(params)
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        # eligibility traces (the streaming-RL convention: held in the optimizer)
        self.traces = [torch.zeros_like(p.detach()) for p in self.params]
        # per-parameter second moment of (delta * trace), Adam-style
        self._v = [torch.zeros_like(p.detach()) for p in self.params]
        self.counter = 0
        # diagnostics
        self.last_z_sum = 0.0
        self.last_step_size = 0.0

    def _accumulate_traces(self, grads):
        with torch.no_grad():
            decay = self.lam * self.gamma
            for i, g in enumerate(grads):
                if g is None:
                    continue
                self.traces[i].mul_(decay).add_(g.detach())

    @property
    def total_traces(self):
        """Per-param trace (for the utility tracker)."""
        return self.traces

    def step(self, td_error: float, grads, reset_traces: bool = False,
             update_mask=None):
        """One streaming update (paper AdaptiveObGD, verbatim structure).

        td_error: scalar TD error δ_t.
        grads: list aligned with params (from autograd.grad).
        reset_traces: zero traces after the update (on episode done / exploration).
        update_mask: list[bool] per param; if False, that param is NOT updated
                     this step (utility gate). Traces + v still accumulate.
        """
        self._accumulate_traces(grads)
        delta = float(td_error)
        self.counter += 1
        with torch.no_grad():
            # ---- update second moment v + compute normalized z_sum (denominator) ----
            z_sum = 0.0
            bc = 1.0 - self.beta2 ** self.counter   # bias correction
            for i, p in enumerate(self.params):
                e = self.traces[i]
                self._v[i].mul_(self.beta2).addcmul_(delta * e, delta * e,
                                                     value=1.0 - self.beta2)
                v_hat = self._v[i] / bc
                z_sum += (e / (v_hat + self.eps).sqrt()).abs().sum().item()
            # ---- ObGD overshooting bound on the normalized denominator ----
            delta_bar = max(abs(delta), 1.0)
            dot_product = delta_bar * z_sum * self.alpha * self.kappa
            step_size = self.alpha / dot_product if dot_product > 1.0 else self.alpha
            # ---- per-parameter normalized update ----
            for i, p in enumerate(self.params):
                if update_mask is not None and not update_mask[i]:
                    continue
                e = self.traces[i]
                v_hat = self._v[i] / bc
                # p.data.add_ / addcdiv_ avoid bumping the autograd version
                # counter, so a transition's graph held across this update stays
                # valid (the standard streaming-RL trick, Elsayed et al. 2024).
                p.data.addcdiv_(delta * e, (v_hat + self.eps).sqrt(),
                                value=-step_size)
        self.last_z_sum = z_sum
        self.last_step_size = step_size
        if reset_traces:
            self.reset()

    def reset(self):
        with torch.no_grad():
            for t in self.traces:
                t.zero_()
        # NOTE: do NOT zero v or counter — the second moment is a long-run
        # statistic that persists across episode boundaries (like Adam's v).
        # Only the eligibility trace resets (credit assignment is per-episode).
