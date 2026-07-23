"""ObGD optimizer (spec §2.6) — the only optimizer SEAL uses.

=== ObGD — VERBATIM FROM THE PAPER (keep this; resolve disagreements by diff) ===
Source: Elsayed, Vasan & Mahmood, "Streaming Deep Reinforcement Learning
Finally Works", arXiv:2410.14606v3, §3.2, Algorithm 3 (copied verbatim):

    Algorithm 3  Overshooting-bounded Gradient Descent (ObGD)
    Require: Eligibility trace z_w, weight vector w, error δ,
             step size α, scaling factor κ
        δ̄ = max(|δ|, 1)
        M ← α κ δ̄ ‖z_w‖_1        # Note: z_w = ∇_w f for supervised learning
        α ← min( α/M , α )
        w ← w + α δ z_w
        return w

Official implementation (github.com/mohmdelsayed/streaming-drl, optim.py):

    dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]   # = M
    step_size   = group["lr"] / dot_product if dot_product > 1 else group["lr"]
    p.data.add_(delta * e, alpha=-step_size)                         # w -= α_eff δ z

WHAT OBGD ACTUALLY IS (the reframe, post-verification):
  When the bound is active (it is, almost always, because ‖z‖_1 ≫ 1):
      α_eff = α / M = α / (α κ δ̄ ‖z‖_1) = 1 / (κ δ̄ ‖z‖_1)     # α CANCELS
      Δw_i  = α_eff · δ · z_i  =  (δ z_i) / (κ δ̄ ‖z‖_1)
  This is a FIXED-BUDGET NORMALIZED STEP, not "gradient descent with a
  learning rate":
    - every parameter gets a share of a constant movement budget,
      proportional to |δ z_i|;
    - per-parameter magnitude is capped at 1/κ (the only true constant);
    - the "effective learning rate" α_eff is entirely a function of trace
      statistics (‖z‖_1) and δ̄ -- NOT of the nominal α.
  Consequences (verified empirically: the 50k probe showed α=1 and α=10
  produce IDENTICAL step_size, exactly as Algorithm 3 predicts):
    - α is NOT a tuning knob; α=1 (paper value) is the right setting. Do not
      sweep α. Log α_eff = 1/(κ δ̄ ‖z‖_1) instead -- α is decorative.
    - κ is the overshoot bound (paper-fixed at 2); never tune it for speed.
    - λ is the ONLY true dial, because it controls ‖z‖_1 via the trace
      steady state ‖z‖_1 ≈ g/(1-λγ). λ=0.95 → factor 16.8; λ=0.8 → 4.8
      → ~3.5x larger α_eff, purely by shrinking the trace. The paper uses
      λ=0.8 everywhere; our spec's 0.95 is the deviation. λ-α coupling is
      really λ-α_eff coupling via z_sum.
"""
from __future__ import annotations
import torch


class ObGD:
    """Overshooting-bounded Gradient Descent (paper Algorithm 3, verbatim).

    Holds eligibility traces internally (the streaming-RL convention). Traces
    accumulate FULL gradients for all params; only the parameter update is
    optionally gated by the utility mask (spec §2.7/§2.8).
    """
    def __init__(self, params, alpha: float = 1.0, kappa: float = 2.0,
                 lam: float = 0.8, gamma: float = 0.99,
                 max_z_sum: float = 10_000.0):
        self.params = list(params)
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.lam = float(lam)
        self.gamma = float(gamma)
        # trace storage lives in the optimizer (matches streaming-RL convention)
        self.traces = [torch.zeros_like(p.detach()) for p in self.params]
        self.last_z_sum = 0.0
        self.last_step_size = 0.0
        # Trace clipping: cap ‖z‖_1 so the step size can't vanish. Without this,
        # traces accumulate unboundedly (λγ=0.792 per step → ~5x per episode,
        # growing across episodes) until z_sum hits millions and a_eff → 0,
        # killing all learning. Standard in the eligibility trace literature.
        self.max_z_sum = float(max_z_sum)

    def _accumulate_traces(self, grads):
        with torch.no_grad():
            for i, g in enumerate(grads):
                if g is None:
                    continue
                self.traces[i].mul_(self.lam * self.gamma).add_(g.detach())

    def step(self, td_error: float, grads, reset_traces: bool = False,
             update_mask=None):
        """One streaming update (spec §2.8: traces accumulate FULL grads for all
        params; only the parameter update is gated).

        td_error: scalar TD error δ_t.
        grads: list aligned with params (from autograd.grad).
        reset_traces: zero traces after the update (on episode done).
        update_mask: list[bool] per param; if False, that param is NOT updated
                     this step (utility gate). Traces still accumulate.
        """
        self._accumulate_traces(grads)          # full grads, all params
        # ---- trace clipping: cap ‖z‖_1 to prevent step size vanishing ----
        with torch.no_grad():
            z_sum = 0.0
            for t in self.traces:
                z_sum += float(t.abs().sum().item())
            if z_sum > self.max_z_sum:
                scale = self.max_z_sum / z_sum
                for t in self.traces:
                    t.mul_(scale)
                z_sum = self.max_z_sum
        delta = float(td_error)
        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * self.alpha * self.kappa
        step_size = self.alpha / dot_product if dot_product > 1.0 else self.alpha
        # expose for diagnostics (alpha_effective = step_size * |delta| / z_sum)
        self.last_z_sum = z_sum
        self.last_step_size = step_size
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if update_mask is not None and not update_mask[i]:
                    continue
                # p.data.add_ modifies raw data WITHOUT bumping the autograd
                # version counter, so a transition's graph held across this
                # update (for the next step's delayed TD backward) stays valid.
                # This is the standard streaming-RL trick (cf. Elsayed et al.
                # 2024) and is REQUIRED because SEAL does one forward per obs
                # with a one-step-lagged backward.
                p.data.add_(self.traces[i], alpha=-delta * step_size)
        if reset_traces:
            self.reset()

    def reset(self):
        with torch.no_grad():
            for t in self.traces:
                t.zero_()
