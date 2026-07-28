"""E-prop optimizer: reward-based e-prop plasticity rule (Eq. 5/36) with an
ObGD-style overshooting bound on the step size.

    ΔW_ji = step · δ_t · F_γλ( L_j^t · ε̄_ji^t )

where:
  δ_t = r_t + γ·V_{t+1} − V_t            (reward prediction error, scalar)
  L_j^t = learning signal for neuron j   (from broadcast.py, Eq. 37)
  ε̄_ji^t = F_κ(low-pass-filtered eligibility trace)  (from eligibility.py)
  F_γλ = exponential low-pass with decay γ·λ (per-synapse tag, Eq. 5)

The tag  F_γλ(L_j · ε̄_ji)  is maintained per synapse and updated each env step.
It is the same mathematical object as ObGD's eligibility trace (model/optim.py):
here the role of "gradient" is played by L_j · ε̄_ji, the e-prop gradient
DIRECTION. ObGD contributes only the step SIZE:

    δ̄ = max(|δ|, 1)
    M = δ̄ · ||tag||₁ · η · κ          (projected effective step size)
    step = η / M  if M > 1  else η    (shrink only when overshooting)

This replaces the previous hard update clipping + 1/√episode_len η schedule,
which traded away ~99% of the step size for stability (η_eff ~ 7e-6) and made
learning glacial — the "stream barrier" (Elsayed et al. 2024). With the
overshooting bound, η can be O(1) and every sample takes the largest stable
step. The episode-LENGTH curriculum (config.episode_schedule) is unaffected;
only the η-coupling is removed (eta_length_scale now defaults off).

Sign convention: the rule ΔW = η·δ·tag implements policy-gradient ASCENT
(δ>0 reinforces the tagged directions) — we ADD step·δ·tag.

This optimizer handles the RECURRENT + INPUT weights (Win, Wrec) of the LSNN.
The readout + CNN weights are trained by autograd + ObGD (model/optim.py),
since they are feedforward and do not need e-prop theory (paper Methods).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class EpropOptimizer:
    """Applies the reward-based e-prop rule to a set of weight tensors.

    One tag tensor per weight, matching shape. Tags decay by γ·λ each step and
    accumulate L_j · ε̄_ji.

    Args:
        params: list of nn.Parameter (Win, Wrec) to optimize
        eta: base learning rate η (O(1) with the overshooting bound)
        gamma: discount γ (tag filter decay = γ·lam; paper uses F_γ, i.e. lam=1)
        lam: tag-filter λ (1.0 = paper's F_γ; <1 shortens the credit window)
        kappa: ObGD scaling factor κ (2.0 in the stream-x paper)
        grad_clip: clip on |δ · tag| per parameter (0 = off; ObGD supersedes it)
        length_scale: legacy — scale η by 1/sqrt(current_max_episode_len).
            Off by default; superseded by the overshooting bound.
    """
    def __init__(self, params, eta: float = 1.0, gamma: float = 0.99,
                 lam: float = 1.0, kappa: float = 2.0, grad_clip: float = 0.0,
                 length_scale: bool = False):
        self.params = list(params)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.kappa = float(kappa)
        self.grad_clip = float(grad_clip)
        self.length_scale = bool(length_scale)
        self.tags = [torch.zeros_like(p) for p in self.params]
        self._length_factor = 1.0  # set by set_episode_length() (legacy)
        self.last_update_norm = 0.0
        self.last_step_size = 0.0

    def reset(self):
        """Clear tags (episode boundary)."""
        for t in self.tags:
            t.zero_()

    def set_episode_length(self, max_len: int):
        """Legacy η ∝ 1/√len scaling (disabled by default; ObGD supersedes it)."""
        if self.length_scale and max_len > 0:
            self._length_factor = 1.0 / (max_len ** 0.5)
        else:
            self._length_factor = 1.0

    def accumulate(self, learning_signal: torch.Tensor, elig_trace: torch.Tensor,
                   weight: torch.Tensor, param_idx: int):
        """Update the tag for one weight tensor: tag <- γλ·tag + L_j · ε̄_ji.

        Runs under torch.no_grad() and detaches inputs — the tag is a manual
        accumulator, NOT part of any autograd graph. Without detaching, a
        grad_fn on learning_signal (via self.B) would accumulate an ever-growing
        graph in self.tags, leaking ~30 MB/step.
        """
        with torch.no_grad():
            # tag_ij <- γλ·tag_ij + L_i · ε̄_ij   (outer product, L over post rows)
            contrib = learning_signal.detach().unsqueeze(1) * elig_trace.detach()
            decay = self.gamma * self.lam
            self.tags[param_idx] = decay * self.tags[param_idx] + contrib

    def step(self, delta: float):
        """Apply the weight update: W <- W + step·δ·tag  (policy-gradient ASCENT).

        The step size is overshooting-bounded (ObGD): shrink η by the
        projected effective step size M = δ̄·||tag||₁·η·κ whenever M > 1.

        Args:
            delta: scalar reward prediction error δ_t
        """
        with torch.no_grad():
            z_sum = 0.0
            for tag in self.tags:
                z_sum += tag.abs().sum().item()
            d_bar = max(abs(delta), 1.0)
            M = d_bar * z_sum * self.eta * self.kappa
            step_size = self.eta / M if M > 1.0 else self.eta
            step_size *= self._length_factor
            self.last_step_size = step_size

            total_norm = 0.0
            for p, tag in zip(self.params, self.tags):
                update = step_size * delta * tag
                if self.grad_clip > 0:
                    un = update.norm().item()
                    if un > self.grad_clip:
                        update = update * (self.grad_clip / (un + 1e-12))
                p.data += update
                total_norm += float((update.norm().item()) ** 2)
        self.last_update_norm = total_norm ** 0.5

    @property
    def tag_norms(self):
        return [float(t.norm().item()) for t in self.tags]
