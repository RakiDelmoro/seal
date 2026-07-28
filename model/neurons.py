"""Spiking neuron models: LIF and ALIF (Bellec et al. 2020, Eqs. 6-10).

Discrete-time dynamics, dt = 1 ms. All state is per-neuron, batch-size 1
(online learning). Neurons are grouped: a population is either all-LIF or
all-ALIF. The LSNN core (lsnn.py) mixes populations.

LIF neuron (Eq. 6-7):
    v_{t+1} = α·v_t + Σ_i W_ji·z_i^t + Σ_i Win_ji·x_i^t - z_j^t·v_th
    z_j^t   = H(v_t - v_th)              (H = Heaviside)
  hidden state h = [v]  (1-D)

ALIF neuron (Eq. 8-10) — LIF + adapting threshold:
    A_j^t   = v_th + β·a_j^t
    z_j^t   = H(v_t - A_t)
    a_{t+1} = ρ·a_t + z_j^t
  hidden state h = [v, a]  (2-D)

Pseudo-derivative (ref. 3,4 of the paper): the non-differentiable spike is
replaced by
    ψ_j^t = (γ_pd / v_th) · max(0, 1 - |v_t - A_t| / v_th)
set to 0 during the refractory period.

 refractory: z_j forced to 0 for `refractory_steps` ms after each spike.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def pseudo_derivative(v: torch.Tensor, threshold: torch.Tensor,
                      gamma_pd: float, v_th: float) -> torch.Tensor:
    """ψ = (γ_pd / v_th) · max(0, 1 - |v - threshold| / v_th).

    `threshold` is v_th (LIF) or A = v_th + β·a (ALIF). Returns a tensor the
    shape of v. Straight-through: this is the surrogate gradient used in the
    eligibility trace (Eq. 23/25), NOT a differentiable function in autograd.
    """
    return (gamma_pd / v_th) * F.relu(1.0 - (v - threshold).abs() / v_th)


class LIFNeurons:
    """Population of Leaky Integrate-and-Fire neurons (1-D hidden state).

    State (per neuron, shape [n]):
        v  — membrane potential
        ref — refractory countdown (steps remaining; 0 = can fire)

    Args:
        n: number of neurons
        alpha: decay factor exp(-dt/τ_m)
        v_th: firing threshold
        gamma_pd: pseudo-derivative scale
        refractory_steps: steps z is forced to 0 after a spike
    """
    def __init__(self, n: int, alpha: float, v_th: float, gamma_pd: float,
                 refractory_steps: int):
        self.n = n
        self.alpha = float(alpha)
        self.v_th = float(v_th)
        self.gamma_pd = float(gamma_pd)
        self.refractory = int(refractory_steps)
        self.v = torch.zeros(n)
        self.ref = torch.zeros(n, dtype=torch.long)

    def reset(self):
        self.v.zero_()
        self.ref.zero_()

    def step(self, i_syn: torch.Tensor) -> tuple:
        """Advance one ms. i_syn = Σ_i W_ji·z_i + Σ_i Win_ji·x_i (shape [n]).

        Returns (z, psi):
            z   — binary spikes [n]
            psi — pseudo-derivative ψ [n] (for eligibility traces)

        Runs under torch.no_grad() — e-prop computes gradients manually via
        eligibility traces, so the neuron dynamics must NOT build an autograd
        graph (it would leak memory since we never backward through the core).
        """
        with torch.no_grad():
            # membrane update (Eq. 6): v_{t+1} = α·v_t + i_syn
            v_new = self.alpha * self.v + i_syn
            # threshold test (Eq. 7); refractory neurons cannot fire
            can_fire = (self.ref <= 0)
            z = (v_new >= self.v_th) & can_fire
            z = z.to(self.v.dtype)
            # pseudo-derivative at the PRE-reset v_new (spike decision point)
            psi = pseudo_derivative(v_new, torch.full_like(v_new, self.v_th),
                                    self.gamma_pd, self.v_th)
            psi = torch.where(can_fire, psi, torch.zeros_like(psi))
            # subtractive reset (Eq. 6 term -z·v_th)
            v_new = torch.where(z > 0.5, v_new - self.v_th, v_new)
            # refractory countdown
            self.ref = torch.where(z > 0.5, torch.full_like(self.ref, self.refractory),
                                   torch.clamp(self.ref - 1, min=0))
            self.v = v_new
        return z, psi

    def dh_dh_prev(self) -> torch.Tensor:
        """∂h_t/∂h_{t-1} = ∂v_t/∂v_{t-1} = α (scalar, Eq. 22)."""
        return torch.tensor(self.alpha)


class ALIFNeurons:
    """Population of Adaptive LIF neurons (2-D hidden state h = [v, a]).

    State (per neuron, shape [n]):
        v  — membrane potential
        a  — adaptation variable (component of threshold)
        ref — refractory countdown

    Args:
        n, alpha, v_th, gamma_pd, refractory_steps: as LIF
        beta: adaptation increment β (Eq. 8)
        rho: adaptation decay exp(-dt/τ_a)
    """
    def __init__(self, n: int, alpha: float, rho: float, v_th: float,
                 beta: float, gamma_pd: float, refractory_steps: int):
        self.n = n
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.v_th = float(v_th)
        self.beta = float(beta)
        self.gamma_pd = float(gamma_pd)
        self.refractory = int(refractory_steps)
        self.v = torch.zeros(n)
        self.a = torch.zeros(n)
        self.ref = torch.zeros(n, dtype=torch.long)

    def reset(self):
        self.v.zero_()
        self.a.zero_()
        self.ref.zero_()

    def threshold(self) -> torch.Tensor:
        """A_t = v_th + β·a_t (Eq. 8)."""
        return self.v_th + self.beta * self.a

    def step(self, i_syn: torch.Tensor) -> tuple:
        """Advance one ms. Returns (z, psi).

        Runs under torch.no_grad() — e-prop computes gradients manually via
        eligibility traces, so neuron dynamics must NOT build an autograd graph.
        """
        with torch.no_grad():
            v_new = self.alpha * self.v + i_syn
            thr = self.threshold()  # uses CURRENT a (before update)
            can_fire = (self.ref <= 0)
            z = (v_new >= thr) & can_fire
            z = z.to(self.v.dtype)
            # pseudo-derivative at the PRE-reset v_new, threshold = A_t
            psi = pseudo_derivative(v_new, thr, self.gamma_pd, self.v_th)
            psi = torch.where(can_fire, psi, torch.zeros_like(psi))
            # reset membrane on spike (subtractive: v - A_t)
            v_new = torch.where(z > 0.5, v_new - thr, v_new)
            # adaptation update (Eq. 10): a_{t+1} = ρ·a_t + z_t
            a_new = self.rho * self.a + z
            # refractory countdown
            self.ref = torch.where(z > 0.5, torch.full_like(self.ref, self.refractory),
                                   torch.clamp(self.ref - 1, min=0))
            self.v = v_new
            self.a = a_new
        return z, psi

    def dh_dh_prev(self, psi: torch.Tensor) -> torch.Tensor:
        """∂h_t/∂h_{t-1} as a 2x2 matrix (per-neuron, Eq. 24 derivation).

        For ALIF the Jacobian of [v_t, a_t] w.r.t. [v_{t-1}, a_{t-1}] is:
            ∂v_t/∂v_{t-1} = α
            ∂a_t/∂a_{t-1} = ρ - ψ·β        (the diagonal from Eq. 24)
            ∂v_t/∂a_{t-1} = 0
            ∂a_t/∂v_{t-1} = ψ              (off-diagonal; spike drives adaptation)

        Returns a [n, 2, 2] tensor (per-neuron Jacobian).
        `psi` is the pseudo-derivative from THIS step (shape [n]).
        """
        n = self.n
        J = torch.zeros(n, 2, 2)
        J[:, 0, 0] = self.alpha        # ∂v/∂v_prev
        J[:, 1, 1] = self.rho - psi * self.beta   # ∂a/∂a_prev (Eq. 24)
        J[:, 1, 0] = psi               # ∂a/∂v_prev
        # ∂v/∂a_prev = 0
        return J
