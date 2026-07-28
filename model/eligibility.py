"""Eligibility traces for e-prop (Bellec et al. 2020, Eqs. 13-14, 22-25).

The eligibility trace ε_ji^t is the maximal locally-computable part of the
loss gradient dE/dW_ji. It is computed FORWARD in time during the network
rollout — no backward pass, no BPTT.

Core factorization (Eq. 1/3):
    dE/dW_ji = Σ_t  L_j^t · ε_ji^t

where L_j^t is a (later approximated) learning signal and ε_ji^t is:

    ε_ji^t = (∂z_j/∂h_j) · ε_vec_ji^t              (Eq. 13, the "eligibility trace")

with the eligibility VECTOR ε_vec satisfying the recursion (Eq. 14):

    ε_vec_ji^t = (∂h_j^t/∂h_j^{t-1}) · ε_vec_ji^{t-1} + ∂h_j^t/∂W_ji

This module provides per-synapse eligibility-vector storage and the recursion,
specialized for LIF (1-D h) and ALIF (2-D h = [v, a]) neurons.

LIF (Eq. 22-23):
    ε_vec = low-pass-filtered presynaptic spike:  ε_vec^{t+1} = α·ε_vec^t + z_i^t
    ε_ji^t = ψ_j^t · ε_vec^t                      (ψ = pseudo-derivative)

ALIF (Eq. 24-25): ε_vec is 2-D = [ε_v, ε_a]
    ε_v^{t+1}  = α·ε_v^t + z_i^t                       (same as LIF, membrane path)
    ε_a^{t+1}  = ψ_j^t·z_i^t + (ρ - ψ_j^t·β)·ε_a^t      (Eq. 24, adaptation path)
    ε_ji^t     = ψ_j^t · (ε_v^t - β·ε_a^t)              (Eq. 25)

All eligibility state lives in the LAYER (computed during forward), not in any
optimizer — this is what makes e-prop strictly online and BPTT-free.
"""
from __future__ import annotations
import torch

from model.neurons import LIFNeurons, ALIFNeurons


class LIFEligibility:
    """Per-synapse eligibility vector + trace for a population of LIF neurons.

    One eligibility vector PER synapse (j, i): ε_vec_ji ∈ R^1 (LIF hidden state
    is 1-D). Stored as a [n_post, n_pre] matrix (one scalar per synapse).

    Args:
        n_post: number of postsynaptic LIF neurons j
        n_pre:  number of presynaptic neurons i (provides spikes z_i)
        alpha:  membrane decay α = exp(-dt/τ_m)
    """
    def __init__(self, n_post: int, n_pre: int, alpha: float):
        self.n_post = n_post
        self.n_pre = n_pre
        self.alpha = float(alpha)
        # ε_vec_ji (scalar per synapse for LIF) — Eq. 22
        self.eps_v = torch.zeros(n_post, n_pre)
        # low-pass-filtered eligibility trace ε̄ (F_κ applied externally if needed)
        self.trace = torch.zeros(n_post, n_pre)

    def reset(self):
        self.eps_v.zero_()
        self.trace.zero_()

    def step(self, z_pre: torch.Tensor, psi_post: torch.Tensor) -> torch.Tensor:
        """Advance one ms. Returns the eligibility trace ε_ji^t [n_post, n_pre].

        Args:
            z_pre:     presynaptic spikes [n_pre] at time t-1 (drives ε_vec)
            psi_post:  pseudo-derivative ψ_j^t [n_post]
        """
        # Eq. 22: ε_vec^{t+1} = α·ε_vec^t + z_i^t  (low-pass of presynaptic spike)
        self.eps_v = self.alpha * self.eps_v + z_pre.unsqueeze(0)  # broadcast over j
        # Eq. 23: ε_ji^t = ψ_j^t · ε_vec^t
        self.trace = psi_post.unsqueeze(1) * self.eps_v
        return self.trace


class ALIFEligibility:
    """Per-synapse eligibility vector + trace for a population of ALIF neurons.

    ε_vec is 2-D per synapse: [ε_v, ε_a]. Stored as two [n_post, n_pre] matrices.

    Args:
        n_post, n_pre: as LIF
        alpha: membrane decay α
        rho:   adaptation decay ρ = exp(-dt/τ_a)
        beta:  adaptation increment β
        approx: if True, use Eq. 26 (drop ψ·β in the ε_a recursion).
    """
    def __init__(self, n_post: int, n_pre: int, alpha: float, rho: float,
                 beta: float, approx: bool = False):
        self.n_post = n_post
        self.n_pre = n_pre
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.beta = float(beta)
        self.approx = bool(approx)
        self.eps_v = torch.zeros(n_post, n_pre)  # membrane-path eligibility
        self.eps_a = torch.zeros(n_post, n_pre)  # adaptation-path eligibility
        self.trace = torch.zeros(n_post, n_pre)

    def reset(self):
        self.eps_v.zero_()
        self.eps_a.zero_()
        self.trace.zero_()

    def step(self, z_pre: torch.Tensor, psi_post: torch.Tensor) -> torch.Tensor:
        """Advance one ms. Returns the eligibility trace ε_ji^t [n_post, n_pre].

        Args:
            z_pre:    presynaptic spikes [n_pre]
            psi_post: pseudo-derivative ψ_j^t [n_post]
        """
        zpre = z_pre.unsqueeze(0)  # [1, n_pre] broadcast over j
        psi = psi_post.unsqueeze(1)  # [n_post, 1] broadcast over i
        # Eq. 22: ε_v^{t+1} = α·ε_v + z_i  (membrane path, same as LIF)
        self.eps_v = self.alpha * self.eps_v + zpre
        # Eq. 24: ε_a^{t+1} = ψ·z_i + (ρ - ψ·β)·ε_a   (or Eq. 26 approx: drop ψ·β)
        if self.approx:
            decay = self.rho
        else:
            decay = self.rho - psi * self.beta
        self.eps_a = psi * zpre + decay * self.eps_a
        # Eq. 25: ε_ji^t = ψ_j^t · (ε_v - β·ε_a)
        self.trace = psi * (self.eps_v - self.beta * self.eps_a)
        return self.trace


def make_eligibility(neurons, n_pre: int):
    """Factory: build the matching eligibility object for a neuron population."""
    if isinstance(neurons, LIFNeurons):
        return LIFEligibility(neurons.n, n_pre, neurons.alpha)
    elif isinstance(neurons, ALIFNeurons):
        return ALIFEligibility(neurons.n, n_pre, neurons.alpha, neurons.rho,
                               neurons.beta)
    raise TypeError(f"unknown neuron type: {type(neurons)}")
