"""Successor-feature value V_sf(s) — "where does this state lead?"

V_sf(s) = θ_sf · s, learned by streaming average-reward TD(λ) on the
REWARD-PREDICTOR stream r̂(s) instead of the raw environment reward.

The successor-features decomposition (Dayan 1993; Barreto et al. 2017) says
the value of a state under the reward function r̂(s) = w_R·s is

    V^π(s; w_R) = ψ^π(s) · w_R,

where ψ^π(s) = E_π[Σ_k γ^k φ(s_k) | s_0 = s] is the successor representation
— expected discounted future state visitation. Learning ψ explicitly costs a
1296×1296 matrix; learning the *composition* ψ·w_R directly with TD on the
auxiliary reward r̂(s') costs one 1296-vector and converges to exactly the
same fixed point:

    δ_sf = (r̂(s') − ρ_sf) + V_sf(s') − V_sf(s)

Average-reward form, independent of the RVI_ENABLE flag: ρ_sf tracks the
mean of the r̂ stream (a different stream from the environment reward), and
centring against it is what keeps V_sf discriminative instead of drifting to
a constant.

What it buys: the raw environment reward on Pong arrives at a steady rate
with the score absent from the state, so a myopic V cannot separate states.
r̂(s') is queried from the learned reward predictor every frame, so credit
propagates densely through the eligibility trace and V_sf learns the
forward-looking landscape. On Pong the dominant reward-predicting region is
LOSS (~21 losses per win), so V_sf mainly encodes avoidance: rollouts that
lead toward imminent losses score low. Avoidance is discriminative — exactly
what the planner needs for ranking.

Same streaming machinery as core/value.py — overshooting-bounded step
(V_ALPHA_MAX), trace clip (V_TRACE_CLIP), normalized accumulating trace —
with its own trace and its own ρ. Imagined TD trains the main critic only;
V_sf learns from real transitions in v1.
"""
from __future__ import annotations
import numpy as np

from config import (N_STATE, V_INIT_STD, V_WEIGHT_DECAY, V_TRACE_CLIP,
                    V_ALPHA_MAX, ETA_RHO, SF_SEED)
from core.value import Value


class SuccessorValue(Value):
    """Linear value over the reward-predictor stream: V_sf(s) ≈ ψ(s)·w_r̂.

    Inherits forward/reset_trace from Value. self.rho here tracks the mean
    of the AUXILIARY (r̂) stream — same attribute, different stream.
    """

    def __init__(self, n_state: int = N_STATE, init_std: float = V_INIT_STD,
                 seed: int = SF_SEED):
        super().__init__(n_state=n_state, init_std=init_std, seed=seed)

    def update(self, s: np.ndarray, reward: float, s_next: np.ndarray,
               done: bool, gamma: float, lam: float, eta: float,
               rho_update: bool = True) -> float:
        """One average-reward TD(λ) update on an auxiliary (r̂) reward.

        Always average-reward, regardless of RVI_ENABLE: the auxiliary
        stream has its own steady mean, and centring on it is what keeps
        V_sf discriminative. Args match Value.update; `reward` here is
        r̂(s_next), not the environment reward.

        Returns the TD error δ_sf (for logging).
        """
        if rho_update:
            self.rho += ETA_RHO * (reward - self.rho)

        v = self.forward(s)
        v_next = self.forward(s_next)
        # Continuing-task form: no γ, no zero-on-done.
        delta = (reward - self.rho) + v_next - v

        # Accumulating eligibility trace (normalized; own trace, own stream).
        norm_sq = max(float(s @ s), 1.0)
        self.e = lam * self.e + s / norm_sq
        if V_TRACE_CLIP > 0:
            trace_norm = np.linalg.norm(self.e)
            if trace_norm > V_TRACE_CLIP:
                self.e *= V_TRACE_CLIP / trace_norm

        # Overshooting-bounded step (same anti-drift net as the main critic).
        ge = float(self.e @ self.e)
        if ge > 1e-12:
            alpha = min(eta, V_ALPHA_MAX / np.sqrt(ge))
        else:
            alpha = eta
        self.w += alpha * delta * self.e

        if V_WEIGHT_DECAY > 0:
            self.w *= (1.0 - V_WEIGHT_DECAY)

        return float(delta)
