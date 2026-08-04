"""Value function V(s) — linear readout learned by streaming TD(λ).

V(s) = w · s

Streaming actor-critic critic (Elsayed et al. 2024, "Streaming Deep RL Finally
Works"). Every transition triggers exactly one online update — no episode
buffering, no backward return sweep, no Monte Carlo fallback:

    δ_t   = r_t + γ V(s_{t+1}) − V(s_t)            (discounted TD error)
        or, with RVI_ENABLE:
    δ_t   = (r_t − ρ) + V(s_{t+1}) − V(s_t)        (average-reward / RVI error)
    e_t   = γλ e_{t-1} + ∇V(s_t)                    (accumulating eligibility trace)
    α_eff = min(η, α_max · ‖δ_t·e_t‖ / ‖δ_t·e_t‖²) (overshooting-bounded step)
    w    += α_eff · δ_t · e_t                       (then weight decay)
The overshooting bound (α_max) is the key streaming anti-drift knob: it caps
each update to a fixed fraction of the current TD error, so the bootstrap
target V(s') — which is wrong early on — cannot pump the weights arbitrarily.
A long λ (0.95) gives near-MC credit assignment while staying O(1)/step.

Average-reward / RVI mode (Yu, Wan & Sutton 2025, arXiv:2512.06218): the TD
error drops γ and subtracts the agent's own running reward rate ρ. This is the
fix for the constant-offset pathology: when rewards arrive at a steady rate
(Pong loses ~every 9 frames) and the score is not part of the state, a
discounted V converges to a negative constant — every state predicts the same
remaining loss count, so V carries no action information. Subtracting ρ
centres the reward stream: quiet frames drip in small positives, loss frames
deliver a sharp negative, and V settles into a per-state landscape
("states that lose the ball sooner than my own pace are worse"). ρ is a
single scalar updated online from REAL rewards only and saved in checkpoints.
"""
from __future__ import annotations
import numpy as np

from config import (N_STATE, V_INIT_STD, V_WEIGHT_DECAY, V_TRACE_CLIP,
                    V_ALPHA_MAX, V_SEED, RVI_ENABLE, ETA_RHO)


class Value:
    """Linear state-value function V(s) = w · s, streaming TD(λ).

    In average-reward (RVI) mode the TD error subtracts the running reward
    rate ρ and drops γ; ρ lives here so it checkpoints with the critic.
    """

    def __init__(self, n_state: int = N_STATE, init_std: float = V_INIT_STD,
                 seed: int = V_SEED):
        rng = np.random.default_rng(seed)
        self.w = (rng.normal(0, init_std, n_state)
                  ).astype(np.float32)
        self.e = np.zeros(n_state, dtype=np.float32)
        # Average-reward critic: running estimate of reward-per-step.
        self.rho = 0.0

    def forward(self, s: np.ndarray) -> float:
        """V(s) = w · s."""
        return float(self.w @ s)

    def reset_trace(self):
        """Clear eligibility trace at episode/life boundary."""
        self.e.fill(0.0)

    def update(self, s: np.ndarray, reward: float, s_next: np.ndarray,
               done: bool, gamma: float, lam: float, eta: float,
               rho_update: bool = True) -> float:
        """One streaming TD(λ) update from a single transition.

        Args:
            s: current state s_t.
            reward: observed reward r_t.
            s_next: next state s_{t+1} (ignored if done).
            done: episode/life ended → bootstrap target is 0.
            gamma: discount γ (discounted mode only).
            lam: trace decay λ.
            eta: nominal learning rate η.
            rho_update: whether this reward may update ρ (REAL transitions
                only; imagined transitions pass False — the model must not
                feed its own predictions back into the reward-rate estimate).

        Returns:
            The TD error δ_t (signed) for the actor to use as advantage.
        """
        if RVI_ENABLE:
            if rho_update:
                self.rho += ETA_RHO * (reward - self.rho)
            v = self.forward(s)
            v_next = self.forward(s_next)
            # Average-reward TD: no γ, no zero-on-done — the world is a
            # continuing task and a life boundary doesn't reset the rate.
            delta = (reward - self.rho) + v_next - v
        else:
            v = self.forward(s)
            v_next = 0.0 if done else self.forward(s_next)
            delta = reward + gamma * v_next - v

        # Accumulating eligibility trace (normalized so ‖s‖ doesn't blow it up).
        norm_sq = max(float(s @ s), 1.0)
        decay = lam if RVI_ENABLE else gamma * lam
        self.e = decay * self.e + s / norm_sq
        if V_TRACE_CLIP > 0:
            trace_norm = np.linalg.norm(self.e)
            if trace_norm > V_TRACE_CLIP:
                self.e *= V_TRACE_CLIP / trace_norm

        # Overshooting-bounded effective step size. The raw update is
        # η·δ·e; its projection on the error direction δ·e/‖δ·e‖ is η·‖δ·e‖.
        # Capping the per-step correction to α_max·|δ| stops a wrong bootstrap
        # (V(s') ~ random early on) from pumping the weights — the streaming
        # fix for the drift naive TD shows on sparse rewards.
        ge = float(self.e @ self.e)
        if ge > 1e-12:
            alpha = min(eta, V_ALPHA_MAX / np.sqrt(ge))
        else:
            alpha = eta
        self.w += alpha * delta * self.e

        if V_WEIGHT_DECAY > 0:
            self.w *= (1.0 - V_WEIGHT_DECAY)

        return float(delta)

    def update_imagined(self, s: np.ndarray, reward: float,
                        s_next: np.ndarray, eta: float,
                        gamma: float) -> float:
        """One-step TD update from an IMAGINED transition.

        Same math and same overshooting bound as update(), but:
          (a) λ = 0 — each imagined transition uses its own instantaneous
              trace s/‖s‖², and
          (b) the real eligibility trace self.e is left untouched.

        The eligibility trace is a thread through *actually experienced*
        consecutive states — that is what makes TD(λ) credit assignment
        valid. Imagined states did not really follow one another in the
        world, so mixing them into that thread would smear credit across
        real and imagined time. Each imagined transition therefore gets its
        own small standalone update instead.

        Returns the TD error δ (for logging).
        """
        v = self.forward(s)
        v_next = self.forward(s_next)
        if RVI_ENABLE:
            delta = (reward - self.rho) + v_next - v
        else:
            delta = reward + gamma * v_next - v

        e = s / max(float(s @ s), 1.0)      # instantaneous trace (λ=0)
        ge = float(e @ e)
        if ge > 1e-12:
            alpha = min(eta, V_ALPHA_MAX / np.sqrt(ge))
        else:
            alpha = eta
        self.w += alpha * delta * e

        if V_WEIGHT_DECAY > 0:
            self.w *= (1.0 - V_WEIGHT_DECAY)

        return float(delta)
