"""SEAL core — OaK-aligned learned components, fully streaming.

Learned components:
  A (banded dynamics) + B (action effect)  — the transition model (knowledge).
  D (inverse model, the paper's W)         — planning direction.
  V(s) (linear value)                      — streaming TD(λ) critic.
  π(s) (softmax policy)                    — streaming actor-critic.

The transition model A/B learns from self-supervised prediction error every
frame. The value function V and policy π learn from the sparse game reward
with per-step TD(λ) updates — one transition in, one update out, no episode
buffering (Elsayed et al. 2024, "Streaming Deep RL Finally Works"). An
overshooting-bounded step size on V stops the bootstrap-driven drift that naive
TD shows on sparse rewards, keeping SEAL strictly streaming (O(1)/step, no
replay, no backward return sweep). Imagination uses A/B to generate rollouts,
V to score them, and π learns by imitating the best plans.

No backpropagation, no replay buffer, no Monte Carlo fallback.
"""
from __future__ import annotations
import numpy as np
from collections import deque

from core.dynamics import BandedDynamics
from core.action_effect import ActionEffect
from core.direction import Direction
from core.gate import FeasibilityGate
from core.value import Value
from core.policy import Policy

from config import (
    N_STATE, N_ACTIONS, ETA_A, ETA_B, ETA_D, ETA_V, ETA_PI_IMIT, ETA_PI_AC,
    GAMMA, LAMBDA, D_WEIGHT_DECAY, GOAL_WINDOW,
    PRE_SCORE_WINDOW, PRE_SCORE_MEMORY,
)


class SEALCore:
    """Learned transition model + streaming value + policy."""

    def __init__(self):
        self.dynamics = BandedDynamics()       # A — transition model
        self.action_effect = ActionEffect()    # B — action effect
        self.direction = Direction()           # D — inverse model
        self.gate = FeasibilityGate()          # F — frozen
        self.value = Value()                   # V — streaming TD(λ) critic
        self.policy = Policy()                 # π — streaming actor-critic

        # Rolling window of recent states for the geometric goal (s*).
        self.recent_states: deque = deque(maxlen=GOAL_WINDOW)

        # Pre-score memory: states that preceded an actual +1 (goal label).
        self.pre_score_states: deque = deque(maxlen=PRE_SCORE_MEMORY)
        self._pre_score_window: deque = deque(maxlen=PRE_SCORE_WINDOW)

        self.step_count = 0

    # ── Episode lifecycle ──────────────────────────────────────────
    def reset_episode(self):
        """Clear per-episode state on done. Weights persist across episodes."""
        self.value.reset_trace()

    # ── Encoding ───────────────────────────────────────────────────
    def encode_action(self, action_idx: int) -> np.ndarray:
        a = np.zeros(N_ACTIONS, dtype=np.float32)
        a[action_idx] = 1.0
        return a

    # ── Prediction ─────────────────────────────────────────────────
    def predict_next_state(self, s: np.ndarray,
                           action_idx: int | None = None) -> np.ndarray:
        """Predict the next state: ŝ = A·s + B·a."""
        out = self.dynamics.forward(s)
        if action_idx is not None:
            out += self.action_effect.forward(self.encode_action(action_idx))
        return out

    # ── Per-step learning ──────────────────────────────────────────
    def step_learn(self, s_t: np.ndarray, action_idx: int,
                   s_tp1: np.ndarray, reward: float, done: bool,
                   source: str = "epsilon") -> dict:
        """Run the online updates for one transition.

        Every component updates from this single transition and then discards
        it — no episode buffering. Args:
            s_t: current state.
            action_idx: executed action.
            s_tp1: next state.
            reward: raw reward (±1 or 0).
            done: episode/life ended.
            source: which gate produced the action ("epsilon","policy",
                    "greedy","top5","no_goal").

        Returns:
            dict of diagnostic metrics.
        """
        a_onehot = self.encode_action(action_idx)
        self.step_count += 1

        # --- 1. Transition model (A, B) — self-supervised, no reward ---
        pred_s = self.dynamics.predict(s_t, a_onehot, self.action_effect.B)
        err = s_tp1 - pred_s
        pred_err_norm = float(np.linalg.norm(err))

        self.dynamics.update(err, s_t, ETA_A)
        self.action_effect.update(err, a_onehot, ETA_B, s_t=s_t)
        self.dynamics.clip()

        # --- 2. Inverse model D — trained on the prediction residual ---
        # The residual is dominated by the action effect B·a, so D learns to
        # map "how the action changed the world" onto the action — exactly the
        # mapping needed for planning.
        self.direction.update(a_onehot, err, ETA_D)
        self.direction.decay(D_WEIGHT_DECAY)

        # --- 3. Critic V: streaming TD(λ) ---
        # One transition in → one update out. The overshooting bound inside
        # value.update caps the per-step correction so a wrong bootstrap V(s')
        # can't drift the weights on the many r=0 frames.
        td_delta = self.value.update(
            s_t, reward, s_tp1, done, gamma=GAMMA, lam=LAMBDA, eta=ETA_V
        )

        # --- 4. Actor π: streaming actor-critic + imagination imitation ---
        # Reinforce the taken action by the TD error δ (the streaming
        # advantage). We do this for every transition, including ε-random ones:
        # a negative δ correctly discourages whatever was taken in a state V
        # now thinks is bad. (Imitation below also nudges π toward imagination.)
        self.policy.update(s_t, action_idx, eta=ETA_PI_AC, scale=td_delta)

        # Imitation: when imagination chose the action, also nudge π toward it.
        if source in ("greedy", "top5"):
            self.policy.update_imitation(s_t, action_idx, ETA_PI_IMIT)

        # --- 5. Goal memory ---
        self.recent_states.append(s_t.copy())
        self._pre_score_window.append(s_t.copy())
        if reward > 0:
            for s_pre in self._pre_score_window:
                self.pre_score_states.append(s_pre.copy())
            self._pre_score_window.clear()

        # --- 6. Episode boundary: just clear the trace ---
        if done:
            self.reset_episode()

        return {
            "pred_err_norm": pred_err_norm,
            "actual_reward": reward,
            "td_delta": td_delta,
        }

    # ── Diagnostics ────────────────────────────────────────────────
    def _rollout_norm_ratio(self, s0: np.ndarray, horizon: int = 5) -> float:
        """‖A^H·s0‖ / ‖s0‖ — diagnostic for rollout shrinkage."""
        s = s0.copy()
        for _ in range(horizon):
            s = self.dynamics.forward(s)
        n0 = np.linalg.norm(s0) + 1e-8
        return float(np.linalg.norm(s) / n0)

    def diagnostics(self) -> dict:
        return {
            "step_count": self.step_count,
            "a_op_norm": self.dynamics.operator_norm_estimate(),
            "n_recent_states": len(self.recent_states),
            "n_pre_score_states": len(self.pre_score_states),
            "d_norm": float(np.linalg.norm(self.direction.D)),
            "v_norm": float(np.linalg.norm(self.value.w)),
            "pi_norm": float(np.linalg.norm(self.policy.theta)),
        }
