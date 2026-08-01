"""SEAL core — the learned world model + value/direction model.

Four boxes, two learned:
  A (banded dynamics), B (action effect)  — the transition model.
    "if I do action a in state s, where will I be?"  Learns from prediction
    error every frame (self-supervised; no reward needed). The ball's
    autonomous motion is captured by A's banded shift structure (ball at
    position X → activation shifts one block to X+1), NOT by a constant
    bias — a constant bias grows into an attractor that collapses multi-step
    rollouts (see git history).
  D (inverse model, the paper's W)  — the value function.
    "to move toward the goal, which action has utility?"  Learns from action
    prediction every frame (self-supervised; no reward needed).

NOT learned (handled elsewhere):
  perception (fixed) → state s
  imagination (fixed procedure) → the policy: plans toward the geometric goal
  goal s* (geometric, read from the state) → "ball on opponent side"
  exploration ε (the only place the game ±1 reward enters)

No learned value function V, no learned policy π, no eligibility traces, no
TD, no credit assignment, no reward centering, no bias b. The game reward
steers exploration only (see training/success_tracker.py), not skill learning.
"""
from __future__ import annotations
import numpy as np
from collections import deque

from core.dynamics import BandedDynamics
from core.action_effect import ActionEffect
from core.direction import Direction
from core.gate import FeasibilityGate

from config import (
    N_STATE, N_ACTIONS, ETA_A, ETA_B, ETA_D,
    D_WEIGHT_DECAY, GOAL_WINDOW,
)


class SEALCore:
    """Learned transition model (A, B, b) + value/direction model (D).

    These are the only learned components. Both learn from self-supervised
    prediction error every frame — the game's ±1 reward is not used here.
    """

    def __init__(self):
        self.dynamics = BandedDynamics()       # A, b — transition model
        self.action_effect = ActionEffect()    # B — action effect
        self.direction = Direction()           # D — inverse model / value (paper's W)
        self.gate = FeasibilityGate()          # F — frozen (all-ones for Pong)

        # Rolling window of recent states for the geometric goal (s*).
        # NOT for credit assignment — s* is "ball on opponent side", read
        # geometrically by the imagination engine.
        self.recent_states: deque = deque(maxlen=GOAL_WINDOW)
        self.step_count = 0

    # ── Episode lifecycle ──────────────────────────────────────────
    def reset_episode(self):
        """Clear per-episode state on done. Weights persist across episodes."""
        # recent_states intentionally NOT cleared (rolling window for the goal)
        pass

    # ── Encoding ───────────────────────────────────────────────────
    def encode_action(self, action_idx: int) -> np.ndarray:
        a = np.zeros(N_ACTIONS, dtype=np.float32)
        a[action_idx] = 1.0
        return a

    # ── Prediction ─────────────────────────────────────────────────
    def predict_next_state(self, s: np.ndarray,
                           action_idx: int | None = None) -> np.ndarray:
        """Predict the next state: ŝ = A·s + B·a + b."""
        out = self.dynamics.forward(s)
        if action_idx is not None:
            out += self.action_effect.forward(self.encode_action(action_idx))
        return out

    # ── Per-step learning ──────────────────────────────────────────
    def step_learn(self, s_t: np.ndarray, action_idx: int,
                   s_tp1: np.ndarray, reward: float, done: bool,
                   learned_from_imagination: bool = False) -> dict:
        """Run the online updates for one transition.

        Only the transition model (A, B, b) and the inverse model (D) learn
        here, both from self-supervised prediction error. The `reward` and
        `done` args are accepted for API compatibility but are NOT used for
        learning — the game reward only steers exploration ε (in
        success_tracker.py), not skill acquisition.

        Args:
            s_t: current state (N_STATE,).
            action_idx: executed action index (0, 1, or 2).
            s_tp1: next state (N_STATE,).
            reward: raw reward (±1 or 0) — unused by learning (steers ε only).
            done: episode/life ended — clears per-episode buffers.
            learned_from_imagination: unused (kept for API compat).

        Returns:
            dict of diagnostic metrics.
        """
        a_onehot = self.encode_action(action_idx)
        self.step_count += 1

        # --- 1. Prediction error for the transition model ---
        pred_s = self.dynamics.predict(s_t, a_onehot, self.action_effect.B)
        err = s_tp1 - pred_s
        pred_err_norm = float(np.linalg.norm(err))

        # --- 2. Update transition model (A, B) from prediction error ---
        self.dynamics.update(err, s_t, ETA_A)
        self.action_effect.update(err, a_onehot, ETA_B)
        self.dynamics.clip()

        # --- 3. Update inverse model D from action prediction ---
        delta_s = s_tp1 - s_t
        self.direction.update(a_onehot, delta_s, ETA_D)
        self.direction.decay(D_WEIGHT_DECAY)

        # --- 4. Track recent states (for the geometric goal) ---
        self.recent_states.append(s_t.copy())

        # --- 5. On done: clear per-episode buffers ---
        if done:
            self.reset_episode()

        return {
            "pred_err_norm": pred_err_norm,
            "actual_reward": reward,
        }

    # ── Diagnostics ────────────────────────────────────────────────
    def _rollout_norm_ratio(self, s0: np.ndarray, horizon: int = 5) -> float:
        """‖A^H·s0‖ / ‖s0‖ — does the transition model shrink rollouts?

        A diagnostic for the imagination engine: if this drops far below 1,
        multi-step rollouts are collapsing toward a default state and the 40
        imagined futures become hard to distinguish.
        """
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
            "d_norm": float(np.linalg.norm(self.direction.D)),
        }
