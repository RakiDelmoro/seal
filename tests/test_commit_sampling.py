"""Commit sampling validation — distinct openings + stubbornness.

The diagnosis this feature answers: on Pong the 40 rollouts pick different
first actions ~44% of the time, yet endpoints end only 0.31 apart — every
rollout re-aims at the SAME goal each step and merges back together.

Commit sampling:
  (1) DEALING — first actions are handed out (K/N_ACTIONS per action), not
      drawn from the steering+noise lottery.
  (2) STUBBORNNESS — each rollout keeps a bonus on its committed action for
      the whole horizon, scaled by steering strength so a STRONG goal can
      still override it.

Tests use synthetic components that reproduce the merge exactly: identity
dynamics, distinct action effects, and direction matrices whose steering is
either dominant (action 2 by a wide margin) or near-tie (the Pong regime).

  1. Dealt openings are balanced across all actions.
  2. Without commit, all rollouts follow dominant steering (the merge).
  3. Dealing ALONE (bonus=0) does not stop the merge — openings differ but
     endpoints reconverge. This is the measured diagnosis, defended as a test.
  4. Dealing + stubbornness keeps the plans apart in the near-tie regime.
  5. The --commit CLI override (module attribute) resolves at call time.
"""
import numpy as np
import pytest

import imagination.sampler as sampler_module
from imagination.sampler import sample_trajectories
from config import N_ACTIONS


# ── Synthetic components reproducing the Pong merge ────────────────

class _IdentityDynamics:
    """ŝ_next = ŝ + B·a (no autonomous drift, no shrinkage)."""

    def predict_batch(self, S, A_actions, B=None):
        return S + A_actions @ B.T


class _ActionEffect:
    def __init__(self, n_state):
        # Distinct effects along axis 0: UP=+1, STAY=0, DOWN=−1.
        B = np.zeros((n_state, N_ACTIONS), dtype=np.float32)
        B[0, 0] = 1.0
        B[0, 2] = -1.0
        self.B = B


class _Gate:
    def forward(self):
        return np.ones(N_ACTIONS, dtype=np.float32)


def _dominant_direction(n_state):
    """Steering with action 2 overwhelmingly dominant (gap ~50)."""
    D = np.zeros((N_ACTIONS, n_state), dtype=np.float32)
    D[2, 0] = 10.0
    return D


def _tie_direction(n_state):
    """Near-tie steering (the Pong regime): action 2 leads action 0 by a
    margin smaller than the stubbornness bonus can cover."""
    D = np.zeros((N_ACTIONS, n_state), dtype=np.float32)
    D[2, 0] = 6.0
    D[0, 0] = 5.0
    return D


class _Direction:
    """Holder exposing .D like core.direction.Direction."""
    def __init__(self, D):
        self.D = D


def _sample(direction_fn, K=12, horizon=5, seed=0, **kw):
    dyn = _IdentityDynamics()
    ae = _ActionEffect(1296)
    gate = _Gate()
    s0 = np.zeros(1296, dtype=np.float32)
    s0[1] = 1.0                        # nonzero: the sampler renormalizes to ‖s0‖
    s_star = np.zeros(1296, dtype=np.float32)
    s_star[0] = 5.0
    return sample_trajectories(s0, s_star, dyn, ae,
                               _Direction(direction_fn(1296)), gate,
                               n_trajectories=K, horizon=horizon,
                               rng=np.random.default_rng(seed), **kw)


# ── 1. Dealt openings are balanced ─────────────────────────────────

def test_dealt_openings_balanced():
    trajs = _sample(_dominant_direction, K=12, horizon=1,
                    commit_enable=True, commit_bonus=0.0)
    firsts = [t["first_action"] for t in trajs]
    for a in range(N_ACTIONS):
        assert firsts.count(a) == 4, f"action {a} dealt {firsts.count(a)}×"


# ── 2. Without commit, everyone follows dominant steering ──────────

def test_commit_off_merges_on_dominant_steering():
    trajs = _sample(_dominant_direction, K=12, horizon=5,
                    commit_enable=False)
    firsts = [t["first_action"] for t in trajs]
    assert firsts.count(2) >= 10, f"expected ~all action 2, got {firsts}"


# ── 3. Dealing alone does NOT stop the merge (the diagnosis) ───────

def test_dealing_alone_still_merges():
    """Distinct openings + bonus=0 against dominant steering: openings
    differ, but the goal-pull reconverges every rollout — endpoints stay
    tight. Different first actions, identical futures: the measured Pong
    failure mode."""
    trajs = _sample(_dominant_direction, K=12, horizon=5,
                    commit_enable=True, commit_bonus=0.0)
    firsts = [t["first_action"] for t in trajs]
    assert len(set(firsts)) == N_ACTIONS, "openings must be distinct"
    ends = np.stack([t["states"][-1] for t in trajs])
    spread = np.mean(np.linalg.norm(ends[:, None] - ends[None, :], axis=2))
    assert spread < 1.0, f"dealing alone must still merge, spread={spread:.3f}"


# ── 4. Dealing + stubbornness keeps the plans apart (the fix) ──────

def test_commit_bonus_keeps_plans_apart():
    """Near-tie steering (the Pong regime): goal-pull alone would send
    nearly everyone to the marginally-best action. With stubbornness, each
    rollout holds its opening (bonus exceeds the small margin) and the
    plans stay separated."""
    trajs = _sample(_tie_direction, K=12, horizon=5, commit_enable=True,
                    commit_bonus=1.0, commit_scale=0.5)
    ends = np.stack([t["states"][-1] for t in trajs])
    spread = np.mean(np.linalg.norm(ends[:, None] - ends[None, :], axis=2))
    assert spread > 1.0, f"committed plans must stay apart, spread={spread:.3f}"


# ── 5. Stubbornness scales with steering strength ──────────────────

def test_bonus_scales_with_steering_strength():
    """The bonus is commit_scale × ‖u‖: with a large bonus coefficient it
    flips near-tie continuations to the committed action even when the
    margin favors another action."""
    trajs = _sample(_tie_direction, K=3, horizon=3, commit_enable=True,
                    commit_bonus=1.0, commit_scale=0.5)
    by_action = {t["first_action"]: t for t in trajs}
    # UP-committed rollout keeps UP (axis-0 coordinate grows) despite the
    # margin favoring action 2.
    assert by_action[0]["states"][-1][0] > 0


# ── 6. Call-time flag resolution (the CLI override path) ───────────

def test_module_flag_resolves_at_call_time(monkeypatch):
    """train.py patches imagination.sampler.SAMPLER_COMMIT_ENABLE; the
    sampler must respect the patched value (not the import-time one)."""
    monkeypatch.setattr(sampler_module, "SAMPLER_COMMIT_ENABLE", True)
    trajs = _sample(_dominant_direction, K=12, horizon=1)  # implicit flag
    firsts = [t["first_action"] for t in trajs]
    assert len(set(firsts)) == N_ACTIONS, "flag on at call time must deal"

    monkeypatch.setattr(sampler_module, "SAMPLER_COMMIT_ENABLE", False)
    trajs = _sample(_dominant_direction, K=12, horizon=1)
    firsts = [t["first_action"] for t in trajs]
    assert firsts.count(2) >= 10, "flag off at call time must merge"
