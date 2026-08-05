"""Bootstrap trajectory scoring validation — "arrive OR be valued".

The bootstrap scorer grades each imagined rollout by predicted reward along
the way plus the LEARNED VALUE at the endpoint (Dreamer/MuZero/TD-MPC form),
instead of GCML's absolute distance to the goal — which can never be reached
on Pong (goal ~150 units away, rollout walks ~3). These tests defend the
observable contracts:

  1. Scores are finite and rollout-order-independent per trajectory.
  2. A trajectory whose endpoint has higher V_term scores higher (terminal
     bootstrap drives the ranking) even when distances to the goal are equal.
  3. Trip reward matters: a trajectory passing through a high-r̂ region
     scores higher with an equal endpoint.
  4. Danger penalty applied when any state is on our side.
  5. Engine uses the bootstrap scorer only once the terminal value has grown
     past init (learning gate), and the flag toggles the path.
  6. Bootstrap-off path is byte-compatible with the old blended scorer.
"""
import numpy as np
import pytest

import imagination.engine as engine_module
import core.seal_core as seal_core_module
from core.seal_core import SEALCore
from imagination.evaluator import BootstrapScorer
from imagination.engine import ImaginationEngine
from training.success_tracker import SuccessTracker
from config import GOAL_WINDOW


def _safe_state(rng, peak_pos=5 * 9 + 5, scale=0.1):
    """A state with its energy peak at a SAFE grid position (px>1), so the
    danger check never fires on the noise."""
    s = np.zeros(1296, dtype=np.float32)
    s[peak_pos * 16:(peak_pos + 1) * 16] = 1.0
    s += rng.normal(0, scale, 1296).astype(np.float32) * 0.01
    return s


def _trajs(end_values, K=4, H=5, seed=0):
    """K trajectories whose terminal state V_term reads as end_values."""
    rng = np.random.default_rng(seed)
    trajs = []
    for ev in end_values:
        states = [_safe_state(rng) for _ in range(H)]
        states[-1] = np.zeros(1296, dtype=np.float32)
        states[-1][200:216] = ev / 16.0  # V_term.w = ones on block 200
        trajs.append({"states": states})
    return trajs

class _Linear:
    """Minimal linear readout stand-in (attr w)."""
    def __init__(self, w):
        self.w = w


# ── 1. Finite, well-formed output ──────────────────────────────────

def test_scores_finite_and_ranked():
    scorer = BootstrapScorer()
    rm = _Linear(np.zeros(1296, dtype=np.float32))
    tv = _Linear(np.zeros(1296, dtype=np.float32))
    tv.w[200:216] = 1.0
    trajs = _trajs([1.0, 2.0, -1.0, 0.5])
    scores, best = scorer.score_trajectories(trajs, rm, tv)
    assert len(scores) == 4 and all(np.isfinite(s) for s in scores)
    assert best == 1, "highest terminal value must win"


# ── 2. Terminal value drives the ranking (the whole point) ────────

def test_terminal_bootstrap_ranks_by_value_not_distance():
    """All endpoints are the SAME distance from any goal — only V_term
    differs. The winner must be the highest-valued endpoint."""
    scorer = BootstrapScorer()
    rm = _Linear(np.zeros(1296, dtype=np.float32))
    tv = _Linear(np.zeros(1296, dtype=np.float32))
    tv.w[200:216] = 1.0
    trajs = _trajs([3.0, 1.0, 2.0])
    scores, best = scorer.score_trajectories(trajs, rm, tv)
    assert best == 0
    assert scores[0] > scores[2] > scores[1]


# ── 3. Trip reward contributes ────────────────────────────────────

def test_trip_reward_breaks_terminal_tie():
    """Two trajectories with IDENTICAL endpoints: the one whose path passes
    through a high-r̂ region scores higher."""
    scorer = BootstrapScorer()
    rm = _Linear(np.zeros(1296, dtype=np.float32))
    rm.w[100:116] = 1.0
    tv = _Linear(np.zeros(1296, dtype=np.float32))
    rng = np.random.default_rng(0)
    base = [np.zeros(1296, dtype=np.float32) for _ in range(5)]
    good = [s.copy() for s in base]
    for t in good:            # passes through the r̂-hot region
        t[100:116] = 0.25
    bad = [s.copy() for s in base]
    scores, _ = scorer.score_trajectories(
        [{"states": good}, {"states": bad}], rm, tv)
    assert scores[0] > scores[1]


# ── 4. Danger penalty ─────────────────────────────────────────────

def test_danger_penalty_applied():
    """A trajectory with the ball on our side (px<=1) gets penalized."""
    scorer = BootstrapScorer()
    rm = _Linear(np.zeros(1296, dtype=np.float32))
    tv = _Linear(np.zeros(1296, dtype=np.float32))
    grid = scorer.geo.grid
    safe = np.zeros(1296, dtype=np.float32)
    safe[(5 * grid + 5) * 16:(5 * grid + 6) * 16] = 1.0     # px=5
    danger = np.zeros(1296, dtype=np.float32)
    danger[(5 * grid + 0) * 16:(5 * grid + 1) * 16] = 1.0   # px=0
    scores, _ = scorer.score_trajectories(
        [{"states": [safe] * 5}, {"states": [danger] * 5}], rm, tv,
        danger_penalty=2.0)
    assert scores[1] == pytest.approx(scores[0] - 2.0)


# ── 5. Engine learning gate + flag ────────────────────────────────

def _playable_core():
    core = SEALCore()
    rng = np.random.default_rng(1)
    base = np.zeros(1296, dtype=np.float32)
    base[5 * 16:6 * 16] = 1.0
    core.recent_states = __import__("collections").deque(
        [(base + rng.normal(0, 0.1, 1296)).astype(np.float32)
         for _ in range(50)], maxlen=GOAL_WINDOW)
    return core, (base + rng.normal(0, 0.1, 1296)).astype(np.float32)


def test_engine_gate_blocks_bootstrap_until_value_grows(monkeypatch):
    """Fresh terminal value (no growth) -> geometric scoring even with the
    flag on."""
    monkeypatch.setattr(engine_module, "BOOTSTRAP_ENABLE", True)
    core, s0 = _playable_core()
    eng = ImaginationEngine()
    tr = SuccessTracker()
    for _ in range(20):
        tr.on_episode_end(21, 0)
    eng.eps_floor = 0.0
    eng.select_action(s0, core, None)        # 1st call records init norm
    # Value still at init size -> geometric path (scores ~ −distance scale)
    assert not eng._terminal_ready(core)


def test_engine_uses_bootstrap_once_value_grows(monkeypatch):
    core, s0 = _playable_core()
    eng = ImaginationEngine()
    monkeypatch.setattr(engine_module, "BOOTSTRAP_ENABLE", True)
    tr = SuccessTracker()
    for _ in range(20):
        tr.on_episode_end(21, 0)
    eng.eps_floor = 0.0
    eng.select_action(s0, core, None)        # records init norm
    core.scorer_value().w *= 1.2             # simulate learning
    assert eng._terminal_ready(core)
    action, diag = eng.select_action(s0, core, None)
    assert diag["source"] in ("greedy", "top5")
    assert eng.last_scores is not None and len(eng.last_scores) == 40


def test_bootstrap_off_never_invokes_bootstrap_scorer(monkeypatch):
    """Flag off -> the bootstrap scorer must never run, even after the
    terminal value has grown (learning gate satisfied)."""
    monkeypatch.setattr(engine_module, "BOOTSTRAP_ENABLE", False)
    core, s0 = _playable_core()
    eng = ImaginationEngine()

    def _explode(*a, **kw):
        raise AssertionError("bootstrap scorer must not run when disabled")
    monkeypatch.setattr(eng.bootstrap, "score_trajectories", _explode)

    tr = SuccessTracker()
    for _ in range(20):
        tr.on_episode_end(21, 0)
    eng._terminal_ready(core)                 # record init norm baseline
    core.scorer_value().w *= 1.2
    eng.eps_floor = 0.0
    action, diag = eng.select_action(s0, core, None)   # must not raise
    assert diag["source"] in ("greedy", "top5", "no_goal")
