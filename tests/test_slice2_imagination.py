"""SEAL integration tests — frozen CNN + geometric imagination architecture.

Tests:
  1. The frozen CNN produces a locality-ordered state of the right shape.
  2. The trajectory sampler produces correctly-structured rollouts.
  3. The geometric evaluator picks the trajectory closest to the goal s*.
  4. The imagination engine runs imagination when a geometric goal exists.
  5. A short real-Pong training run completes and stays bounded.
  6. Checkpoints round-trip weights.
"""
import numpy as np
import pytest

from config import N_STATE, CNN_GRID, CNN_CHANNELS
from imagination.engine import ImaginationEngine
from imagination.sampler import sample_trajectories
from imagination.evaluator import evaluate_trajectories
from imagination.geometric_goal import GeometricGoal
from perception.pipeline import PerceptionPipeline
from core.seal_core import SEALCore
from training.success_tracker import SuccessTracker


# ───────────────────────── fast unit tests ──────────────────────────

def test_cnn_produces_locality_ordered_state():
    """The frozen CNN outputs a state of the right shape."""
    pipe = PerceptionPipeline()
    frame = np.random.rand(84, 84).astype(np.float32)
    s, _ = pipe.forward(frame)
    assert s.shape == (N_STATE,)
    assert np.linalg.norm(s) > 0


def test_trajectory_sampling_shapes():
    """Sampler produces K rollouts with first_action + predicted `states`."""
    core = SEALCore()
    s = np.random.randn(N_STATE).astype(np.float32)
    s_star = np.random.randn(N_STATE).astype(np.float32)
    trajs = sample_trajectories(
        s, s_star,
        core.dynamics, core.action_effect, core.direction, core.gate,
        n_trajectories=5, horizon=3,
    )
    assert len(trajs) == 5
    for t in trajs:
        assert t["first_action"] in [0, 1, 2]
        assert len(t["states"]) == 3
        assert t["states"][0].shape == (N_STATE,)


def test_evaluator_picks_best():
    """Geometric evaluator picks the trajectory closest to the goal s*."""
    s_star = np.zeros(N_STATE, dtype=np.float32)
    s_star[:5] = 1.0

    def mk(dist):
        st = s_star.copy(); st[0] += dist
        return {"first_action": 0, "states": [st]}

    trajs = [mk(3.0), mk(0.5), mk(9.0)]
    scores, best_idx = evaluate_trajectories(trajs, s_star=s_star)
    assert best_idx == 1
    assert scores[1] > scores[0] and scores[1] > scores[2]


def test_engine_imagination_runs_when_goal_exists():
    """With a geometric goal available, imagination dominates."""
    core = SEALCore(); eng = ImaginationEngine(); tr = SuccessTracker()
    for _ in range(20):
        tr.on_episode_end(21, 0)
    geo = GeometricGoal()
    s_opp = np.zeros(N_STATE, dtype=np.float32)
    # px=7 (opponent side), py=4 → position p = 4*9 + 7 = 43
    a, b = geo.ranges[43]
    s_opp[a:b] = 3.0
    for _ in range(15):
        core.recent_states.append(s_opp.copy())
    srcs = []
    for _ in range(20):
        act, diag = eng.select_action(s_opp, core, tr)
        srcs.append(diag["source"])
        assert act in [0, 1, 2]
    imag = sum(1 for s in srcs if s in ("greedy", "top5"))
    assert imag > 10, f"imagination should dominate: {srcs}"


def test_engine_random_when_no_goal():
    """With no goal-eligible state, the engine falls back to random."""
    core = SEALCore(); eng = ImaginationEngine(); tr = SuccessTracker()
    for _ in range(20):
        tr.on_episode_end(21, 0)
    s_empty = np.zeros(N_STATE, dtype=np.float32)
    for _ in range(15):
        core.recent_states.append(s_empty.copy())
    act, diag = eng.select_action(s_empty, core, tr)
    assert diag["source"] == "no_goal"


def test_no_learned_value_or_policy():
    """The core has no value function or policy — only A, B, D."""
    core = SEALCore()
    assert not hasattr(core, "value")
    assert not hasattr(core, "policy")
    assert hasattr(core, "dynamics")
    assert hasattr(core, "action_effect")
    assert hasattr(core, "direction")


# ───────────────────────── real-Pong smoke test ─────────────────────

@pytest.fixture(scope="module")
def trained_core():
    """Run a few real-Pong episodes through the actual train.train."""
    from train import train
    res = train(n_episodes=5, seed=0, verbose=False)
    return res["core"]


def test_smoke_learns_and_stays_bounded(trained_core):
    """After a short real-Pong run, D is bounded and A stable."""
    diag = trained_core.diagnostics()
    print(f"\n  After 5 episodes: d_norm={diag['d_norm']:.3f}, "
          f"a_op_norm={diag['a_op_norm']:.4f}, steps={diag['step_count']}")
    assert diag["d_norm"] < 10.0
    assert 0.9 < diag["a_op_norm"] < 1.1


def test_checkpoint_roundtrips_weights():
    """Checkpoint save/load persists A, B, D."""
    import tempfile, os
    from utils.checkpoint import save_checkpoint, load_checkpoint
    core = SEALCore()
    core.step_count = 42
    tmp = tempfile.mktemp(suffix=".npz")
    save_checkpoint(core, tmp, {"episodes": 5})
    core2, meta = load_checkpoint(tmp)
    assert int(meta["episodes"]) == 5
    assert core2.step_count == 42
    assert np.allclose(core2.dynamics.A_band, core.dynamics.A_band)
    os.remove(tmp)
