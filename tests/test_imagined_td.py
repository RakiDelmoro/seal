"""Imagined-TD validation — the mechanism-level tests for streaming Dyna.

Reconstructed as permanent pytest tests (the old standalone
validate_imagined_td.py was deleted during a cleanup). Each test defends one
observable contract of the imagined-TD feature:

  1. The reward model r̂ actually learns (normalized-LMS rule converges on
     presented states; disjoint regions stay untouched).
  2. The core mechanism: value flows backward through imagination —
     V(s0) rises purely from rehearsing imagined futures whose predicted
     reward is positive.
  3. Safety: imagined updates never touch the real eligibility trace
     (byte-identical before/after).
  4. Checkpoint round-trip preserves the reward model weights.
  5. Short real-Pong rollout: no NaNs anywhere, all ITD metrics finite.
  6. From-memory rehearsal: empty memory is a safe no-op; a populated
     memory with a rewarded state raises that state's value.
"""
import numpy as np
import pytest

from config import ETA_R, N_STATE, N_ACTIONS
from core.seal_core import SEALCore
from core.reward_model import RewardModel
from imagination.imagined_td import imagined_td, imagined_td_from_memory
from utils.checkpoint import save_checkpoint, load_checkpoint


def _make_state(seed: int, block: int = 100) -> np.ndarray:
    """A bounded synthetic state with activation in one 16-dim block."""
    s = np.zeros(N_STATE, dtype=np.float32)
    s[block:block + 16] = 0.25
    return s


# ── 1. Reward model learns ─────────────────────────────────────────

def test_reward_model_learns():
    """r̂'s normalized-LMS rule converges: presented states become predicted
    exactly, and disjoint state regions are left untouched (locality)."""
    rm = RewardModel()
    s0 = _make_state(0, block=100)          # rewarded state
    s_far = _make_state(0, block=600)       # disjoint block — never rewarded

    err0 = abs(5.0 - rm.forward(s0))
    far_before = rm.forward(s_far)
    for _ in range(200):
        rm.update(s0, 5.0, 0.5)

    assert abs(5.0 - rm.forward(s0)) < 1e-3 * err0, "r̂ did not converge on s0"
    assert np.isfinite(rm.w).all()
    # Learning is linear and local: the ONLY force on a never-rewarded,
    # non-overlapping state is the global weight decay (no learning signal
    # leaks there). Its prediction must equal pure decay within float32
    # rounding — a real leak would be ~1e-2 scale, four orders above this.
    from config import R_WEIGHT_DECAY
    decay_factor = (1.0 - R_WEIGHT_DECAY) ** 200
    assert abs(rm.forward(s_far) - far_before * decay_factor) < 1e-6


# ── 2. Core mechanism: value flows through imagination ─────────────

def test_imagined_td_propagates_value_backward():
    """V(s0) rises purely from imagining futures with positive predicted reward."""
    core = SEALCore()
    s0 = _make_state(0)

    # Teach r̂ that arriving at s0 is rewarded. A starts near-identity, so
    # imagined rollouts stay close to s0 and r̂ predicts positive reward there.
    for _ in range(300):
        core.reward_model.update(s0, 5.0, ETA_R)
    assert core.reward_model.forward(s0) > 1.0, "r̂ failed to learn the reward"

    v_before = core.value.forward(s0)
    m = imagined_td(core, s0, K=1, horizon=3, explore=0.0, eta=1e-3,
                    rng=np.random.default_rng(2))
    v_after = core.value.forward(s0)

    assert m["n_updates"] == 3
    assert np.isfinite(v_after)
    assert v_after > v_before, (
        f"imagined rehearsal did not raise value: {v_before:+.6f} -> {v_after:+.6f}")


# ── 3. Safety: real eligibility trace untouched ────────────────────

def test_imagined_updates_do_not_touch_real_trace():
    """Imagined TD uses update_imagined (λ=0); the real trace stays intact."""
    core = SEALCore()
    s0, s1 = _make_state(0), _make_state(1, block=200)
    # Put real experience into the trace first (so it is non-trivially non-zero).
    core.value.update(s0, 0.0, s1, False, gamma=0.99, lam=0.95, eta=1e-4)
    e_before = core.value.e.copy()

    imagined_td(core, s0, K=2, horizon=3, rng=np.random.default_rng(3))

    assert np.array_equal(core.value.e, e_before), "real trace was modified!"


# ── 4. Checkpoint round-trip preserves r̂ ───────────────────────────

def test_checkpoint_roundtrips_reward_model():
    """R_w survives save/load alongside the other learned weights."""
    import tempfile, os
    core = SEALCore()
    core.reward_model.w[:] = np.random.default_rng(4).normal(0, 0.1, N_STATE)
    tmp = tempfile.mktemp(suffix=".npz")
    save_checkpoint(core, tmp)
    core2, _ = load_checkpoint(tmp)
    assert np.allclose(core2.reward_model.w, core.reward_model.w)
    os.remove(tmp)


# ── 5. Real Pong: no NaNs, ITD metrics finite ──────────────────────

def test_real_pong_rollout_stays_finite():
    """60 real frames through step_learn (ITD on): everything stays finite."""
    from env.pong_wrapper import PongEnv
    from perception.pipeline import PerceptionPipeline

    core = SEALCore()
    env = PongEnv(seed=0)
    pipe = PerceptionPipeline()
    frame, _ = env.reset()
    pipe.reset()

    for i in range(60):
        s = pipe.forward(frame)
        action = int(np.random.randint(N_ACTIONS))
        nf, r, term, trunc, _ = env.step(action)
        s_next = pipe.forward(nf)
        m = core.step_learn(s, action, s_next, r, term or trunc, source="epsilon")
        for key in ("imagined_delta_avg", "r_hat_avg", "td_delta", "pred_err_norm"):
            assert np.isfinite(m[key]), f"{key} not finite at frame {i}: {m[key]}"
        if term or trunc:
            frame, _ = env.reset()
            pipe.reset()
            core.reset_episode()
            continue
        frame = nf
    env.close()


# ── 6. From-memory rehearsal ────────────────────────────────────────

def test_from_memory_empty_is_safe_noop():
    """Cold start: empty pre-score memory → zero updates, no crash."""
    core = SEALCore()
    w_before = core.reward_model.w.copy()
    v_before = core.value.w.copy()
    m = imagined_td_from_memory(core, rng=np.random.default_rng(5))
    assert m == {"n_updates": 0, "imagined_delta_avg": 0.0, "r_hat_avg": 0.0}
    assert np.array_equal(core.reward_model.w, w_before)
    assert np.array_equal(core.value.w, v_before)


def test_from_memory_rehearsal_raises_value_of_good_state():
    """Rehearsing from a proven-good state raises its value (r̂ positive there)."""
    core = SEALCore()
    s_mem = _make_state(6, block=400)
    # Simulate a +1: this state is in the pre-score memory, and r̂ learned
    # that arriving here is rewarded.
    core.pre_score_states.append(s_mem.copy())
    for _ in range(300):
        core.reward_model.update(s_mem, 5.0, ETA_R)

    v_before = core.value.forward(s_mem)
    m = imagined_td_from_memory(core, K=1, horizon=3, explore=0.0, eta=1e-3,
                                rng=np.random.default_rng(6))
    v_after = core.value.forward(s_mem)

    assert m["n_updates"] == 3
    assert m["r_hat_avg"] > 0.5, "memory rollout should reach the rewarded region"
    assert v_after > v_before, (
        f"rehearsal did not raise value: {v_before:+.6f} -> {v_after:+.6f}")
