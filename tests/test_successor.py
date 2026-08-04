"""Successor-feature value (V_sf) validation — mechanism-level tests.

V_sf learns ψ(s)·w_r̂ ("where does this state lead?") by running streaming
TD(λ) on the reward-predictor stream r̂ instead of the raw reward. These
tests defend its observable contracts:

  1. Default off: scorer_value() is the main critic and step_learn leaves
     the successor untouched.
  2. Enabled: scorer_value() routes to V_sf and step_learn trains it.
  3. Average-reward semantics on the AUXILIARY stream: ρ_sf tracks r̂, not
     the environment reward.
  4. Isolation: V_sf updates never touch the main critic (weights, trace, ρ).
  5. Checkpoint round-trip preserves V_sf weights and ρ_sf.
  6. Signal gating: the engine's V-signal detector measures V_sf when SF is
     on and recovers contrast (high std) from a discriminative V_sf.
"""
import numpy as np
import pytest

import core.seal_core as seal_core_module
from core.seal_core import SEALCore
from core.successor import SuccessorValue
from utils.checkpoint import save_checkpoint, load_checkpoint


def _make_state(seed: int, block: int = 100) -> np.ndarray:
    s = np.zeros(1296, dtype=np.float32)
    s[block:block + 16] = 0.25
    return s


@pytest.fixture
def sf_on(monkeypatch):
    monkeypatch.setattr(seal_core_module, "SF_ENABLE", True)
    yield
    # monkeypatch restores the module attribute automatically


# ── 1. Default off ─────────────────────────────────────────────────

def test_sf_off_by_default():
    core = SEALCore()
    assert core.scorer_value() is core.value, (
        "SF off: imagination must score with the main critic V")
    s1, s2 = _make_state(1, 100), _make_state(2, 300)
    w_before = core.successor.w.copy()
    core.step_learn(s1, 1, s2, 0.0, False, source="epsilon")
    assert np.array_equal(core.successor.w, w_before), (
        "SF off: successor weights must not be updated")


# ── 2. Enabled: routing + learning ─────────────────────────────────

def test_sf_on_routes_to_successor(sf_on):
    core = SEALCore()
    assert core.scorer_value() is core.successor


def test_sf_on_step_learn_trains_successor(sf_on):
    core = SEALCore()
    s1, s2 = _make_state(1, 100), _make_state(2, 300)
    w_before = core.successor.w.copy()
    for _ in range(5):
        m = core.step_learn(s1, 1, s2, 0.0, False, source="epsilon")
    assert not np.array_equal(core.successor.w, w_before), (
        "SF on: step_learn must update successor weights")
    assert np.isfinite(m["sf_delta"])

def test_rho_sf_tracks_auxiliary_stream_not_env_reward(sf_on, monkeypatch):
    """ρ_sf is the mean of the r̂ stream, not the environment reward.

    Pin r̂ to a fixed constant (learning disabled) and feed env reward +1
    every step: ρ_sf must converge to the r̂ constant while the main
    critic's ρ converges to +1 — the two streams are separate."""
    from core.reward_model import RewardModel
    monkeypatch.setattr(RewardModel, "update",
                        lambda self, s, reward, eta: 0.0)
    core = SEALCore()
    # r̂(s2) = 0.5 for the arrival state (block 300, 0.25 per dim).
    core.reward_model.w.fill(0.0)
    core.reward_model.w[300:316] = 0.125   # 16 dims × 0.25 × 0.125 = 0.5
    s1, s2 = _make_state(1, 100), _make_state(2, 300)
    for _ in range(2000):
        core.step_learn(s1, 1, s2, 1.0, False, source="epsilon")
    assert abs(core.successor.rho - 0.5) < 0.01, (
        f"ρ_sf should track the r̂ stream (0.5), got {core.successor.rho}")
    assert abs(core.value.rho - 1.0) < 0.01, (
        f"main critic ρ should track env reward (+1), got {core.value.rho}")


def test_rho_sf_follows_reward_model():
    """ρ_sf converges to the mean of whatever r̂ says."""
    sf = SuccessorValue()
    s1, s2 = _make_state(1, 100), _make_state(2, 300)
    for _ in range(3000):
        sf.update(s1, 0.5, s2, done=False, gamma=0.99, lam=0.95, eta=0.0)
    assert abs(sf.rho - 0.5) < 1e-3, f"ρ_sf should track 0.5, got {sf.rho}"


# ── 4. Isolation from the main critic ──────────────────────────────

def test_successor_never_touches_main_critic(sf_on):
    core = SEALCore()
    s1, s2 = _make_state(1, 100), _make_state(2, 300)
    # Prime the main critic's trace and ρ with a real update.
    core.step_learn(s1, 1, s2, -1.0, False, source="epsilon")
    trace_before = core.value.e.copy()
    rho_before = core.value.rho
    # Successor-only updates (same transition, zero env reward).
    for _ in range(10):
        core.successor.update(s1, 0.25, s2, done=False, gamma=0.99,
                              lam=0.95, eta=1e-4)
    assert np.array_equal(core.value.e, trace_before)
    assert core.value.rho == rho_before


# ── 5. Checkpoint round-trip ───────────────────────────────────────

def test_checkpoint_roundtrips_successor(tmp_path, sf_on):
    core = SEALCore()
    core.successor.w[0] = 1.234
    core.successor.rho = -0.042
    path = str(tmp_path / "sf.npz")
    save_checkpoint(core, path)
    core2, _ = load_checkpoint(path)
    assert core2.successor.w[0] == pytest.approx(1.234, abs=1e-6)
    assert core2.successor.rho == pytest.approx(-0.042)


# ── 6. Signal gating measures V_sf when enabled ────────────────────

def test_signal_gate_uses_successor(sf_on):
    """Engine's V-signal detector reads V_sf: a discriminative V_sf
    (different values on different blocks) yields std above threshold."""
    from imagination.engine import ImaginationEngine
    core = SEALCore()
    eng = ImaginationEngine()
    # Make V_sf strongly state-dependent: weight on block 100 only.
    core.successor.w.fill(0.0)
    core.successor.w[100:116] = 1.0
    # ≥20 samples (gate requirement): half the states activate the weighted
    # block (V_sf = 4.0), half don't (V_sf = 0.0) → high std.
    for i in range(24):
        block = 100 if i % 2 == 0 else 500
        eng._recent_values.append(core.scorer_value().forward(
            _make_state(i, block)))
    assert eng._v_has_signal(core), (
        "discriminative V_sf must register as 'has signal'")
