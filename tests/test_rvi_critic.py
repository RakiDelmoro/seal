"""Average-reward (RVI) critic validation — mechanism-level tests.

The RVI critic subtracts the agent's own running reward rate ρ from every
reward, so the TD error is δ = (r − ρ) + V(s') − V(s). These tests defend the
observable contracts of that change:

  1. ρ tracks the long-run reward rate of the stream it sees.
  2. The quiet-frame contrast: with ρ < 0, a zero-reward frame is a small
     POSITIVE TD error ("I survived") — the centreing the planner needs.
  3. ρ is updated from REAL rewards only: update(rho_update=False) and all
     imagined updates leave ρ untouched (no model-to-itself feedback).
  4. Checkpoint round-trip preserves ρ.
  5. RVI off: the discounted path is byte-identical to the original rule.
"""
import numpy as np
import pytest

import core.value as value_module
from core.value import Value
from utils.checkpoint import save_checkpoint, load_checkpoint


def _make_state(seed: int, block: int = 100) -> np.ndarray:
    s = np.zeros(1296, dtype=np.float32)
    s[block:block + 16] = 0.25
    return s


# ── 1. ρ tracks the reward rate ────────────────────────────────────

def test_rho_tracks_constant_reward():
    """A constant reward r is learned exactly: ρ → r."""
    v = Value()
    s1, s2 = _make_state(1, 100), _make_state(2, 200)
    for _ in range(3000):
        v.update(s1, -1.0, s2, done=False, gamma=0.99, lam=0.95, eta=1e-4)
    assert abs(v.rho - (-1.0)) < 1e-3, f"ρ should track -1.0, got {v.rho}"


def test_rho_tracks_sparse_loss_rate():
    """One loss every 10 frames → ρ → -0.1 (the Pong regime, ~-0.106)."""
    v = Value()
    s1, s2 = _make_state(1, 100), _make_state(2, 200)
    for i in range(10000):
        r = -1.0 if i % 10 == 9 else 0.0
        v.update(s1, r, s2, done=False, gamma=0.99, lam=0.95, eta=1e-4)
    assert abs(v.rho - (-0.1)) < 5e-3, f"ρ should track -0.1, got {v.rho}"


# ── 2. Quiet frames become positive once ρ is negative ─────────────

def test_quiet_frame_is_positive_td_error():
    """With w=0 (V constant) and ρ<0, a 0-reward frame gives δ = -ρ > 0:
    'surviving this frame is better than my usual pace'."""
    v = Value()
    v.w.fill(0.0)
    v.rho = -0.1
    s1, s2 = _make_state(1, 100), _make_state(2, 200)
    delta_quiet = v.update(s1, 0.0, s2, done=False, gamma=0.99, lam=0.95,
                           eta=0.0, rho_update=False)
    delta_loss = v.update(s1, -1.0, s2, done=False, gamma=0.99, lam=0.95,
                          eta=0.0, rho_update=False)
    assert delta_quiet == pytest.approx(0.1, abs=1e-6)
    assert delta_loss == pytest.approx(-0.9, abs=1e-6)
    assert delta_quiet > 0 > delta_loss


# ── 3. ρ only moves on real rewards ────────────────────────────────

def test_rho_update_false_leaves_rho_untouched():
    v = Value()
    s1, s2 = _make_state(1, 100), _make_state(2, 200)
    v.update(s1, 0.5, s2, done=False, gamma=0.99, lam=0.95, eta=1e-4)
    rho_before = v.rho
    for _ in range(100):
        v.update(s1, 0.5, s2, done=False, gamma=0.99, lam=0.95, eta=1e-4,
                 rho_update=False)
    assert v.rho == rho_before, "rho_update=False must not touch ρ"


def test_imagined_updates_never_move_rho():
    """update_imagined consumes ρ but never updates it."""
    v = Value()
    s1, s2 = _make_state(1, 100), _make_state(2, 200)
    v.update(s1, -1.0, s2, done=False, gamma=0.99, lam=0.95, eta=1e-4)
    rho_before = v.rho
    for _ in range(100):
        v.update_imagined(s1, 5.0, s2, eta=1e-4, gamma=0.99)
    assert v.rho == rho_before, "imagined updates must not touch ρ"


# ── 4. Checkpoint round-trip preserves ρ ───────────────────────────

def test_checkpoint_roundtrips_rho(tmp_path):
    from core.seal_core import SEALCore
    core = SEALCore()
    core.value.rho = -0.106
    path = str(tmp_path / "rvi.npz")
    save_checkpoint(core, path)
    core2, _ = load_checkpoint(path)
    assert core2.value.rho == pytest.approx(-0.106)


# ── 5. RVI off → original discounted rule ──────────────────────────

def test_rvi_off_matches_discounted_rule(monkeypatch):
    monkeypatch.setattr(value_module, "RVI_ENABLE", False)
    v = Value()
    v.w.fill(0.0)
    v.rho = -0.1  # must be ignored when RVI is off
    s1, s2 = _make_state(1, 100), _make_state(2, 200)
    # With w=0: δ_discounted = r + γ·0 − 0 = r (done=False, V(s')=0 too).
    delta = v.update(s1, -1.0, s2, done=False, gamma=0.99, lam=0.95, eta=0.0)
    assert delta == pytest.approx(-1.0, abs=1e-6)
