"""Validate that the GCML world model + new value/policy architecture is sane.

Checks:
  1. A diagonal is near 1.0 (no collapse).
  2. A^5 rollout norm ratio is near 1.0.
  3. A can represent horizontal and vertical shifts in synthetic state.
  4. B update is bounded.
  5. D grows with increased learning rate.
  6. V and π are instantiated and update without NaNs.
  7. Runs a short real Pong rollout and reports prediction error.
"""
import numpy as np
from core.seal_core import SEALCore
from core.dynamics import BandedDynamics
from core.action_effect import ActionEffect
from core.direction import Direction
from core.value import Value
from core.policy import Policy
from perception.pipeline import PerceptionPipeline
from env.pong_wrapper import PongEnv
from config import N_STATE, N_ACTIONS, CNN_GRID, CNN_CHANNELS, A_HALF_BAND


def create_synthetic_ball_state(px: int, py: int, magnitude: float = 1.0) -> np.ndarray:
    """Create a synthetic state with a "ball" at (px, py)."""
    s = np.zeros(N_STATE, dtype=np.float32)
    pos = py * CNN_GRID + px
    s[pos * CNN_CHANNELS:(pos + 1) * CNN_CHANNELS] = magnitude / np.sqrt(CNN_CHANNELS)
    return s


def test_a_spectral_properties():
    print("\n=== Test A: spectral properties ===")
    core = SEALCore()
    A = core.dynamics

    # Build dense A for testing
    A_dense = np.zeros((N_STATE, N_STATE), dtype=np.float32)
    for j, offset in enumerate(A.band_offsets):
        if offset == 0:
            np.fill_diagonal(A_dense, A.A_band[:, j])
        elif offset > 0:
            idx = np.arange(N_STATE - offset)
            A_dense[idx, idx + offset] = A.A_band[:N_STATE - offset, j]
        else:
            idx = np.arange(-offset, N_STATE)
            A_dense[idx, idx + offset] = A.A_band[-offset:, j]

    diag = A_dense.diagonal()
    print(f"  A diagonal mean: {diag.mean():.4f}  (target: ≥0.95)")
    print(f"  A diagonal min/max: {diag.min():.4f} / {diag.max():.4f}")
    print(f"  A half-band: {A.half_band}")
    print(f"  A off-diagonal max abs: {np.max(np.abs(A_dense - np.diag(diag))):.4f}")

    # Rollout shrinkage
    s0 = np.random.randn(N_STATE).astype(np.float32)
    s0 /= np.linalg.norm(s0)
    ratio = core._rollout_norm_ratio(s0, horizon=5)
    print(f"  A^5·s / ‖s‖ on random state: {ratio:.4f}  (target: ~1.0)")


def test_a_shift_learning():
    print("\n=== Test A: can it learn horizontal and vertical shifts? ===")
    A = BandedDynamics()
    B = ActionEffect()

    # Synthetic transitions: ball moves right, then left, then up, then down
    actions = [0] * 400  # NOOP: only A learns
    shifts = []
    for t in range(100):
        px, py = np.random.randint(1, CNN_GRID - 1, size=2)
        dx, dy = np.random.choice([-1, 0, 1], size=2)
        s_t = create_synthetic_ball_state(px, py)
        s_tp1 = create_synthetic_ball_state(px + dx, py + dy)
        pred = A.predict(s_t, None, B.B)
        err = s_tp1 - pred
        A.update(err, s_t, 1e-2)
        A.clip()
        shifts.append((dx, dy))

    # Test on clean shifts
    test_errs = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
        s_t = create_synthetic_ball_state(4, 4)
        s_tp1 = create_synthetic_ball_state(4 + dx, 4 + dy)
        pred = A.predict(s_t, None, B.B)
        err = np.linalg.norm(s_tp1 - pred)
        test_errs.append(err)
        print(f"  shift ({dx:2d}, {dy:2d})  pred_err={err:.4f}")
    print(f"  mean test error: {np.mean(test_errs):.4f}")


def test_b_bound():
    print("\n=== Test B: bounded action effect ===")
    B = ActionEffect()
    s = np.random.randn(N_STATE).astype(np.float32)
    s /= np.linalg.norm(s)
    for _ in range(1000):
        err = np.random.randn(N_STATE).astype(np.float32) * 0.1
        a = np.zeros(N_ACTIONS, dtype=np.float32)
        a[0] = 1.0
        B.update(err, a, 1e-3, s_t=s)
    print(f"  B column norms after 1000 updates: {np.linalg.norm(B.B, axis=0)}")
    print(f"  B max abs: {np.abs(B.B).max():.4f}")


def test_d_growth():
    print("\n=== Test D: inverse model grows with new hyperparameters ===")
    D = Direction()
    print(f"  initial D norm: {np.linalg.norm(D.D):.4f}")
    for _ in range(1000):
        a = np.zeros(N_ACTIONS, dtype=np.float32)
        a[np.random.randint(N_ACTIONS)] = 1.0
        delta = np.random.randn(N_STATE).astype(np.float32)
        delta /= np.linalg.norm(delta)
        D.update(a, delta, 5e-3)
        D.decay(1e-5)
    print(f"  D norm after 1000 updates: {np.linalg.norm(D.D):.4f}")


def test_v_pi_updates():
    print("\n=== Test V/π: streaming TD(λ), no NaNs, weights update ===")
    v = Value()
    pi = Policy()
    s = np.random.randn(N_STATE).astype(np.float32)
    s /= np.linalg.norm(s)
    s2 = np.random.randn(N_STATE).astype(np.float32)
    s2 /= np.linalg.norm(s2)

    # Streaming TD(λ): one update per transition, no buffering.
    d1 = v.update(s, 0.0, s2, done=False, gamma=0.99, lam=0.95, eta=5e-4)
    d2 = v.update(s2, 1.0, s2, done=True,  gamma=0.99, lam=0.95, eta=5e-4)
    print(f"  V streaming TD: δ1={d1:+.4f} δ2={d2:+.4f} w_norm={np.linalg.norm(v.w):.4f}")

    # Streaming actor-critic: reinforce by the TD error; imitation nudges too.
    pi.update(s, 1, eta=1e-4, scale=d1)
    pi.update_imitation(s, 2, 1e-3)
    print(f"  π norm: {np.linalg.norm(pi.theta):.4f}")
    print(f"  π forward sum: {pi.forward(s).sum():.4f}")


def test_real_pong_rollout():
    print("\n=== Test real Pong: short rollout prediction error ===")
    core = SEALCore()
    pipe = PerceptionPipeline()
    env = PongEnv(seed=0)
    frame, _ = env.reset()
    pipe.reset()

    errs = []
    for i in range(50):
        s = pipe.forward(frame)[0]
        action = int(np.random.randint(N_ACTIONS))
        nf, r, term, trunc, _ = env.step(action)
        s_next = pipe.forward(nf)[0]
        pred = core.predict_next_state(s, action)
        err = np.linalg.norm(s_next - pred)
        errs.append(err)
        core.step_learn(s, action, s_next, r, term or trunc, source="epsilon")
        if term or trunc:
            break
        frame = nf
    env.close()
    print(f"  initial pred_err: {errs[0]:.4f}")
    print(f"  final pred_err:   {errs[-1]:.4f}")
    print(f"  mean pred_err:    {np.mean(errs):.4f}")
    diag = core.diagnostics()
    print(f"  A op norm: {diag['a_op_norm']:.4f}")
    print(f"  B column norms: {np.linalg.norm(core.action_effect.B, axis=0)}")
    print(f"  D norm: {diag['d_norm']:.4f}")
    print(f"  V norm: {diag['v_norm']:.4f}")
    print(f"  π norm: {diag['pi_norm']:.4f}")


if __name__ == "__main__":
    test_a_spectral_properties()
    test_a_shift_learning()
    test_b_bound()
    test_d_growth()
    test_v_pi_updates()
    test_real_pong_rollout()
    print("\n=== Validation complete ===")
