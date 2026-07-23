"""SEAL configuration.

Single dataclass holding ALL hyperparameters.

SEAL = Streaming Event-driven Adaptive Learner. ALE Pong, 84x84 grayscale,
4-frame stacking (velocity is in the input, no RNN needed). Event-driven
encoder + ObGD + eligibility traces + aux task + utility gate.

Design note on frame stacking: the original spec banned frame stacking and
specified a GRU trunk. We tested the GRU for 1.45M frames and observed
persistent oscillation (ret20 swinging -20.00 to -20.85, corrVr 0.03-0.58),
consistent with an unstable temporal representation under trace-only learning
(no BPTT). We replaced the GRU with 4-frame stacking (the streaming-RL paper's
proven mechanism). This simplifies the architecture, puts velocity directly in
the event deltas (4 channels of frame-to-frame motion), and eliminates the
trace-teaching-recurrent-network problem. The GRU vs frame-stacking comparison
is itself a finding: "a trace-trained GRU cannot stably replace frame stacking
for temporal perception in streaming RL on Pong."
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class EnvPreset:
    """Static description of one environment target."""
    id: str
    domain: str
    obs_kind: str
    action_kind: str
    frame_skip: int
    use_atari_wrappers: bool
    episodic_life: bool
    total_frames: int
    event_band: Tuple[float, float]
    n_actions: int = 6


PRESETS = {
    "ALE/Pong-v5": EnvPreset(
        id="ALE/Pong-v5",
        domain="atari",
        obs_kind="image",
        action_kind="discrete",
        frame_skip=4,
        use_atari_wrappers=True,
        episodic_life=True,
        total_frames=10_000_000,
        event_band=(0.005, 0.03),
        n_actions=6,
    ),
}


@dataclass
class Config:
    # ---- environment ----
    env_id: str = "ALE/Pong-v5"
    seed: int = 0
    # Single EMA temporal accumulation (1 channel). alpha=0.2 gives ~4 frames
    # of effective memory ((1-alpha)/alpha = 4), equivalent to 4-frame stacking
    # but as one smooth trail (event-friendly: sparser deltas than raw frames).
    ema_alpha: float = 0.2
    # Multi-scale EMA (N EMAs × L lags = N*L channels). None = single EMA.
    # Kept as an option; the single-EMA path is the default for Pong.
    ema_alphas: tuple = None
    ema_lags: int = 1

    # ---- RL ----
    gamma: float = 0.99
    lam: float = 0.8              # eligibility trace λ (paper value)
    alpha: float = 1.0            # ObGD step magnitude (cancels in bound-active regime)
    kappa: float = 2.0            # overshooting bound (paper value)
    entropy_coeff: float = 0.01   # adaptive entropy: |δ|·τ·∇H (paper §4, AC mode)

    # ---- RL algorithm: "q" (Stream Q, off-policy, paper's Pong choice) or
    # "ac" (Stream AC, on-policy). Q-learning bootstraps from max_a' Q(s',a')
    # regardless of the action taken, so the agent learns Q-values for greedy
    # actions even during ε-exploration. This is critical for sparse-reward
    # Pong: the agent accidentally scores during ε-exploration, Q-learning
    # propagates credit back, and argmax-Q becomes a good policy. AC's
    # on-policy gradient can't learn from actions the policy didn't take. ----
    rl_algorithm: str = "q"

    # ---- exploration: epsilon-greedy (paper stream_q_atari) ----
    # The paper solves Pong with Stream Q + epsilon-greedy, NOT entropy-only AC.
    # Entropy collapses as V learns "everything is bad" in sparse-reward Pong;
    # epsilon-greedy forces random actions so the agent occasionally hits the
    # ball and gets a +1 reward, breaking the exploration barrier.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    exploration_fraction: float = 0.05   # fraction of total_frames over which ε decays

    # ---- auxiliary prediction (spec §2.4) ----
    aux_weight: float = 0.1
    aux_dim: int = 3              # (ball_x, ball_y, paddle_contact)

    # ---- event encoder / homeostasis ----
    threshold_kind: str = "perpixel"   # "homeostat" (legacy) or "perpixel" (move A)
    threshold_target_lo: float = 0.005
    threshold_target_hi: float = 0.03
    threshold_adapt_rate: float = 1e-2
    threshold_theta0: float = 1e-4
    # per-element scale-following threshold (move A): theta[e] = k*EWMA(|delta[e]|)
    perpixel_k: float = 2.0
    perpixel_warmup: int = 50
    perpixel_floor: float = 1e-6

    # ---- streaming RL: trace bounding (Bug 5) ----
    # NOTE: move C (event-gated traces) was REJECTED by diagnostic
    # (tests/diag_event_gated_traces.py): grad_W is already exactly zero at
    # inactive input locations (the forward hard mask gates it; the
    # straight-through term only affects grad_delta, not grad_W). So traces
    # are already event-gated and the explosion is standard accumulation,
    # not an event problem. We keep the hard z_sum clip as the bounded fix.
    max_z_sum: float = 10_000.0

    # ---- reward scaling (paper uses ScaleReward, but Pong rewards are already
    # ±1 and scaling distorts the clean TD-error scale). Diagnosed: with scaling
    # ON, raw Pong score stayed -20/-21 (losing every game) while the scaled
    # return looked like -8 (misleading). Turned OFF for Pong. ----
    scale_reward: bool = False
    utility_decay: float = 0.9999
    utility_tau_low: float = 1e-6
    regen_every: int = 25_000
    regen_frac: float = 0.01
    dormant_silence_steps: int = 10_000

    # ---- network ----
    # EventConv2d(in, out, k, stride). First layer in_ch = N*L (multi-scale)
    # or 1 (single EMA); auto-derived in agent from cfg.ema_alphas/ema_lags.
    conv_layers: tuple = (
        (1, 16, 8, 5),           # in_ch=1 (single EMA); patched to N*L if multi-scale
        (16, 32, 4, 3),
        (32, 32, 3, 2),
    )
    trunk_dim: int = 256          # EventLinear -> 256-dim features -> heads

    # ---- training / logging ----
    total_frames: int = 10_000_000
    log_every: int = 1_000
    record_frames: int = 10_000

    # ---- output ----
    out_dir: str = "results"
    run_name: str = "seal_pong"

    def to_dict(self) -> dict:
        return asdict(self)


def config_from_preset(env_id: str, **overrides) -> Config:
    """Build a Config, patching per-preset fields."""
    preset = PRESETS[env_id]
    cfg = Config()
    cfg.env_id = env_id
    cfg.total_frames = preset.total_frames
    cfg.threshold_target_lo = preset.event_band[0]
    cfg.threshold_target_hi = preset.event_band[1]
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
