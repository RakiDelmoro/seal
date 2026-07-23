"""SEAL configuration.

Single dataclass holding ALL hyperparameters.

SEAL = Streaming Event-driven Adaptive Learner. ALE Pong, 84x84 grayscale,
single EMA temporal accumulation (1 channel, ~4-frame memory). Event-driven
encoder + Stream Q (off-policy) + ObGD + eligibility traces + epsilon-greedy
exploration + aux task + utility gate.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
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
        n_actions=6,
    ),
}


@dataclass
class Config:
    # ---- environment ----
    env_id: str = "ALE/Pong-v5"
    seed: int = 0
    # Single EMA temporal accumulation (1 channel). alpha=0.2 gives ~4 frames
    # of effective memory ((1-alpha)/alpha = 4) — velocity is in the trail,
    # no RNN / no frame stacking needed.
    ema_alpha: float = 0.2

    # ---- RL (Stream Q, off-policy) ----
    gamma: float = 0.99
    lam: float = 0.8              # eligibility trace λ (paper value)
    alpha: float = 1.0            # ObGD step magnitude (cancels in bound-active regime)
    kappa: float = 2.0            # overshooting bound (paper value)

    # ---- exploration: epsilon-greedy ----
    # Stream Q bootstraps from max_a' Q(s',a') regardless of the action taken,
    # so the agent learns greedy Q-values even during epsilon-exploration. This
    # is critical for sparse-reward Pong: the agent accidentally scores during
    # exploration, Q-learning propagates credit back, argmax-Q becomes a good
    # policy. Traces reset on exploration actions (off-policy correction).
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    exploration_fraction: float = 0.05   # fraction of total_frames over which ε decays

    # ---- auxiliary prediction ----
    aux_weight: float = 0.1
    aux_dim: int = 3              # (ball_x, ball_y, paddle_contact)

    # ---- event encoder: per-element threshold (move A) ----
    # theta[e] = k * EWMA(|delta[e]|). Per-element from the first step, robust
    # to heavy tails, self-calibrating (static elements armed at the floor,
    # active elements fire at a stable tail rate). Structurally prevents dead
    # layers (Bug 4). Exactness invariant holds (theta=0 => dense).
    perpixel_k: float = 2.0
    perpixel_warmup: int = 50
    perpixel_floor: float = 1e-6

    # ---- streaming RL: trace bounding (Bug 5 safety net) ----
    # Traces are already event-gated by the forward hard mask (grad_W is exactly
    # zero at inactive input locations), so z_sum stays bounded naturally.
    # This hard clip is a safety net that rarely engages.
    max_z_sum: float = 10_000.0

    # ---- utility / plasticity ----
    utility_decay: float = 0.9999
    utility_tau_low: float = 1e-6
    regen_every: int = 25_000
    regen_frac: float = 0.01
    dormant_silence_steps: int = 10_000

    # ---- network ----
    # EventConv2d(in, out, k, stride). First layer takes 1 EMA channel.
    conv_layers: tuple = (
        (1, 16, 8, 5),
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
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
