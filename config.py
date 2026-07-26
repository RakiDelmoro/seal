"""SEAL configuration.

Single dataclass holding ALL hyperparameters.

SEAL = Streaming Event-driven Adaptive Learner. ALE Pong, 84x84 grayscale,
4-frame stacking (velocity is in the input, no RNN). Event-driven encoder +
Stream Q (off-policy) + AdaptiveObGD + eligibility traces + epsilon-greedy
exploration + SPR auxiliary representation learning + utility gate.

Matches the streaming-RL paper's proven Atari architecture (4-frame stack,
32→64→64 convs, 256 trunk) with our event-driven encoder layered on top.
Auxiliary representation learning follows "Squeezing More from the Stream"
(arXiv:2602.09396): SPR self-prediction with a momentum target network +
orthogonal gradient projection to prevent the aux loss from conflicting
with the RL update.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict


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
    # 4-frame stacking (paper's proven Atari temporal input). Velocity is in
    # the 4 stacked frames — no RNN / no EMA needed. Each frame is a separate
    # channel, giving the conv exact positions to difference.
    frame_stack: int = 4

    # ---- RL (Stream Q, off-policy) ----
    gamma: float = 0.99
    lam: float = 0.8              # eligibility trace λ (paper value)
    alpha: float = 1.0            # ObGD step magnitude (cancels in bound-active regime)
    kappa: float = 2.0            # overshooting bound (paper value)
    # AdaptiveObGD second-moment normalization (Adam-style, paper verbatim):
    # per-param v[p] = β2·v + (1-β2)·(δ·trace)²; trace normalized by sqrt(v_hat).
    beta2: float = 0.999
    eps: float = 1e-8

    # ---- exploration: epsilon-greedy (fixed linear schedule) ----
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    exploration_fraction: float = 0.05   # fraction of total_frames over which ε decays

    # ---- Q-learning overestimation guard ----
    # Q-learning's max_a' Q(s',a') bootstrap is the deadly-triad offender.
    # We clip the bootstrap to ±q_clip as a guard on top of AdaptiveObGD's κ-bound.
    q_clip: float = 21.0

    # ---- SPR auxiliary representation learning (arXiv:2602.09396) ----
    # Self-Predictive Representations: the online encoder predicts its own
    # future latent state (K steps ahead) from the current latent + actions,
    # matched against a stop-gradient momentum (EMA) target encoder. The aux
    # gradient is orthogonalized against the Q gradient so it only shapes the
    # encoder in directions that don't conflict with the RL update, and
    # norm-bounded so it can't destabilize the trunk.
    spr_weight: float = 1.0         # SPR loss coefficient (λ_SPR in the paper)
    spr_horizon: int = 3            # K: predict z_{t+1..t+K} from z_t + actions
    spr_tau: float = 0.01           # EMA weight for the target encoder (θ' ← (1-τ)θ + τθ')
    spr_proj_dim: int = 256         # projection head output dim
    spr_lr: float = 1e-3            # bounded LR for the orthogonalized SPR SGD step
    spr_grad_clip: float = 1.0      # max norm of the SPR gradient before orthogonalization

    # ---- event encoder: per-element threshold ----
    perpixel_k: float = 2.0
    perpixel_warmup: int = 50
    perpixel_floor: float = 1e-6

    # ---- utility / plasticity ----
    utility_decay: float = 0.9999
    utility_tau_low: float = 1e-6
    regen_every: int = 25_000
    regen_frac: float = 0.01
    dormant_silence_steps: int = 10_000

    # ---- network (matches paper's Atari architecture) ----
    conv_layers: tuple = (
        (4, 32, 8, 5),
        (32, 64, 4, 3),
        (64, 64, 3, 2),
    )
    trunk_dim: int = 256

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
