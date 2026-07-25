"""SEAL configuration.

Single dataclass holding ALL hyperparameters.

SEAL = Streaming Event-driven Adaptive Learner. ALE Pong, 84x84 grayscale,
4-frame stacking (velocity is in the input, no RNN). Event-driven encoder +
Stream Q (off-policy) + AdaptiveObGD + eligibility traces + epsilon-greedy
exploration + aux task + utility gate.

Matches the streaming-RL paper's proven Atari architecture (4-frame stack,
32→64→64 convs, 256 trunk) with our event-driven encoder layered on top.
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
    # The paper's fix for trace explosion — z_sum stays O(n_params), no ceiling.
    beta2: float = 0.999
    eps: float = 1e-8

    # ---- exploration: epsilon-greedy ----
    # Stream Q bootstraps from max_a' Q(s',a') regardless of the action taken,
    # so the agent learns greedy Q-values even during epsilon-exploration.
    # Traces reset on exploration actions (off-policy correction).
    #
    # Fixed linear schedule (the streaming-RL paper's value). An adaptive
    # corrVr-driven ε was tried and validated in closed-loop: it created a
    # feedback loop (low corrVr -> ε stays high -> overestimation -> corrVr
    # stays low) that made divergence MORE likely, not less. Reverted to the
    # proven fixed schedule. corrVr is still computed and logged as a
    # value-learning diagnostic; it just no longer drives ε.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    exploration_fraction: float = 0.05   # fraction of total_frames over which ε decays

    # ---- Q-learning overestimation guard ----
    # Q-learning's max_a' Q(s',a') bootstrap is the deadly-triad offender
    # (function-approx + bootstrap + off-policy → runaway optimistic Q). We
    # clip the bootstrap to [-q_clip, +q_clip] as a belt-and-suspenders guard
    # on top of AdaptiveObGD's κ-bound. Pong returns are ±21, so a value
    # estimate outside ±q_clip is physically impossible.
    q_clip: float = 21.0

    # ---- auxiliary prediction: game-agnostic GVF bank (model/gvf.py) ----
    # Each GVF is a linear TD(λ) prediction of a discounted future cumulant
    # derived only from the event mask + reward (no game-specific knowledge).
    # Coefficient applied to every GVF's shaping term in the encoder loss.
    gvf_weight: float = 0.1

    # ---- SwiftTD on the heads (Javed et al. RLC 2024) ----
    swift_theta: float = 1e-3       # IDBD meta step-size θ
    swift_eta: float = 0.1          # max correction ratio η (overshoot bound)
    swift_eps_decay: float = 0.99   # step-size decay factor ε
    swift_alpha_init: float = 1e-7  # initial per-feature step size (α_init)
    swift_eta_min: float = 1e-15    # min step size (η_min); β clipped to [ln η_min, ln η]
    swift_gamma: float = 0.99       # discount for SwiftTD traces (== gamma)

    # ---- event encoder: per-element threshold (move A) ----
    # theta[e] = k * EWMA(|delta[e]|). Per-element from the first step, robust
    # to heavy tails, self-calibrating. Structurally prevents dead layers (Bug 4).
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
    # EventConv2d(in, out, k, stride). First layer takes 4 stacked frames.
    # 32→64→64 convs (paper capacity) + 256 trunk.
    conv_layers: tuple = (
        (4, 32, 8, 5),
        (32, 64, 4, 3),
        (64, 64, 3, 2),
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
