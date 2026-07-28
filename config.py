"""SEAL configuration — e-prop LSNN on ALE Pong.

SEAL = Streaming Event-driven Adaptive Learner, now realized as a recurrent
network of spiking neurons (LSNN = LIF + ALIF) trained online by reward-based
e-prop (Bellec et al., Nature Communications 2020, Eq. 5/36) with symmetric
feedback weights (B_jk = Wout_kjᵀ, weight transport).

Single dataclass holding ALL hyperparameters. No backend flag — e-prop is the
only learning method. Atari Pong, 84x84 grayscale, ONE frame per env step
(the LSNN recurrence carries temporal context; no frame stacking).
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
    # ONE normalized 84x84 frame per env step; the recurrent LSNN core carries
    # temporal context (replaces the old 4-frame stack).
    frame_stack: int = 1

    # ---- RL (actor-critic, reward-based e-prop) ----
    gamma: float = 0.99            # discount factor γ (Rt = Σ γ^k r_{t+k})
    # Eligibility-trace λ for the ObGD readout/CNN traces (decay = γλ = 0.792,
    # the stream-x default). The LSNN e-prop tags use lam_rec below.
    lam: float = 0.8
    c_v: float = 1.0               # critic weight c_V in E = E_π + c_V·E_V
    # Policy entropy bonus, weighted by sign(δ) (stream-x form). Under ObGD's
    # normalized updates the softmax drifts toward saturation; 0.2 (plus
    # ε-greedy below) keeps behavior exploratory over long runs.
    entropy_coef: float = 0.2

    # ---- value & TD-error clipping (0 = OFF; superseded by ObGD) ----
    # OFF by default. Rationale: (1) v_clip=10 < |true Pong return| = 21 makes
    # the TD targets inconsistent — the critic can never represent the true
    # value and rams into the clip; (2) delta_clip censors exactly the
    # highest-information samples (terminal scoring events); (3) ObGD's
    # δ̄ = max(|δ|,1) normalization auto-shrinks the step when δ is large, so
    # stability no longer needs clipping (stream-x runs fully unclipped).
    # Set >0 to re-enable as a safety blanket.
    v_clip: float = 0.0            # clamp critic V to [-v_clip, +v_clip] (0=off)
    delta_clip: float = 0.0        # clamp |δ| <= delta_clip (0=off)

    # ---- spiking neuron dynamics (Eqs. 6-10, dt = 1 ms) ----
    dt_ms: float = 1.0             # simulation timestep in ms
    tau_m: float = 20.0            # LIF membrane time constant (ms) -> α = exp(-dt/τ_m)
    tau_a: float = 1000.0          # ALIF adaptation time constant (ms) -> ρ = exp(-dt/τ_a)
    tau_out: float = 20.0          # readout (leaky output neuron) time constant (ms)
    v_threshold: float = 1.0       # base firing threshold v_th (mV, normalized)
    beta: float = 0.07             # ALIF adaptation increment β (Eq. 8)
    gamma_pd: float = 0.3          # pseudo-derivative scale ψ (1/v_th · γ_pd)
    refractory_ms: float = 3.0     # spike refractory period (ms); z forced to 0

    # ---- how many ms of LSNN simulation per env step ----
    # One env step ≈ frame_skip emulator frames (Pong: 4 frames ≈ 64 ms).
    # We sub-step the LSNN at 1 ms resolution. 4 ms of LSNN sim per env step
    # is the default compute knob (paper runs many ms per frame).
    sim_ms_per_step: int = 4

    # ---- LSNN core (recurrent) ----
    n_lif: int = 240               # number of LIF neurons
    n_alif: int = 160              # number of ALIF neurons
    rec_sparsity: float = 0.0      # fraction of recurrent weights zeroed (0 = dense)
    dale: bool = False             # Dale's law (E/I split) — off by default
    win_scale: float = 0.02        # input weight init scale (kept small for sparse firing)
    wrec_scale: float = 0.01       # recurrent weight init scale
    # ALIF eligibility-trace simplification: if True, drop the ψ·β term in
    # Eq. 24 (Eq. 26 approximation). Performance indistinguishable on the
    # paper's temporal-credit-assignment task; keep False for exactness.
    alif_elig_approx: bool = False

    # ---- spiking CNN front-end (pixels -> input spike trains) ----
    # Two stride-conv layers downsample 84x84 -> a population of input neurons.
    # TRAINABLE (paper-faithful, Fig. 4b: error fed back to the spiking CNN):
    # the input-layer learning signal L_in = Winᵀ·L_j is injected at the spike
    # rates and autograd takes the local gradient through the feedforward conv
    # stack; δ-gating + γλ traces live in the ObGD optimizer. A frozen random
    # encoder is an information bottleneck that caps sample efficiency.
    train_cnn: bool = True              # train the front-end (False = frozen ablation)
    cnn_feedback: str = "symmetric"     # 'symmetric' (Winᵀ) | 'random' (paper's ba_cnn)
    conv_layers: tuple = (
        (1, 32, 8, 5),             # (in_ch, out_ch, kernel, stride): 84 -> 16
        (32, 64, 4, 3),            # 16 -> 5
    )
    n_input_neurons: int = 1600        # flattened conv output -> input spike units
    input_spike_rate: float = 0.05      # target input firing rate (per ms; ~50 Hz)
    input_gain_init: float = 1.0        # initial sigmoid gain for input encoding

    # ---- readout (leaky output neurons, Eq. 11) ----
    n_actions: int = 6
    # Actor (policy) + critic (value) heads, both leaky linear over LSNN spikes.
    # Trained by autograd + SGD (feedforward; e-prop not needed here, per Methods).

    # ---- symmetric e-prop feedback weights B_jk ----
    # B_jk = Wout_kjᵀ is read live from the readout at each step (weight
    # transport). No init scale, learning rate, or plasticity rule applies —
    # B is a live view, not a separate parameter. See model/broadcast.py.

    # ---- e-prop plasticity rule (Eq. 5/36) with ObGD step-size bound ----
    # ΔW_ji = step · δ_t · F_γλ( L_j^t · ε̄_ji^t )
    # Step sizes are overshooting-bounded (model/optim.py, model/eprop_optimizer.py):
    #   step = η / max(1, δ̄·‖tag‖₁·η·κ)   — so η is O(1), not 3e-4/√len.
    # This is the sample-efficiency fix from streaming-RL (Elsayed et al. 2024):
    # stability comes from the bound, not from tiny steps.
    eta_rec: float = 1.0              # base step size for recurrent + input weights
    eta_out: float = 1.0              # base step size for readout (ObGD)
    eta_cnn: float = 1.0              # base step size for CNN front-end (ObGD)
    # κ is THE effective step-size knob in ObGD's normalized regime (M>1):
    # update ≈ sign(δ)·e/(‖e‖₁·κ) — near-constant size, so weights random-walk
    # with drift; small weight decay bounds ||W||. Validated over 15k-frame
    # runs: paper values (3/2) oscillate on sparse-reward Pong with a linear
    # readout; κ_policy=20/κ_value=50 keeps the critic bounded near the
    # terminal reward. κ_rec must stay GENTLE: κ_rec<=0.5 blew the recurrent
    # core up (96% neurons dormant, |Win| 3x) within 15k frames — the LSNN
    # tag L1 is huge, so its normalized step is already ~1e-6 at κ_rec=2.
    kappa_rec: float = 2.0            # ObGD κ for the e-prop (Win/Wrec) update — do NOT lower
    kappa_policy: float = 20.0        # ObGD κ for the actor readout
    kappa_value: float = 100.0        # ObGD κ for the critic (100: halved V bounce vs 50)
    kappa_cnn: float = 5.0            # ObGD κ for the CNN front-end
    wd_policy: float = 1e-3           # weight decay, actor readout group
    wd_value: float = 1e-3            # weight decay, critic readout group
    wd_cnn: float = 1e-4              # weight decay, CNN front-end group
    # ε-greedy exploration (stream-x's validated Atari recipe): decouples
    # exploration from softmax sharpness — a collapsing policy entropy can
    # never kill exploration. 0 disables.
    explore_eps: float = 0.05
    # Structural entropy floor: actor logits = logit_cap·tanh(y/logit_cap), so
    # softmax can never saturate one-hot and the policy channel of L_j can
    # never die (observed failure: entropy 0.00 for 50+ straight episodes).
    logit_cap: float = 2.0            # 0 disables
    # ScaleReward wrapper (stream-x): divide rewards by the running std of the
    # discounted return trace (floor 1.0). Shrinks the return magnitudes the
    # critic must represent -> δ becomes reward-dominated, not critic-noise-
    # dominated (observed failure: V swinging ±40 on a ±21 game at 100k frames).
    scale_reward: bool = True
    lam_rec: float = 1.0              # LSNN tag-filter λ (1.0 = paper's F_γ; <1 shortens credit window)
    grad_clip: float = 0.0            # legacy hard clip on |δ·tag| (0=off; ObGD supersedes)

    # ---- stability: episode-length schedule (paper's deep-RL trick) ----
    # Increase episode length in phases: short diverse episodes early build
    # skills; long episodes later fine-tune the policy. This CURRICULUM is kept.
    # The paper's η ∝ 1/√len coupling is OFF — it was a stability substitute
    # for mechanisms we now have (ObGD bound); keeping both double-shrinks η.
    episode_schedule: tuple = (
        (0,          600),    # (start_step, max_episode_len) — see a full rally early
        (500_000,   1200),
        (1_000_000, 2000),    # full-length from here on
    )
    eta_length_scale: bool = False   # legacy: scale η by 1/sqrt(len) — superseded by ObGD

    # ---- plasticity (dormant spiking-unit regeneration) ----
    # Observed failure at 100k frames: dormant_frac climbing monotonically to
    # 0.28 — death outran resurrection (4 neurons / 25k steps). Tuned to
    # ~20 neurons / 10k steps.
    regen_every: int = 10_000
    dormant_silence_ms: float = 10_000.0  # no spike for this many ms = dormant
    regen_frac: float = 0.05

    # ---- training / logging ----
    total_frames: int = 10_000_000
    log_every: int = 1_000
    record_frames: int = 10_000
    warmup_frames: int = 1_000       # random-action warmup (normalize stats, no learn)

    # ---- output ----
    out_dir: str = "results"
    run_name: str = "seal_eprop_pong"

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- derived decay factors (computed once, not stored) ----
    @property
    def alpha(self) -> float:
        """LIF membrane decay α = exp(-dt/τ_m)."""
        import math
        return math.exp(-self.dt_ms / self.tau_m)

    @property
    def rho(self) -> float:
        """ALIF adaptation decay ρ = exp(-dt/τ_a)."""
        import math
        return math.exp(-self.dt_ms / self.tau_a)

    @property
    def kappa(self) -> float:
        """Readout leak κ = exp(-dt/τ_out)."""
        import math
        return math.exp(-self.dt_ms / self.tau_out)

    @property
    def refractory_steps(self) -> int:
        return max(1, int(round(self.refractory_ms / self.dt_ms)))


def config_from_preset(env_id: str, **overrides) -> Config:
    """Build a Config, patching per-preset fields."""
    preset = PRESETS[env_id]
    cfg = Config()
    cfg.env_id = env_id
    cfg.total_frames = preset.total_frames
    cfg.n_actions = preset.n_actions
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
