# SEAL — Sampling from cognitive maps Enables Adaptive Learning

<p align="center">
  <strong>A brain-inspired agent that plans by imagining trajectories on a learned cognitive map, then acts from accumulated reactive habits — all through local synaptic plasticity, no backpropagation.</strong>
</p>

---

## What is SEAL?

SEAL is a reinforcement learning agent for visual sequential decision tasks (currently: Atari Pong), built from a synthesis of two ideas:

1. **The GCML** — *Generative Cognitive Map Learner* (Lin, Yang, Zhao, Pezzulo & Maass, *Nature Machine Intelligence* 2026): learns a cognitive map of the world through local plasticity, then *imagines* goal-directed trajectories on that map to plan. No deep learning, no backprop, no replay — just online delta rules.

2. **The OaK architecture** — *Options and Knowledge* (Sutton, RLC 2025): a vision for general intelligence centered on a learned **policy** and **value function** that accumulate reactive competence over time.

SEAL takes the GCML's cognitive-map-and-imagination engine and extends it with OaK's policy and value function, plus two domain-specific additions for Pong (autonomous dynamics and sparse-delayed-reward credit assignment). The result: an agent that can both **plan** (imagine 40 futures, rank by value + geometry) and **react** (learned habits), learning everything online with biologically plausible local plasticity.

### The name

**S**ampling from cognitive maps **E**nables **A**daptive **L**earning.

The core mechanism is *sampling* — stochastic trajectory generation on a learned cognitive map — and the promise is *adaptive learning* — continuous, online, never-retrain adjustment to changing conditions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 FIXED PERCEPTION (no learning)                    │
│                                                                  │
│  Frame ──┐                                                       │
│          ├─ 2-channel → frozen CNN (2 stride-conv layers)        │
│  |Δframe|┘   Conv1: 2→16, 8×8, stride 4 → 20×20                  │
│              Conv2: 16→16, 4×4, stride 2 → 9×9                   │
│              flatten (py, px, channel)              ▼            │
│                                          State s ∈ ℝ^1296         │
│  (9×9 grid × 16 ch; each position = contiguous 16-dim block)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            LEARNED CORE (local plasticity, online)               │
│                                                                  │
│  A (banded 1296×1296, ±160) s → next s    "how the world moves"  │
│  B (1296×3)                 action → Δs   "what actions do"      │
│  D (3×1296)                 Δs → action   "inverse model" (W)    │
│  V (1×1296)                 s → return    "how good is here"     │
│  π (3×1296)                 s → action    "reactive habits"      │
│  r̂ (1×1296)                 s → reward    "reward predictor"     │
│  F (frozen all-ones)        s → feasibility                      │
│                                                                  │
│  All updates: ΔW = η·error·inputᵀ / ‖input‖² (normalized LMS /   │
│  TD(λ) — no backprop, no replay, no GPU). No bias b.             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            ACTION SELECTION (policy + imagination)               │
│                                                                  │
│  Gate 1: ε-random (adaptive) ──→ random action (exploration)     │
│  Gate 2: π confident?        ──→ policy action (System 1, fast)  │
│           (20% force-imagination override keeps teaching π)      │
│  Gate 3: goal s* exists?     ──→ imagination (System 2)          │
│           Sample 40 noisy rollouts (horizon 5):                  │
│             Δ = s* − ŝ (goal direction)                          │
│             u = D·Δ    (inverse model)                           │
│             ε ~ N(0, σ²) (diversity)                             │
│             ŝ = A·ŝ + B·a (predict, norm-renormalized)           │
│             score = α·mean V(ŝ_t) + (1−α)·(−‖ŝ_H − s*‖₁)        │
│                     − danger penalty                             │
│           pick best (or top-5) → imitate into π                  │
│           else ──→ random action (no goal yet)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    Execute action → observe reward
                             │
                             ▼
              Online learning (all components, every frame)
              + Imagined TD: rehearse K=2 short futures (horizon 3)
                on the world model; train V via predicted reward r̂
                (streaming Dyna — no replay buffer, O(1) memory)
```

### The two systems

**System 1 — Policy (fast, reactive):** `π · s` maps state to action directly. No simulation, no planning. Learns by imitating imagination's chosen action each frame and by a per-step actor-critic update (reinforce the taken action by the TD(λ) error). Accumulates competence over thousands of frames.

**System 2 — Imagination (slow, planning):** samples 40 noisy trajectories (horizon 5) on the cognitive map and scores each by a **blend of the learned value function V and geometric distance to the goal s***. The blend weight `α` starts at 0 (pure geometry, since V is random) and grows toward 0.5 as V gains signal. The inverse model D points each rollout toward s*. Noise generates diversity; the blended score ranks trajectories.

The policy learns *from* imagination's successes, so it gets better over time. Even when the policy is confident, imagination runs 20% of the time to keep providing fresh imitation signal.

### The goal s*

s* is not learned — it is **read from experience**. Preference order:
1. **Pre-score states** — when the agent scores (+1), the states from the preceding ~12 frames are saved as *proven* goal states. The most recent one becomes s*. The +1 is used here as a *goal label* ("this state led to scoring → aim there"), not as a learning signal.
2. **Geometric proxy (cold start)** — before any +1, s* is the recent state whose ball is furthest on the opponent's side of the 9×9 grid (highest energy peak in the rightmost columns).

This makes imagination goal-directed from frame 1 without any reward-driven weight update on the planning path.

### Action selection: three gates, no phases

SEAL learns and acts from frame 1 — there is no explore/play split. The imagination engine has three internal gates that smoothly transition behavior as the agent learns:

```
Gate 1: ε-random? ──yes──→ random action (exploration)
          │ no
          ▼
Gate 2: π confident? ──yes──→ policy action (System 1, fast)
          │ no                  (20% force-imagination override)
          ▼
Gate 3: goal s* exists? ──yes──→ imagination (System 2), 40 rollouts
          │ no
          ▼
        random action (no goal yet)
```

- **Gate 1 (ε-random):** exploration prevents getting stuck. ε adapts from a Laplace-smoothed success rate — losing → explore more, winning → trust the policy (down to a 5% floor, never zero).
- **Gate 2 (policy fast path):** when π's max softmax probability exceeds the confidence threshold, use π directly. A 20% force-imagination override keeps imitation signal flowing.
- **Gate 3 (goal):** imagination needs a goal s* to aim at. If none is available yet (too few recent states, no visible ball), fall back to a random action. When V has no signal yet, the scorer sets α=0 so ranking is purely geometric — imagination still runs and still teaches π.

---

## Key design decisions

### Fixed perception, learned readouts

The visual frontend is a frozen random CNN (2-layer stride-conv, Kaiming init, never trained) fed with `(frame, |Δframe|)`. It contains **zero Pong knowledge** — it is a generic fixed nonlinear lifting (edges → combinations → walls/paddles). Only the linear readouts (A, B, D, V, π, r̂) are trained. This is the paper's "linear readouts from fixed nonlinear circuits" principle, justified by the random-features / kernel trick. Conv runs single-threaded (`torch.set_num_threads(1)`) — multi-threaded conv/BLAS thrashes when several CPU training processes run side by side.

### Locality-ordered state

The CNN output is flattened `(py, px, channel)`: each spatial position maps to a *contiguous 16-dim block* of state dims. This is what makes the banded dynamics matrix A able to represent motion — a ball shifting one grid cell shifts activation by exactly one block:
- horizontal shift: ±16 dims
- vertical shift: ±144 dims (9 positions × 16 ch)
- diagonal: ±160 dims

A's half-band of ±160 covers all three.

### Banded dynamics matrix A

A is 1296×1296 but only ±160 entries per row are learnable (K=321/row, ~416K params, not 1.68M). This is SEAL's extension beyond the GCML — the paper's domains don't have autonomous dynamics (the ball in Pong moves on its own). The banded structure keeps the delta rule's signal-to-noise viable. Safeguards:

- **No bias b** — a constant drift term grows into an attractor that collapses multi-step rollouts (all imagined futures drift to the same state). Autonomous motion is captured by A's shift structure instead.
- **Spectral clip** — the diagonal is kept in [0.95, 1.0] and A is rescaled whenever its estimated operator norm exceeds 1.0, so 5-step rollouts don't diverge.
- **Dense cache** — for batched imagination the band is materialized once per frame into a dense matrix (vectorized scatter) and reused for all 200 rollout steps via BLAS.

### Streaming TD(λ) credit assignment (not replay)

Pong's ±1 rewards come 20+ frames after the causing actions. V learns by streaming TD(λ): an accumulating eligibility trace (λ=0.95, clipped magnitude) propagates each TD error backward over recently visited states. Two anti-drift nets keep it stable on sparse rewards: an overshooting-bounded step size (each update corrects at most 25% of the TD error) and weight decay. Everything is O(1) per step — no replay buffer, no episode buffers.

### Average-reward (RVI) critic — the fix for "everything equally bad"

The discounted critic above answers "how many future losses remain?" — and
because the score is not part of the CNN state, the honest answer is the same
constant in almost every state, so V converges to a flat negative (`v_norm`
climbs forever; imagination's `score_std` collapses toward zero). RVI mode
(on by default, `RVI_ENABLE`) subtracts the agent's own running reward rate ρ
from every reward:

    δ = (r − ρ) + V(s′) − V(s)        (no γ — Pong is a continuing task)

ρ is one scalar updated online from real rewards only (imagined TD must never
feed ρ — the model cannot talk to itself about the reward rate). Quiet frames
become small positives ("I survived — better than my pace"), loss frames sharp
negatives, and V settles into a per-state landscape instead of a constant.
Nothing is hardcoded: ρ is the agent's own discovered pace, and the landscape
is learned purely from where the ±1s actually land. Reference: Yu, Wan &
Sutton 2025, average-reward RL via relative value iteration
(arXiv:2512.06218). Toggle for A/B: `train.py --rvi on|off`.

### Successor-feature value V_sf — "where does this state lead?"

Even centred (RVI), the myopic critic V(s)=w·s can't separate states: the raw
reward carries no per-state preference information. The successor-features
decomposition (Dayan 1993; Barreto et al. 2017) says value under reward r̂ is
V^π(s; w_R) = ψ^π(s)·w_R — future state-visitation dotted with the learned
reward weights. SEAL learns the composition directly: `core/successor.py`
runs streaming average-reward TD(λ) on the REWARD-PREDICTOR stream r̂(s_{t+1})
instead of the raw reward. r̂ is queried every frame, so credit propagates
densely and V_sf learns a forward-looking landscape. On Pong the dominant
reward-predicting region is LOSS (~21 losses per win), so V_sf mainly encodes
avoidance — rollouts heading toward imminent losses score low. Avoidance is
discriminative, which is exactly what imagination's ranking needs.

Wiring: imagination scores rollouts with `core.scorer_value()` (V_sf when
`SF_ENABLE`, else V). V_sf has its own ρ (the r̂-stream mean — a different
stream from the environment reward) and its own trace. The main critic keeps
driving the actor. Default OFF — A/B with `train.py --sf on|off`.

### Bootstrap trajectory scoring — "arrive OR be valued"

GCML scores rollouts by absolute distance to the goal (`−‖ŝ_H − s*‖`). That
assumes the rollout can ARRIVE. On Pong it cannot: the goal is ~150 units
away (a ~100-frame-old state, stale) while a 5-step rollout walks ~3 —
measured: the best of 40 rollouts got closer to the goal in 0 of 43 windows,
while doing nothing got closer 35% of the time. All plans tie on distance,
"best" is noise.

The standard fix across model-based RL — Dreamer (arXiv:1912.01603: short
horizons need value bootstrapping), MuZero (leaves scored by the value
network), TD-MPC (5-step rollouts + learned terminal Q), classical MPC
(terminal cost) — grades the short rollout by predicted reward along the way
plus LEARNED VALUE at the endpoint:

    score = Σ_t γᵗ r̂(ŝ_t) + γᴴ · V_term(ŝ_H) − danger_penalty · 𝟙[danger]

`V_term` is `core.scorer_value()` (V_sf when SF is on). The rollout no longer
needs to reach the ghost-goal; s* still steers rollout direction via D, but
the grade comes from value — making the effective planning horizon infinite
without lengthening rollouts. The scorer is vectorized (one matmul each for
trip rewards and terminal values). Learning gate: bootstrap scoring activates
only once the terminal value's norm has grown 5% past init — before that,
imagination falls back to pure geometric scoring (GCML's original form).
Default **OFF after a negative 120k A/B** (0.34% vs 0.65% win rate): the
bootstrap terms carry ~30× less variance than the geometric distance term —
the 40 rollouts end only 0.31 apart in a ~10-norm state space, so value
scores tie and top-5 sampling collapses. The geometric blend is noisy too,
but its large distance numbers keep action diversity alive. Fix rollout
DIVERSITY first, then revisit. A/B with `train.py --bootstrap on|off`.

### Normalized LMS

All delta rules divide by `‖input‖²`, making the effective learning rate independent of input magnitude (typical Pong states have `‖s‖² ≈ 140`).

### Imagined TD (streaming Dyna, no replay)

After each real transition, SEAL also rehearses: it rolls out K=2 short
futures (horizon 3) from the new state using the learned dynamics (A, B),
steps them with the current policy π, asks a learned reward predictor
`r̂(s) = w_R·s` how much reward each imagined step would bring, and gives the
critic a one-step TD update at each imagined step. Nothing is stored —
rollouts are generated fresh from the current weights and consumed
immediately: O(1) memory, no replay buffer, no off-policy correction. This
propagates value information through the world model, spreading credit
beyond the eligibility-trace horizon and making V more predictive for
imagination's ranking. Two safeguards: each imagined update's learning rate
is scaled by a confidence κ that decays with depth (model error compounds),
and imagined transitions use λ=0 so the real eligibility trace is never
contaminated with synthetic states. Measured cost: ~+4% frame time after the
dynamics vectorization; toggle via `IMAGINED_TD_ENABLE` (config) or
`train.py --imagined-td on|off` (A/B runs). A from-memory variant rehearses
from proven-good pre-score states (`IMAGINED_TD_FROM_MEMORY_ENABLE`, off by
default).

---

## File structure

```
seal/
├── perception/              # Fixed visual frontend (no learning)
│   ├── frozen_cnn.py        #   Frozen random 2-layer stride-conv (torch, 1 thread)
│   └── pipeline.py          #   frame + |Δframe| → 1296-dim locality-ordered state
│
├── core/                    # Learned readouts (all local rules, no backprop)
│   ├── dynamics.py          #   A: banded autonomous dynamics (no bias b),
│   │                        #      vectorized build/update + dense BLAS cache
│   ├── action_effect.py     #   B: action → state change
│   ├── direction.py         #   D: inverse model (paper's W, eq 13)
│   ├── value.py             #   V: streaming TD(λ) critic; average-reward (RVI)
│   │                        #      mode with running reward rate ρ (+ update_imagined)
│   ├── successor.py         #   V_sf: successor-feature value — TD(λ) on the r̂
│   │                        #      stream ("where does this state lead?")
│   ├── policy.py            #   π: reactive policy (imitation + actor-critic)
│   ├── reward_model.py      #   r̂: reward predictor for imagined TD
│   ├── gate.py              #   F: frozen feasibility gate (all-ones for Pong)
│   └── seal_core.py         #   Combines all + per-step online updates (step_learn)
│
├── imagination/             # Goal-directed trajectory sampling (System 2)
│   ├── geometric_goal.py    #   s* selection (pre-score memory / geometric proxy)
│   │                        #   + geometric rollout scoring + danger detection
│   ├── evaluator.py         #   Blend α·V + (1−α)·geometry + danger penalty;
│   │                        #      BootstrapScorer: Σ γᵗr̂ + γᴴV_term − danger
│   ├── sampler.py           #   40 noisy rollouts (batched, horizon 5)
│   ├── engine.py            #   Three gates: ε / policy fast path / imagination
│   └── imagined_td.py       #   Streaming Dyna: rehearsed futures train V
│
├── env/                     # Environment wrappers
│   ├── pong_wrapper.py      #   Pong: raw ±1 rewards, 84×84 grayscale
│   ├── envs_atari.py        #   Vendored Atari wrappers (D7)
│   └── norm_wrappers.py     #   Streaming Welford normalization
│
├── training/                # Training utilities
│   └── success_tracker.py   #   Adaptive ε from Laplace-smoothed success rate
│
├── utils/                   # Checkpointing + logging
│   ├── checkpoint.py        #   Save/load all weights (A, B, D, V, π, r̂)
│   └── metrics.py           #   Per-episode CSV logger
│
├── tests/                   # Integration tests
│   └── test_slice2_imagination.py  # CNN, sampler, evaluator, engine, real-Pong
│                                   # smoke test, checkpoint round-trip
│
├── config.py                # All hyperparameters (single source of truth)
├── train.py                 # Main entry: learn + act from frame 1 (single phase)
├── test.py                  # Evaluation: greedy play, no learning
├── validate_gcml.py         # Sanity checks: A, B, D, V, π + real-Pong rollout
├── validate_imagined_td.py  # Sanity checks for imagined TD (6 tests)
└── analyze_ab.py            # Compare A/B imagined-TD training logs + plot
```

---

## Getting started

### Prerequisites

```
numpy, gymnasium, ale-py, torch, pytest
matplotlib  (only for analyze_ab.py plots)
```

If you run several training processes side by side, pin each to one core:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. The code itself
pins torch to 1 thread at import (see `perception/frozen_cnn.py`).

### Run training (headless, no GUI)

The standard way to train — runs as fast as possible with no display, logs
per-episode metrics to a CSV, and saves checkpoints periodically.

```bash
# Short run (episode-count mode)
python train.py --episodes 100 --seed 0

# Long run (frame-budget mode, with checkpointing + logging)
python train.py --frame-budget 10000000 --seed 0 \
  --log-path results/seal.csv \
  --checkpoint-interval 500000 --checkpoint-dir results

# Resume training from a checkpoint (keeps the learned world model + direction)
python train.py --resume results/seal_final_7006k.npz \
  --frame-budget 10000000 --seed 999 \
  --log-path results/seal.csv \
  --checkpoint-interval 500000 --checkpoint-dir results
```

Checkpoints are saved to `results/seal_<frames//1000>k.npz` (e.g.
`seal_300k.npz` after 300,000 frames) and the latest as
`results/seal_final_<frames//1000>k.npz`.

### Monitor progress (headless)

```bash
tail -f results/seal.csv
```

Key columns to watch:
- `scored`, `lost` — game score
- `epsilon` — exploration rate (drops as the success rate rises → the flywheel)
- `pred_err_avg` — world-model prediction error (should trend down)
- `d_norm` — direction model norm (stays bounded, no divergence)
- `rollout_norm_ratio` — 5-step rollout shrinkage (A-only; the sampler
  renormalizes, so this is a diagnostic, not the planning norm)
- `score_std` — spread of the 40 imagination scores (high = diverse plans)
- `rho` — the critic's running reward-rate estimate (≈ −0.106 on Pong, i.e.
  one loss every ~9.4 frames; constant 0 when RVI is off)
- `td_delta_avg` — should sit near 0 once ρ has centred the reward stream
- `sf_norm`, `sf_rho` — successor-feature value norm and its ρ (r̂-stream
  mean; meaningful only when SF is on)

### Run the imagined-TD A/B experiment

```bash
# Two matched runs (same seed, same budget), ITD on vs off
python train.py --frame-budget 300000 --seed 0 --imagined-td on \
  --log-path results/ab_with_itd.csv
python train.py --frame-budget 300000 --seed 0 --imagined-td off \
  --log-path results/ab_no_itd.csv

# Compare: learning curves, critic/world-model health, head-to-head + plot
python analyze_ab.py
```

### Run the RVI critic A/B experiment

```bash
# Two matched runs (same seed, same budget), average-reward critic on vs off
python train.py --frame-budget 300000 --seed 0 --rvi on \
  --log-path results/ab_rvi_on.csv
python train.py --frame-budget 300000 --seed 0 --rvi off \
  --log-path results/ab_rvi_off.csv
```

What to watch: `rho` converges to ≈ −0.106; `td_delta_avg` re-centres near 0;
`v_norm` stops its linear climb; `score_std` (imagination contrast) recovers.
The win rate (`scored`) is the real test.

### Run the successor-feature (V_sf) A/B experiment

```bash
python train.py --frame-budget 300000 --seed 0 --sf on \
  --log-path results/ab_sf_on.csv
python train.py --frame-budget 300000 --seed 0 --sf off \
  --log-path results/ab_sf_off.csv
```

What to watch: `sf_rho` converges to the r̂-stream mean; `score_std` should
stay higher with SF on (V_sf discriminates states that flat V cannot).

### Run the bootstrap scoring A/B experiment

```bash
python train.py --frame-budget 300000 --seed 0 --sf on --bootstrap on \
  --log-path results/ab_bs_on.csv
python train.py --frame-budget 300000 --seed 0 --sf on --bootstrap off \
  --log-path results/ab_bs_off.csv
```

What to watch: once V_sf grows past init (gate opens), `score_std` should be
higher with bootstrap on — plans are ranked by value instead of tied on
distance. Win rate is the real test.

### Run tests

```bash
python -m pytest tests/test_slice2_imagination.py -v          # integration tests
python -m pytest tests/test_slice2_imagination.py -k "not smoke" -q  # fast unit tests only
```

### Evaluate a trained model (headless)

```bash
# From a checkpoint
python test.py --checkpoint results/seal_final_7006k.npz --eval-episodes 20

# Or quick train-then-eval
python test.py --train-episodes 50 --eval-episodes 20 --seed 0
```

---

## The two intellectual roots

| GCML (Nature Machine Intelligence, 2026) | OaK (Sutton, RLC 2025) | SEAL |
|---|---|---|
| Forward model V | Transition models | A, B |
| Inverse model W | — | D |
| Imagination (stochastic sampling) | Plan with transition models | Imagination engine |
| Local plasticity, online | Continual learning | All delta rules |
| — | Learn a policy | π |
| — | Learn a value function | V (TD(λ)) |
| — | Background/imagined experience (Dyna) | Imagined TD (streaming, no buffer) |
| — | Meta-learned step sizes | (future work) |
| — | Options (temporal abstractions) | (future work) |

### References

- H. Lin, Y. Yang, R. Zhao, G. Pezzulo, W. Maass. *Neural sampling from cognitive maps enables goal-directed imagination and planning.* Nature Machine Intelligence **8**, 1045–1065 (2026). https://www.nature.com/articles/s42256-026-01254-4
- R. Sutton. *The OaK Architecture: A Vision of SuperIntelligence from Experience.* RLC 2025. https://www.amii.ca/videos/oak-architecture-rich-sutton-rlc2025
- The Alberta Plan for AI Research: https://www.incompleteideas.net/AlbertaPlan.pdf

---

## What SEAL guarantees (regardless of Pong performance)

- A complete intelligent agent — perception, cognitive map, planning, policy, value, continual learning — built from **local plasticity alone**
- **No backpropagation**, no replay buffer, no GPU required (runs on CPU at ~27 fps single-threaded)
- **Online learning**: every frame, every component, never stops
- **Instant adaptation**: change the goal s* and imagination immediately aims there; no retraining
- **Biologically plausible**: every update is a delta rule a synapse could compute
- **Neuromorphically implementable**: suitable for in-memory computing at low power

---

## License

Research project. Not currently licensed for redistribution.
