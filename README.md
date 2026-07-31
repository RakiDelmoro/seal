# SEAL — Sampling from cognitive maps Enables Adaptive Learning

<p align="center">
  <strong>A brain-inspired agent that plans by imagining trajectories on a learned cognitive map, then acts from accumulated reactive habits — all through local synaptic plasticity, no backpropagation.</strong>
</p>

---

## What is SEAL?

SEAL is a reinforcement learning agent for visual sequential decision tasks (currently: Atari Pong), built from a synthesis of two ideas:

1. **The GCML** — *Generative Cognitive Map Learner* (Lin, Yang, Zhao, Pezzulo & Maass, *Nature Machine Intelligence* 2026): learns a cognitive map of the world through local plasticity, then *imagines* goal-directed trajectories on that map to plan. No deep learning, no backprop, no replay — just online delta rules.

2. **The OaK architecture** — *Options and Knowledge* (Sutton, RLC 2025): a vision for general intelligence centered on a learned **policy** and **value function** that accumulate reactive competence over time.

SEAL takes the GCML's cognitive-map-and-imagination engine and extends it with OaK's policy and value function, plus two domain-specific additions for Pong (autonomous dynamics and sparse-delayed-reward credit assignment). The result: an agent that can both **plan** (imagine 40 futures, rank by value) and **react** (learned habits), learning everything online with biologically plausible local plasticity.

### The name

**S**ampling from cognitive maps **E**nables **A**daptive **L**earning.

The core mechanism is *sampling* — stochastic trajectory generation on a learned cognitive map — and the promise is *adaptive learning* — continuous, online, never-retrain adjustment to changing conditions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FIXED PERCEPTION (no learning)                 │
│                    General-purpose visual frontend                │
│                                                                  │
│  Frame → DoG → Gabor(spatial, 4×3×16) ──→ 192 features ──┐      │
│         → |Δframe| → Gabor(motion, 4×2×64) → 512 features ─┤      │
│                                                             │      │
│  E: fixed random block-sparse projection 704 → 1000         │      │
│  (locality-ordered: adjacent positions → adjacent state dims)│      │
│                                                             ▼      │
│                                              State s ∈ ℝ^1000       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              LEARNED CORE (local plasticity, online)              │
│                                                                  │
│  A (banded 1000×1000)  s → next state     "how the world moves"  │
│  B (1000×3)            action → Δs        "what actions do"      │
│  b (1000)              bias               "constant drift"       │
│  D (3×1000)            Δs → action        "inverse model" (W)    │
│  V (1×1000)            s → return         "how good is here"     │
│  π (3×1000)            s → action prefs   "reactive habits"      │
│  F (frozen)            s → feasibility    "what's possible"      │
│                                                                  │
│  All updates: ΔW = η · error · input^T / ‖input‖²  (normalized   │
│  LMS / Hebbian / TD(λ) — no backprop, no replay, no GPU)         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ACTION SELECTION (policy + imagination)             │
│                                                                  │
│  ε-random (exploration, adaptive from success rate)              │
│      │                                                           │
│      ▼                                                           │
│  Policy fast path (~75%)          Imagination (~25%)             │
│  a = argmax(π · s)                Sample 40 noisy rollouts:      │
│  (System 1 — reactive habits)       Δ = s* - ŝ  (goal compass)   │
│                                     u = D · Δ   (inverse model)  │
│                                     ε ~ N(0, σ²) (noise)         │
│                                     ŝ = A·ŝ + B·a + b (predict)  │
│                                     score = Σ V(ŝ_t)             │
│                                     + danger penalty             │
│                                     pick best → imitate π        │
│                                     (System 2 — planning)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    Execute action → observe reward
                             │
                             ▼
              Online learning (all components, every frame)
```

### The two systems

**System 1 — Policy (fast, reactive):** `π · s` directly maps state to action. No simulation, no planning. Learns by imitating imagination's good decisions and by TD reinforcement (good surprise → do it more). Accumulates competence over thousands of frames.

**System 2 — Imagination (slow, planning):** samples 40 noisy trajectories (horizon 5) on the cognitive map, scores each by accumulated predicted value `Σ V(ŝ_t)`, picks the best. The inverse model D provides direction toward the goal s* (best-V state in recent memory). Noise generates diversity; value ranks trajectories.

The policy learns *from* imagination's successes, so it gets better over time. Even when the policy is confident, imagination runs 20% of the time to keep providing fresh imitation signal.

### Action selection: three gates, no phases

SEAL learns and acts from frame 1 — there is no explore/play split. The imagination engine has three internal gates that smoothly transition behavior as the agent learns:

```
Gate 1: ε-random? ──yes──→ random action (exploration)
          │ no
          ▼
Gate 2: π confident? ──yes──→ policy action (System 1, fast)
          │ no                        (20% force-imagination override)
          ▼
Gate 3: V has signal? ──no──→ random action (V flat, ranking is noise)
          │ yes
          ▼
Imagination (40 rollouts, System 2) → imitate π
```

- **Gate 1 (ε-random):** exploration prevents getting stuck. ε adapts from the success rate — losing → explore more (up to 30%), winning → trust the policy (down to 5% floor, never zero).
- **Gate 2 (policy fast path):** when π agrees with imagination >80% of the time, skip the 40 rollouts and use π directly. 20% force-imagination keeps imitation signal flowing.
- **Gate 3 (V-signal):** if V is flat (can't distinguish good from bad states), `Σ V(ŝ_t) ≈ 0` for all trajectories → ranking is noise. Skip the rollouts and act randomly (same result, 40× faster). This replaces the old binary explore/play split with a per-decision quality check.

---

## Key design decisions

### Fixed perception, learned readouts

The visual frontend (DoG + Gabor + temporal differencing) is fixed and contains **zero Pong knowledge** — it detects edges and motion in any image. Only the linear readouts (A, B, D, V, π) are trained. This is the paper's "linear readouts from fixed nonlinear circuits" principle, justified by the random-features / kernel trick.

### Locality-structured encoder E

E is block-sparse: each spatial position's Gabor features map to a *contiguous block* of state dimensions, ordered by `(py, px)`. This ensures spatially adjacent positions map to index-adjacent state dims — which makes the banded dynamics matrix A able to represent motion (a ball shifting one grid cell shifts activation by ~one block, within the ±16 band). The moving-dot gate test confirmed 100% direction accuracy.

### Banded dynamics matrix A

A is 1000×1000 but only ±16 entries per row are learnable (33K params, not 1M). This is SEAL's extension beyond the GCML — the paper's domains don't have autonomous dynamics (the ball in Pong moves on its own). The banded structure makes the delta rule's signal-to-noise ratio viable, and a spectral safeguard (diagonal clip + periodic operator-norm rescaling) prevents multi-step rollout divergence.

### Eligibility traces (not replay)

Pong's ±1 rewards come 20+ frames after the causing actions. A bounded FIFO eligibility trace (30 frames, 120 KB) propagates TD errors backward with exponential decay. This is online TD(λ) credit assignment — not a replay buffer (no sampling, no offline training), consistent with the "no replay" constraint.

### Normalized LMS

All delta rules divide by `‖input‖²`, making the effective learning rate independent of input magnitude. Pong states have `‖s‖²≈324`; without normalization, the learning rates would diverge.

---

## File structure

```
seal/
├── perception/              # Fixed visual frontend (no learning)
│   ├── retinal.py           #   DoG edge enhancement
│   ├── gabor.py             #   Gabor filter banks (spatial + motion)
│   ├── temporal.py          #   Frame differencing
│   ├── encoder.py           #   E: block-sparse locality-ordered projection
│   ├── pipeline.py          #   frame → 704 features → 1000-dim state
│   └── tests/               #   Dimension + locality + moving-dot gate tests
│
├── core/                    # Learned plasticity components (all local rules)
│   ├── dynamics.py          #   A: banded autonomous dynamics + bias b
│   ├── action_effect.py     #   B: action → state change
│   ├── direction.py         #   D: inverse model (paper's W, eq 13)
│   ├── value.py             #   V: TD(λ) value function (cumulative return)
│   ├── policy.py            #   π: reactive policy (imitation + TD reinforcement)
│   ├── gate.py              #   F: frozen feasibility gate (all-ones for Pong)
│   ├── eligibility.py       #   Bounded FIFO trace for credit assignment
│   ├── seal_core.py         #   Combines all + per-step online updates
│   └── reward.py            #   (Legacy R, superseded by value.py)
│
├── imagination/             # Goal-directed trajectory sampling
│   ├── sampler.py           #   40 noisy rollouts (batched, horizon 5)
│   ├── evaluator.py         #   Score by Σ V(ŝ_t) + danger penalty
│   ├── engine.py            #   Policy + imagination blend + exploration
│   └── goal_memory.py       #   s* = best-V state in recent window
│
├── env/                     # Environment wrappers
│   ├── pong_wrapper.py      #   Pong: raw ±1 rewards, 84×84 grayscale
│   ├── phase_manager.py     #   (Legacy, no longer used — gates are in the engine)
│   ├── envs_atari.py        #   Vendored Atari wrappers (D7)
│   └── norm_wrappers.py     #   Streaming Welford normalization
│
├── training/                # Training utilities
│   ├── explore.py           #   (Legacy Phase 1, superseded by unified train.py)
│   ├── play.py              #   (Legacy Phase 2, superseded by unified train.py)
│   └── success_tracker.py   #   Adaptive ε from Laplace-smoothed success rate
│
├── utils/                   # Checkpointing + logging
│   ├── checkpoint.py        #   Save/load all weights (A, B, b, D, V, π)
│   └── metrics.py           #   Per-episode CSV logger
│
├── tests/                   # Integration tests
│   ├── test_slice1_pong.py  #   Phase 1 on real Pong (gates: V-variance, A-stability)
│   └── test_slice2_imagination.py  # Phase 2 (gates: imagination, pipeline, learning)
│
├── config.py                # All hyperparameters
├── train.py                 # Main entry: explore → (gate) → play
└── test.py                  # Evaluation: greedy play, no learning
```

---

## Getting started

### Prerequisites

```
numpy, scipy, gymnasium, ale-py, pytest
```

### Run training

```bash
# Short run (episode-count mode)
python train.py --episodes 100 --seed 0

# Long run (frame-budget mode, with checkpointing + logging)
python train.py --frame-budget 1000000 --checkpoint-interval 50000 \
  --log-path results/seal_v4_1m.csv --seed 0

# Resume from a checkpoint
python train.py --resume results/seal_final_100k.npz \
  --frame-budget 500000 --checkpoint-interval 50000 --seed 0
```

### Monitor progress

```bash
tail -f results/seal_v4_1m.csv
```

Key columns to watch:
- `scored`, `lost` — game score
- `pi_confidence` — policy's agreement with imagination (grows over time)
- `pi_norm`, `v_norm` — policy and value are learning
- `v_variance_signal` — value function has signal (phase gate threshold: 0.05)

### Run tests

```bash
python -m pytest perception/tests/ -v          # Slice 0: perception + moving-dot gate
python -m pytest tests/test_slice1_pong.py -v  # Slice 1: Pong integration
python -m pytest tests/test_slice2_imagination.py -v  # Slice 2: imagination + pipeline
```

### Evaluate a trained model

```bash
# From a checkpoint
python test.py --checkpoint results/seal_final_100k.npz --eval-episodes 20

# Or quick train-then-eval
python test.py --train-episodes 50 --eval-episodes 20 --seed 0
```

---

## The two intellectual roots

| GCML (Nature Machine Intelligence, 2026) | OaK (Sutton, RLC 2025) | SEAL |
|---|---|---|
| Forward model V | Transition models | A, B, b |
| Inverse model W | — | D |
| Imagination (stochastic sampling) | Plan with transition models | Imagination engine |
| Local plasticity, online | Continual learning | All delta rules |
| — | Learn a policy | π |
| — | Learn a value function | V (TD(λ)) |
| — | Meta-learned step sizes | (future work) |
| — | Options (temporal abstractions) | (future work) |

### References

- H. Lin, Y. Yang, R. Zhao, G. Pezzulo, W. Maass. *Neural sampling from cognitive maps enables goal-directed imagination and planning.* Nature Machine Intelligence **8**, 1045–1065 (2026). https://www.nature.com/articles/s42256-026-01254-4
- R. Sutton. *The OaK Architecture: A Vision of SuperIntelligence from Experience.* RLC 2025. https://www.amii.ca/videos/oak-architecture-rich-sutton-rlc2025
- The Alberta Plan for AI Research: https://www.incompleteideas.net/AlbertaPlan.pdf

---

## What SEAL guarantees (regardless of Pong performance)

- A complete intelligent agent — perception, cognitive map, planning, policy, value, continual learning — built from **local plasticity alone**
- **No backpropagation**, no replay buffer, no GPU required (runs on CPU at ~17 fps)
- **Online learning**: every frame, every component, never stops
- **Instant adaptation**: change the goal s* and imagination immediately aims there; no retraining
- **Biologically plausible**: every update is a delta rule a synapse could compute
- **Neuromorphically implementable**: suitable for in-memory computing at low power

---

## License

Research project. Not currently licensed for redistribution.
