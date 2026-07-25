# SEAL — Streaming Event-driven Adaptive Learner

SEAL is a reinforcement learning agent that learns online, one sample at a time, with no replay buffer and no backpropagation-through-time. It combines event-driven neural network layers (only processing what changed in the input) with streaming RL (online, batch-size-1, eligibility traces for temporal credit assignment) to achieve compute-efficient learning on ALE Pong.

## Architecture

```
4 stacked frames [1, 4, 84, 84]  (velocity is in the input, no RNN)
  → EventConv2d(4→16, 8, s5)  → LeakyReLU + LayerNorm
  → EventConv2d(16→32, 4, s3) → LeakyReLU + LayerNorm
  → EventConv2d(32→32, 3, s2) → LeakyReLU + LayerNorm
  → flatten → EventLinear(256) → LeakyReLU + LayerNorm
  → heads:
      value:  Linear(256, 1)
      policy: Linear(256, 6)    # softmax, adaptive entropy
      aux:    Linear(256, 3)    # predicts ball position from event mask
```

**Event-driven encoder:** Each conv layer caches its previous output and only processes pixels that changed above an adaptive threshold (homeostatic). Compute scales with activity, not model size. Analytic FLOP savings are reported per step.

**Streaming RL:** ObGD optimizer (overshooting-bounded gradient descent) on the event encoder, eligibility traces (λ=0.8), actor-critic with adaptive entropy bonus. **SwiftTD** (Javed et al. RLC 2024) on the linear heads — True Online TD(λ) + per-feature IDBD step-size optimization + overshoot bound on the eligibility vector + step-size decay, exact where it applies (linear). One forward pass per observation, one update per step, sample discarded immediately.

**Aux task:** A game-agnostic bank of General Value Functions (GVFs) — linear TD(λ) predictions of discounted future cumulants (motion density, positive reward, negative reward, motion spread) whose cumulants come only from the event mask + reward. Replaces the old Pong-specific ball-position aux; transfers unchanged across games. Learned by SwiftTD alongside the Q head.

## Project Structure

```
seal/
  config.py                # all hyperparameters (single dataclass)
  train.py                 # entry point: headless training
  watch.py                 # entry point: training with live Pygame GUI
  plotting.py              # result plots
  model/
    agent.py               # StreamingActorCritic (encoder + heads + step logic)
    event_layers.py        # EventConv2d, EventLinear (incremental delta-conv)
    thresholds.py          # HomeostaticThreshold (adaptive event gate)
    optimizers.py          # ObGD (paper Algorithm 3, verbatim) — encoder
    swift_td.py            # SwiftTD (Javed et al. RLC 2024) — linear heads
    gvf.py                 # game-agnostic GVF bank (Horde/UNREAL-style aux)
    traces.py              # eligibility trace mechanism (docs; traces in ObGD)
    utility.py             # UtilityTracker + dead-unit regeneration
    sparse_init.py         # 90% sparse initialization (paper Appendix F)
    metrics.py             # FLOP counter, aux targets, CSV logger, dormant tracking
  env/
    envs.py                # make_env, EnvSpec, FrameStackWrapper, warmup
    envs_atari.py          # vendored Atari wrappers (NoopReset, FireReset, EpisodicLife)
    norm_wrappers.py        # streaming NormalizeObservation (Welford, no buffer)
  tests/
    test_stage0_envs.py    # env harness: shapes, episode boundaries, frame recorder
    test_stage1_encoder.py # event encoder: exactness (1e-5), sparsity, heatmap, drift
    test_stage2_3_rl.py    # RL smoke: dense + SEAL stability, IDBD optimizer
  README.md
  RUNBOOK.md
```

## Quick Start

```bash
pip install gymnasium ale-py pygame torch

# train with live GUI (watchable):
python watch.py --frames 5000000 --seed 0 --fps 60

# train headless (faster):
python train.py --frames 5000000 --seed 0

# resume from checkpoint:
python watch.py --frames 5000000 --seed 0 --fps 60 --resume results/seal-pong_latest.pt

# generate plots:
python plotting.py
```

## Key Components

### Event-Driven Encoder (`model/event_layers.py`)
- **EventConv2d / EventLinear:** Incremental layers that compute `out = out_prev + W(delta * mask)` where `delta = x - x_prev` and `mask` is a straight-through thresholded gate.
- **Exactness invariant:** With threshold θ=0, output matches dense conv to 1e-5. Verified by Stage-1 tests.
- **FLOP accounting:** Analytic FLOPs reported per step via active-output-location count (not wall-clock).

### Homeostatic Threshold (`model/thresholds.py`)
- Adapts θ per layer to keep event rate in a target band (0.5–3% for ALE Pong).
- Dead-layer recovery: if a layer fires 0% for 100 steps, θ is cut by 0.3× until the layer wakes.

### ObGD Optimizer (`model/optimizers.py`)
- Paper Algorithm 3, verbatim (Elsayed et al. 2024).
- Overshooting-bounded: per-parameter update capped at 1/κ.
- Effective step size α_eff = 1/(κ·δ̄·‖z‖₁) — governed by trace statistics, not nominal α.

### Utility Tracker (`model/utility.py`)
- Per-parameter utility tracking: weights with low utility are gated off.
- Dead-unit regeneration: bottom 1% utility units are reinitialized every 25k steps.

### Environment (`env/`)
- ALE Pong, 84×84 grayscale, 4-frame stacking, MaxAndSkip(4), EpisodicLife.
- Streaming Welford observation normalization (no buffer, clip ±5).
- No reward scaling (Pong rewards are already ±1).

## Tech Stack

- PyTorch
- Gymnasium + ALE-py
- Pygame (GUI for watch mode)

## References

- Elsayed, Vasan & Mahmood, "Streaming Deep Reinforcement Learning Finally Works", arXiv:2410.14606, 2024.
- Javed, Sharifnassab & Sutton, "SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning", RLC 2024 (RLJ_RLC_2024_111).
