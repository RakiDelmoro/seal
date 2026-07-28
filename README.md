# SEAL — Streaming Event-driven Adaptive Learner

SEAL is a reinforcement learning agent that learns **online, one frame at a
time, with no replay buffer and no backpropagation through time**. It is a
recurrent network of spiking neurons (an **LSNN** — LIF + ALIF neurons) trained
by **reward-based e-prop** (Bellec et al., *Nature Communications* 2020), with
**symmetric feedback weights** (B_jk = Wout_kjᵀ, weight transport). Target task:
ALE Pong.

The name **Adaptive** refers to (1) the **adaptive LIF (ALIF) neurons** with
adapting firing thresholds (spike-frequency adaptation, Eqs. 8–10) and
(2) **adaptive dormant-unit regeneration** that reinitializes silent spiking
neurons online — *not* to the feedback weights, which use symmetric e-prop.

The brain-inspired design combines three ingredients from neuroscience that
e-prop theory tells us how to combine for online network learning through
gradient descent:
1. **Eligibility traces** — per-synapse, forward-computed traces of how a
   weight influenced a neuron's recent spikes (the locally-computable part of
   the loss gradient).
2. **Neuron-specific learning signals** — top-down signals L_j routed to each
   neuron via feedback weights B_jk, conveying the output error.
3. **A global reward prediction error δ** — gating synaptic plasticity in real
   time (actor-critic / policy gradient).

## Architecture

```
ALE/Pong-v5  (raw RGB)
   │  NoopReset(30) → MaxAndSkip(4) → EpisodicLife → FireReset
   │  Resize(84) → Grayscale → NormalizeObservation (streaming Welford, clip ±5)
   │  → ONE normalized 84×84 frame per env step  (no frame stacking)
   ▼
SpikingCNN (TRAINABLE stride-conv front-end, paper Fig. 4b)
   │  Conv(1→32, 8, s5) → LeakyReLU → Conv(32→64, 4, s3) → LeakyReLU
   │  rectified-proportional rate coding → input spike trains  [1600 units]
   ▼
LSNN recurrent core  (sim_ms_per_step = 4 ms of sub-stepping at dt = 1 ms)
   │  240 LIF + 160 ALIF neurons, Win [400,1600], Wrec [400,400]
   │  Eligibility traces ε_ji (Eq. 14/22-25) updated each ms, forward-only
   ▼
LayerNorm(z_rate) → Leaky readout (Eq. 11)  → actor logits [6] + critic V [1]
   │  softmax policy π(a|y) → ε-greedy action (explore_eps, stream-x recipe)
   ▼
env.step(a) → r, next_obs  →  δ = r + γV' − V   (unclipped; ObGD δ̄-normalizes)
   │
   ├── e-prop on Win/Wrec:  ΔW = step·δ·F_γλ(L_j · ε̄_ji)   (Eq. 5/36)
   │     step overshooting-bounded (ObGD): step = η/max(1, δ̄·‖tag‖₁·η·κ)
   ├── ObGD on readout + CNN (autograd, δ-free losses, per-group κ/weight-decay)
   │     CNN learning signal: L_in = Winᵀ·L_j (one more symmetric-feedback hop)
   └── B_jk = Wout_kjᵀ  (symmetric e-prop, read live — no separate update)
   │
   discard sample  →  loop
```

**No BPTT.** Temporal credit lives in the **forward eligibility traces** (held
in the layer, not the optimizer). Velocity/temporal context lives in the
**recurrent LSNN state** (no frame stack). This is what makes SEAL *streaming*:
O(1) memory beyond the model weights and the per-synapse eligibility vectors.

## Project Structure

```
seal/
  config.py                # all hyperparameters (single dataclass)
  train.py                 # entry point: headless training
  play.py                  # entry point: greedy inference from checkpoint
  plotting.py              # result plots
  model/
    agent.py               # SEALAgent (encoder + core + heads + e-prop step)
    neurons.py             # LIF, ALIF cells (Eqs. 6-10) + pseudo-derivative ψ
    spiking_conv.py        # TRAINABLE stride-conv front-end -> input spike trains
    lsnn.py                # recurrent LSNN core (Win, Wrec, eligibility traces)
    eligibility.py         # ε-vector recursion (Eq. 14), ε-trace (Eqs. 22-25)
    readout.py             # leaky actor + critic heads (Eq. 11) + LayerNorm input
    broadcast.py           # symmetric B_jk = Wout_kjᵀ feedback weights (live view)
    eprop_optimizer.py     # ΔW = step·δ·F_γλ(tag), ObGD-bounded step
    optim.py               # ObGD (Elsayed et al. 2024): overshooting-bounded GD
    utility.py             # dormant spiking-unit regeneration
    metrics.py             # CSV logger, spike-rate, policy entropy
  env/
    envs.py                # make_env, EnvSpec, warmup (single-frame pipeline)
    envs_atari.py          # vendored Atari wrappers
    norm_wrappers.py        # streaming NormalizeObservation (Welford, no buffer)
  tests/
    test_neurons.py        # LIF/ALIF dynamics vs analytic α, ρ; reset; ψ window
    test_eligibility.py    # Σ_t L·ε ≈ finite-diff dE/dW  (the e-prop theorem)
    test_eprop_pong.py     # RL smoke: no NaN, B_jk drifts, tags reset, sparse firing
  docs/ARCHITECTURE_FLOW.md
```

## Quick Start

```bash
pip install gymnasium ale-py pygame torch

# train headless:
python train.py --frames 5000000 --seed 0

# train with live GUI:
python train.py --frames 5000000 --seed 0 --gui --fps 60

# resume from checkpoint:
python train.py --frames 5000000 --seed 0 --resume results/seal-50.pt

# play greedily from a checkpoint:
python play.py --checkpoint results/seal-pong_best.pt
```

## Key Components

### Spiking neurons (`model/neurons.py`)
- **LIF** (Eq. 6-7): `v_{t+1} = α·v_t + i_syn - z·v_th`, `z = H(v - v_th)`.
- **ALIF** (Eq. 8-10): LIF + adapting threshold `A = v_th + β·a`, `a_{t+1} = ρ·a + z`.
- **Pseudo-derivative** ψ = (γ_pd/v_th)·max(0, 1 − |v−A|/v_th) (γ_pd = 0.3), the
  surrogate gradient for the non-differentiable spike. Zeroed during refractory.

### Eligibility traces (`model/eligibility.py`)
- The locally-computable part of `dE/dW_ji` (Eq. 1/3): `ε_ji^t = ψ_j·ε_vec_ji^t`.
- LIF (Eq. 22-23): `ε_vec = α·ε_vec + z_pre` (low-pass of presynaptic spikes).
- ALIF (Eq. 24-25): 2-D `ε_vec = [ε_v, ε_a]`; the adaptation component `ε_a`
  decays with the slow time constant ρ, providing **highways into the future**
  for temporal credit assignment.
- Validated by finite-difference/autograd checks (`test_eligibility.py`).

### Symmetric e-prop feedback (`model/broadcast.py`)
- **Symmetric e-prop** (default & only mode): B_jk = Wout_kjᵀ, read live from
  the readout weights at each `learning_signal()` call. This is the variant the
  paper used for the Atari Pong result (Fig. 4b/d). It makes L_j equal to the
  exact partial derivative ∂E/∂z_j, giving the strongest, lowest-noise learning
  signal — at the cost of weight transport (biologically implausible, but
  maximally sample-efficient). B is not a separate parameter and has no
  plasticity rule; it tracks Wout automatically.

### Reward-based e-prop (`model/eprop_optimizer.py`, `model/agent.py`)
- Plasticity rule (Eq. 5/36): `ΔW_ji = step·δ_t·F_γλ(L_j^t · ε̄_ji^t)`.
- δ = r + γV_{t+1} − V_t (reward prediction error), unclipped.
- L_j (Eq. 37): `Σ_k B_jk·(π_k − 1_{a=k}) + c_V·B_j^V` — neuron-specific.
- Win/Wrec on e-prop; readout + CNN on autograd ObGD (feedforward; e-prop
  not needed there, per Methods + the paper's own Atari code).
- Stability via **ObGD overshooting bound** (below); the episode-LENGTH
  curriculum (increasing lengths) is kept, but its η ∝ 1/√len coupling is
  off — ObGD supersedes it.

### ObGD — streaming-RL sample efficiency (`model/optim.py`)
From "Streaming Deep RL Finally Works" (Elsayed et al. 2024): streaming RL
fails through instability, not information scarcity (the "stream barrier").
ObGD keeps a per-parameter γλ eligibility trace of the gradient and bounds
the *effective* step size: `step = η / max(1, δ̄·‖e‖₁·η·κ)`. This allows
η = O(1) — every sample takes the largest stable step instead of ~1e-6
scraps. Applied to (a) the e-prop tag update on Win/Wrec, (b) the readout
(actor/critic groups, κ=10, small weight decay), (c) the CNN front-end.
Companion ingredients from the same paper: LayerNorm on the readout input
(makes the bound operative), ε-greedy exploration (decouples exploration
from policy sharpness), λ=0.8 traces on the autograd path.

### Trainable spiking CNN front-end (paper Fig. 4b)
The paper feeds the prediction error "back both to the LSNN **and the spiking
CNN**" (Fig. 4b caption; their code trains the torso with its own optimizer).
SEAL does this the e-prop way: the input-layer learning signal
`L_in = Winᵀ·L_j` (one more hop of symmetric feedback) is injected at the
spike rates `p` (E[spikes] = p — Rao-Blackwellized straight-through), and
autograd takes the local gradient through the feedforward conv stack. No
BPTT: the CNN is memoryless within a frame. A frozen random encoder is an
information bottleneck that caps sample efficiency. Ablation: `--train_cnn`
config flag (False = frozen random).

### Plasticity (`model/utility.py`)
- Dormant = no spike for `dormant_silence_ms`. Every `regen_every` steps, the
  longest-silent units are reinitialized (ReDo-style), combating
  loss-of-plasticity in spiking nets.

## Tech Stack
- PyTorch, Gymnasium + ALE-py, Pygame (GUI).

## References
- Bellec, Scherr, Subramoney et al., "A solution to the learning dilemma for
  recurrent networks of spiking neurons", *Nature Communications* 11:3625
  (2020). https://doi.org/10.1038/s41467-020-17236-y
- Elsayed, Vasan & Mahmood, "Streaming Deep Reinforcement Learning Finally
  Works", arXiv:2410.14606 (2024). ObGD, LayerNorm, ε-greedy, sparse/
  streaming techniques (stream-x algorithms). Code: github.com/mohmdelsayed/streaming-drl
