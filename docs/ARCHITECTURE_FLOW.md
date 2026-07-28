# SEAL — Architecture Flow Chart

SEAL = **S**treaming **E**vent-driven **A**daptive **L**earner.
A recurrent network of spiking neurons (LSNN) trained online by reward-based
e-prop. One frame in → one action out → one e-prop update → sample discarded.
No replay buffer. No BPTT. No frame stacking. Batch-size 1, online forever.

```
ALE/Pong-v5  (raw RGB)
   │  NoopReset(30) → MaxAndSkip(4) → EpisodicLife → FireReset
   │  Resize(84) → Grayscale → NormalizeObservation (streaming Welford, clip ±5)
   │  → ONE normalized 84×84 frame per env step  (no frame stacking)
   ▼
┌───────────────── SpikingCNN front-end (TRAINABLE, paper Fig. 4b) ──────────┐
│  Conv(1→32, 8, s5) → LeakyReLU → Conv(32→64, 4, s3) → LeakyReLU            │
│  → flatten [1600] → normalize (zero-mean/unit-std)                          │
│  rectified-proportional rate coding:  p = max_p·tanh(gain·relu(feats)/max_p)│
│  → Bernoulli spike  x_i^t [1600]   (sparse; only above-average features)   │
│  (trained via input-layer learning signal L_in = Winᵀ·L_j + ObGD;           │
│   config train_cnn=False reverts to frozen-random for ablation)            │
└────────────────────────────────────────────────────────────────────────────┘
   │  input spikes x_i^t  [1600]
   ▼
┌─────────────── LSNN recurrent core (sim_ms_per_step = 4 ms) ───────────────┐
│  Sub-steps at dt = 1 ms. Population: 240 LIF + 160 ALIF = 400 neurons.     │
│  Weights:  Win [400,1600] (input→core),  Wrec [400,400] (recurrent).       │
│                                                                            │
│  EACH ms sub-step:                                                         │
│    i_syn = Wrec · z_prev + Win · x              (synaptic input currents)  │
│    LIF:   v ← α·v + i_lif;  z_lif = H(v - v_th);  reset v -= z·v_th        │
│    ALIF:  A = v_th + β·a;  v ← α·v + i_alif;  z_alif = H(v - A)            │
│           reset v -= z·A;  a ← ρ·a + z            (Eq. 6-10)               │
│    ψ = (γ_pd/v_th)·max(0, 1-|v-A|/v_th)          (pseudo-derivative)       │
│                                                                            │
│  ELIGIBILITY TRACES (forward, per-synapse, Eq. 14/22-25):                  │
│    LIF:   ε_vec ← α·ε_vec + z_pre;        ε_ji = ψ·ε_vec   (Eq. 22-23)     │
│    ALIF:  ε_v  ← α·ε_v + z_pre                                   (Eq. 22) │
│           ε_a  ← ψ·z_pre + (ρ - ψ·β)·ε_a        (Eq. 24, the slow trace)   │
│           ε_ji = ψ·(ε_v - β·ε_a)                (Eq. 25)                   │
│  ↑ ε_a decays with the SLOW adaptation time constant → "highways into the  │
│    future" that let δ at the end of an episode reach spikes from long ago. │
│                                                                            │
│  After 4 ms:  spike_rate = spike_count / 4   [400]  (readout input)        │
└────────────────────────────────────────────────────────────────────────────┘
   │  z_rate [400],  eligibility ε̄_win [400,1600], ε̄_wrec [400,400]
   ▼
┌─────────────────── Leaky readout (Eq. 11, autograd-trained) ───────────────┐
│  y ← κ·y + Wout·z_rate + b        (leaky output neurons)                   │
│  actor logits = y[:6]  →  softmax → π(a|y)  →  sample action a             │
│  critic V     = y[6]            (scalar value prediction)                  │
│  (feedforward — ordinary autograd + SGD; e-prop NOT needed for readout)    │
└────────────────────────────────────────────────────────────────────────────┘
   │  action a
   ▼
env.step(a) → next_obs, r, term, trunc
   │  next_obs re-enters SpikingCNN → LSNN → readout → next V'
   ▼
┌─────────────────── Reward-based e-prop update (Eq. 5/36) ──────────────────┐
│  δ = r + γ·V'·(1−done) − V                 (reward prediction error)       │
│                                                                            │
│  Learning signal L_j (Eq. 37, neuron-specific):                            │
│    policy_err_k = π_k − 1_{a=k};   critic_err = V' − V                     │
│    L_j = Σ_k B_jk·policy_err_k + c_V·B_j^V·critic_err                      │
│    where B_jk = Wout_kjᵀ  (symmetric e-prop, read live — weight transport)  │
│                                                                            │
│  e-prop tag (F_γ low-pass filter, per-synapse):                            │
│    tag_ji ← γ·tag_ji + L_j · ε̄_ji                                          │
│                                                                            │
│  Weight update (Win, Wrec):                                                │
│    W_ji += η·δ·tag_ji               (η scaled by 1/√max_episode_len)      │
│    (grad_clip safety blanket on |η·δ·tag|)                                 │
│                                                                            │
│  Readout update (autograd, actor-critic):                                  │
│    loss = -log_prob·δ + c_V·½δ² - entropy_coef·H(π)                        │
│    SGD step on Wout, b                                                     │
│                                                                            │
│  Symmetric e-prop feedback:                                               │
│    B_jk = Wout_kjᵀ  (live view; no separate parameter, no plasticity)      │
│    (L_j equals the exact partial derivative ∂E/∂z_j — strongest signal)    │
└────────────────────────────────────────────────────────────────────────────┘
   │  global_step++
   ├── dormant-unit regen every regen_every steps (utility.py)
   ▼
CSV log row  (step, episode, return, td_err, v, spike_rate_hz,
              policy_entropy, b_drift, tag_norm_win, tag_norm_wrec,
              dormant_frac, max_episode_len)
   │
   ▼
pending ← next_state,  a ← a_next  ─── loop back to env.step ───┐
                                                                │
   if done:  reset_episode (clear LSNN state + traces + tags),  │
             env.reset                                          │
                                                                │
   ▲──────── LSNN eligibility traces held per-synapse ──────────┘
```

## The whole loop in one line

```
frame ─▶ SpikingCNN ─▶ LSNN(+eligibility) ─▶ readout ─▶ sample a ─▶ env.step
                         ▲                                            │
                         └── ε_ji computed forward (Eq. 14/22-25) ────┘

δ = r + γV' - V ─▶ L_j = B_jk·err (B=Woutᵀ) ─▶ tag ← γ·tag + L_j·ε̄ ─▶ W += η·δ·tag
                                                          ──── discard ────┘
```

## Why this is streaming (and BPTT-free)

Every box runs **once per environment step**, batch-size 1. Temporal credit
assignment lives in the **forward eligibility traces** (the slow ALIF
component ε_a reaches far back into the past), NOT in any backward pass. The
neuron-specific learning signal L_j routes the output error to each neuron via
**symmetric feedback weights B_jk = Wout_kjᵀ** (weight transport; the variant
the paper used for Atari Pong, Fig. 4b/d). The reward prediction error δ gates
the synaptic update in real time. O(1) memory beyond model weights + per-synapse
eligibility vectors.

**Adaptive** in the name refers to the ALIF adapting neurons and the dormant-
unit regeneration, not to the feedback weights.
