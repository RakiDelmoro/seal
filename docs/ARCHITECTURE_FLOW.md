# SEAL — Architecture Flow Chart

SEAL = **S**treaming **E**vent-driven **A**daptive **L**earner.
One observation in → one action out → one update applied → sample discarded.
No replay buffer. No BPTT. No RNN. Batch-size 1, online forever.

```
ALE/Pong-v5  (raw RGB)
   │  NoopReset(30) → MaxAndSkip(4) → EpisodicLife → FireReset
   │  Resize(84) → Grayscale → NormalizeObservation (streaming Welford, clip ±5)
   │  FrameStack(4)  →  obs [4, 84, 84]   (t, t-1, t-2, t-3; velocity in channels)
   ▼
_to_obs  HWC → CHW → [1, 4, 84, 84]
   │
   │  ┌─ warmup (once, 1000 random-action frames, no learning) ─────────────┐
   │  │   primes Welford stats + per-element θ before any weight update      │
   │  │   90% sparse init applied to whole net                               │
   │  └──────────────────────────────────────────────────────────────────────┘
   ▼
┌─────────────────────── Event-Driven Encoder (incremental) ─────────────────────┐
│                                                                                │
│  each EventConv2d / EventLinear layer, per frame:                              │
│                                                                                │
│     delta = x − x_prev            (cached from previous frame)                 │
│        │                                                                       │
│        │  PerPixelThreshold.observe:  θ[e] = k · EWMA(|delta[e]|)              │
│        │  (θ = 0 during warmup → output ≡ dense conv, exact to 1e-5)          │
│        ▼                                                                       │
│     mask = |delta| > θ           (hard event gate, forward)                    │
│        │  straight-through:  mask + (σ(delta) − σ(delta).detach())             │
│        ▼                                                                       │
│     d = delta · mask_st          (only changed elements pass)                  │
│        │                                                                       │
│        ▼                                                                       │
│     out = out_prev + W(d)        (incremental; bias applied once at t=0)       │
│        │                                                                       │
│        └── cache x_prev, out_prev for next step  (episode reset → recompute)   │
│                                                                                │
│  EventConv2d(4→32,  8, s5)  → LeakyReLU + LayerNorm  ┐                         │
│  EventConv2d(32→64, 4, s3)  → LeakyReLU + LayerNorm  ├ per-element θ each      │
│  EventConv2d(64→64, 3, s2)  → LeakyReLU + LayerNorm  │                         │
│  flatten → EventLinear(→256) → LeakyReLU + LayerNorm ┘                         │
│                                                                                │
│  FLOP accounting: active-output-locations × k² × in × out × 2  (reported,      │
│  not wall-clock)  →  analytic 10–50× savings vs dense conv.                    │
└────────────────────────────────────────────────────────────────────────────────┘
   │  feats h [256]
   ▼
Heads  ← LayerNorm(256, affine-free)   (no learnable params → heads are PURE LINEAR)
   │  Q head:   Linear(256→6)   (Q-values per action; Q IS the value, no V head)
   │  aux head: Linear(256→3)   (ball_x, ball_y, paddle_contact)
   │  z = ln(h)  [256]  ← the LINEAR head input, SwiftTD's feature vector φ
   ▼
aux_targets  ← extract_aux_targets(first-conv event mask centroid)   (free label)
   │
   ▼
ε-greedy action selection
   │  ε: 1.0 → 0.01 linear over 5% of total_frames
   │  greedy:  a = argmax Q(s,a)        ← what we LEARN (Stream Q, off-policy)
   │  explore: a = uniform random        ← what we EXECUTE (is_exploration=True)
   ▼
Transition{logits, aux, action, feats, head_features=z, aux_targets, is_expl}  (held "pending")
   │  action a
   ▼
env.step(a)  → next_obs, r, term, trunc
   │
   │  next_obs re-enters encoder → next logits
   │  bootstrap:  v_next = max_a' Q(s',a')   (0 if done; greedy, off-policy)
   ▼
┌───────────────────────── Streaming TD(λ) Learn (agent.learn) ──────────────────┐
│  Two optimizers, one per the paper's neural recipe:                            │
│    ENCODER → AdaptiveObGD   (traced TD(λ), κ-bound)                            │
│    HEADS   → SwiftTD        (True Online TD(λ) + IDBD + bound + decay)          │
│                                                                                │
│  δ = r + γ · v_next · (1−done) − Q(s,a)            (scalar TD error, Q head)   │
│     γ = 0.99, λ = 0.8                                                          │
│  bootstrap: v_next = max_a' Q(s',a') from next_pending (0 if done)             │
│                                                                                │
│  reset traces?  yes if (done) OR (is_exploration)   ← off-policy correction    │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
   │
   ├─── ENCODER path ────────────────────────────────────────────────────────────┐
   │                                                                            │
   ▼                                                                            │
┌───────────────────── AdaptiveObGD  (paper Algorithm 3, verbatim) ───────────────┐
│  Elsayed et al. 2024, "Streaming Deep RL Finally Works", arXiv:2410.14606      │
│  (applied to the event encoder / trunk params only)                            │
│                                                                                │
│  loss = −Q(s,a) + gvf_weight · Σ_k w_k · ½(GVF_k − c_k)²   (encoder grad src)  │
│        δ is NOT in the loss — ObGD multiplies the gradient by δ internally     │
│  grads = autograd.grad(loss, encoder_params)    ← one-step, no recurrent graph │
│                                                                                │
│  trace    e[p] ← λγ · e[p] + grad[p]                  (eligibility trace)      │
│  2nd mom  v[p] ← β2 · v[p] + (1−β2)·(δ·e[p])²        (Adam-style, per-param)   │
│  v̂       = v / (1 − β2^t)                             (bias correction)       │
│  z_sum   = Σ |e[p] / √(v̂[p]+ε)|                      (normalized, O(n_params))│
│  δ̄       = max(|δ|, 1)                                                       │
│  α_eff   = α / (κ · δ̄ · z_sum)   if product > 1 else α   (overshooting bound) │
│  W[p]   −= α_eff · δ · e[p] / √(v̂[p]+ε)              (gated by utility)        │
│                                                                                │
│  κ = 2.0, β2 = 0.999, ε = 1e-8.  κ-bound = SEAL's overcorrection protection     │
│  for the nonlinear encoder (the paper does NOT apply True Online to conv        │
│  kernels — SwiftTD's exactness is linear-only).                                │
└────────────────────────────────────────────────────────────────────────────────┘
   │                                                                            │
   └────────────────────────────────────────────────────────────────────────────┘
   │
   ├─── HEADS path ──────────────────────────────────────────────────────────────┐
   │                                                                            │
   ▼                                                                            │
┌───────────────────── SwiftTD  (Javed et al. RLC 2024, Algorithm 1) ─────────────┐
│  (applied to the LINEAR heads — exact True Online TD(λ) setting)               │
│  one independent SwiftTD linear learner per output row over φ = [z (256), 1]    │
│                                                                                │
│  Q head:  only the TAKEN action's learner runs the full step each env step     │
│           (δ = r + γ·max_a'Q(s',a') − Q(s,a);  others' traces just decay γλ)   │
│           traces reset on done / exploration (off-policy, Stream Q)            │
│  GVF bank: all 4 learners step every step (off-policy value predictions,       │
│            not action-conditioned). δ_k = c_k + γ·GVF_k(s') − GVF_k(s), each    │
│            with its own λ (0.9 / 0.5 / 0.5 / 0.9). bootstrap = the GVF's own    │
│            next prediction (0 if terminal). cumulants from event mask + reward. │
│                                                                                │
│  TRUE ONLINE TD(λ)  — the Tier-1 trace upgrade, exact here:                    │
│    δw[i] = δ′·z[i] − z_δ[i]·v_δ       (v_δ carried: prediction change last    │
│    w[i] += δw[i]                        update; removes the double-counting)   │
│                                                                                │
│  IDBD step-size optimization  (per-feature β[i], loss-aware):                  │
│    β[i] += θ/e^β · (δ′ − v_δ)·p[i]    (grow steps of useful features, shrink  │
│    α[i] = e^β[i];  clip β to [ln η_min, ln η]   irrelevant ones)               │
│                                                                                │
│  OVERSHOOT BOUND  (correction ratio τ = Σ α[i]φ[i]² capped at η):              │
│    z_δ[i] = min(1, η/τ)·e^β[i]·φ[i]   (eligibility increment scaled so a step  │
│                                        never moves the prediction past target) │
│                                                                                │
│  STEP-SIZE DECAY  (bound fired, τ > η):                                        │
│    β[i] += φ[i]²·ln(ε_decay)         (shrink the steps that caused overshoot)  │
│                                                                                │
│  θ=1e-3, η=0.1, ε_decay=0.99, α_init=1e-7, η_min=1e-15, λ=0.8, γ=0.99.         │
│  Per-learner state: z,h,htemp,hold,p,z̄,z_δ,β (257) + v_old,v_δ (scalars).      │
└────────────────────────────────────────────────────────────────────────────────┘
   │  global_step++
   ▼
┌───────────────────────── Plasticity (model/utility.py) ─────────────────────────┐
│                                                                                │
│  per-unit:   unit_utility ← running mean |feats[256]|                           │
│              since_active[active]=0 else +1      (silence counter)              │
│                                                                                │
│  every 25k steps → ReDo-style regeneration:                                    │
│     dormant = silent > 10k steps  AND  low utility                             │
│     reinit incoming weights to dead units, zero outgoing  (revive dead units)  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
CSV log row  (step, episode, return, td_err, event_flops, dense_flops,
              event_rate, frac_weights_updated, dormant_frac, feat_rank, α_eff, θ)
   │
   ▼
pending ← next_pending,  a ← a_next     ─── loop back to env.step ───┐
                                                                     │
   if done:  reset_episode (clear encoder caches + traces), env.reset│
                                                                     │
   ▲────────────── x_prev / out_prev cached per layer ───────────────┘
```

## The whole loop in one line

```
obs ─▶ event encoder ─▶ Q ─▶ argmax+ε ─▶ env.step ─▶ δ ─▶ encoder:ObGD  ─▶ discard sample
                ▲                                              │  heads:SwiftTD
                └────────── cache x_prev / out_prev ───────────┘
```

Every box runs **once per environment step**, batch-size 1, sample used once and
discarded. Temporal credit assignment lives in the **eligibility traces** (held
in the optimizer), not in any recurrent graph. Velocity lives in the **4 stacked
frames**, not in an RNN. This is what makes SEAL *streaming*: O(1) memory beyond
the model weights and the cached previous input/output per layer.
