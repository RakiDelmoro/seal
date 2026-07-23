# SEAL Pipeline — From Input to Update

A step-by-step trace of what happens to one observation as it flows through
SEAL. Follow this to debug any metric in the logs.

```
Config (current defaults):
  rl_algorithm = "q"          (Stream Q, off-policy)
  threshold_kind = "perpixel" (move A: per-element scale-following theta)
  ema_alphas = None           (single EMA, alpha=0.2 → ~4 frame memory)
  scale_reward = False        (Pong rewards are already ±1)
  epsilon: 1.0 → 0.01 over 5% of total_frames
  lam = 0.8, kappa = 2.0, max_z_sum = 10000
```

---

## STAGE 1 — Environment pipeline (one env.step → one observation)

Wrappers applied in order (inside → outside). Each wraps the previous:

```
ALE Pong ROM (210×160×3 RGB)
  │
  ├─ 1. RecordEpisodeStatistics    tracks return/length for info["episode"]
  ├─ 2. NoopResetEnv               0-30 random no-ops on reset (randomization)
  ├─ 3. MaxAndSkipObservation(4)   repeat action 4×, return max of 2 frames
  │                                 (→ 1 agent step = 4 ROM frames)
  ├─ 4. EpisodicLifeEnv            treat life-loss as episode end (more resets)
  ├─ 5. FireResetEnv               press FIRE on reset (Pong needs no FIRE, no-op)
  ├─ 6. ResizeObservation(84,84)   210×160 → 84×84
  ├─ 7. GrayscaleObservation       RGB → 1-channel gray, keep_dim → (84,84,1)
  ├─ 8. NormalizeObservation(clip=5)  per-pixel Welford mean/std, clip ±5
  │                                      (streaming, no buffer)
  ├─ 9. ScaleReward                [DISABLED — scale_reward=False]
  └─ 10. EMAWrapper(alpha=0.2)     ema = 0.2·frame + 0.8·ema_prev
                                     (1 channel, ~4-frame geometric memory)
```

**Output of env.step:** `obs` is `[84, 84, 1]` float32 (HWC), `r` is ±1 or 0
(raw Pong reward, unscaled), `info["raw_reward"]` = same ±1.

The EMA is the **temporal memory**: instead of stacking 4 raw frames (the
paper's approach), we keep 1 accumulated frame that holds a ~4-frame trail.
The ball appears as a streak whose length encodes velocity. This is the
"velocity is in the input" design — no RNN needed.

---

## STAGE 2 — Agent forward pass (agent.act(obs))

### 2a. Format input
```
obs [84,84,1] HWC  →  _to_obs()  →  x [1,1,84,84] CHW float32
```

### 2b. Event encoder (3 EventConv2d + 1 EventLinear)
```
x [1,1,84,84]
  │
  ├─ EventConv2d(1→16, k8, s5)   →  [1,16,16,16]
  │    │
  │    │  INSIDE each EventConv2d (the event mechanism):
  │    │    delta = x - x_prev                          [1,1,84,84]
  │    │    threshold.observe(delta)                    updates per-pixel theta
  │    │         adelta[e] = 0.99·adelta[e] + 0.01·|delta[e]|   (EWMA)
  │    │         theta[e]  = clip(2.0·adelta[e], 1e-6, 1.0)
  │    │    mask  = |delta| > theta                    (hard gate, forward)
  │    │    mask_st = mask + sigmoid(delta) - sigmoid(delta).detach()  (ST grad)
  │    │    d = delta * mask_st                        (zero where no event)
  │    │    out = out_prev + W(d)                       (incremental!)
  │    │         bias applied ONCE at first frame, then bias*0
  │    │    cache: x_prev ← x,  out_prev ← out          (for next step)
  │    ↓
  │  LeakyReLU + LayerNorm                             [1,16,16,16]
  │
  ├─ EventConv2d(16→32, k4, s3)  →  [1,32,5,5]   (same mechanism)
  │  LeakyReLU + LayerNorm                             [1,32,5,5]
  │
  ├─ EventConv2d(32→32, k3, s2)  →  [1,32,2,2]   (same mechanism)
  │  LeakyReLU + LayerNorm                             [1,32,2,2]
  │
  └─ flatten → EventLinear(128→256) → [1,256]     (same event mechanism)
     LeakyReLU + LayerNorm                             [1,256]  = feats
```

**Key invariants:**
- With theta=0, `out == dense conv` exactly (verified: 3e-7 error). The event
  gate only *skips* computation where nothing changed.
- `out_prev` is the running output state. Each step adds `W(delta·mask)` to it.
  On episode boundary, `reset_cache()` forces a full recompute (re-seeds).
- Per-pixel theta (move A): static background pixels → theta→floor → armed to
  fire the instant the ball arrives. Object pixels → theta tracks their motion
  scale. No layer can go globally dead.

### 2c. Heads
```
feats [1,256]
  │
  ├─ LayerNorm(256)
  ├─ value head:  Linear(256→1)   →  v  (scalar, V(s) in AC / not used in Q)
  ├─ policy head: Linear(256→6)   →  logits  (Q-values in Q mode, one per action)
  └─ aux head:    Linear(256→3)   →  aux  (predicts ball x,y,paddle-contact
                                            from event-mask centroid — free
                                            supervision from the event mechanism)
```

### 2d. Action selection (epsilon-greedy)
```
if random() < epsilon:          # exploration
    action = random(0..5)
    is_exploration = True
else:                           # exploitation
    action = argmax(logits)     # Q mode: greedy = best Q-value
```
epsilon decays linearly: 1.0 → 0.01 over 5% of total_frames (e.g. 250k of 5M).

**Returns:** `(action, Transition)` — the Transition holds v, logits, aux,
action, feats for the *next* step's TD update.

---

## STAGE 3 — Agent learning (agent.learn(pending, r, v_next, done))

Called one step *later* — `pending` is the previous transition, `r` is the
reward received after acting on it, `v_next` is the bootstrap from the current
frame.

### 3a. TD error (Stream Q, off-policy)
```
q_sa = pending.logits[action]              # Q(s, a) for the action taken
v_next = max_a' Q(s', a')                  # bootstrap = greedy next Q (NOT the
                                            #   action actually taken → off-policy)
delta = r + gamma·v_next·(1-done) - q_sa   # TD error δ
```

### 3b. Loss (paper-exact: NO delta in the loss — ObGD supplies it)
```
loss = -q_sa + aux_weight·MSE(aux, aux_targets)
```
- Only the taken action's Q-value gets gradient (Q-learning).
- No policy gradient, no entropy (epsilon-greedy handles exploration).
- aux_targets come FREE from the event mask centroid (ball position).

### 3c. Backward (one-step, no BPTT)
```
grads = autograd.grad(loss, all_params)    # one-step; graph dropped
```

### 3d. Trace accumulation + ObGD update
```
# eligibility traces (temporal credit assignment, replaces BPTT)
trace[p] = lam·gamma·trace[p] + grad[p]      # accumulating traces

# trace clipping (Bug 5 safety net — rarely engages now)
if ||z||_1 > max_z_sum:  scale all traces by max_z_sum/||z||_1

# ObGD (overshooting-bounded gradient descent, paper Algorithm 3)
delta_bar = max(|delta|, 1)
M = alpha·kappa·delta_bar·||z||_1           # overshoot bound
alpha_eff = min(alpha/M, alpha) = 1/(kappa·delta_bar·||z||_1)   # alpha cancels!
W[p] -= alpha_eff · delta · trace[p]         # fixed-budget normalized step

# trace reset
reset = done OR is_exploration               # off-policy correction:
                                             #   random actions get no credit
```

**Key insight (from optimizers.py):** alpha *cancels* in the bound-active
regime. The effective step size is governed by `kappa·delta_bar·||z||_1`, NOT
by the nominal alpha. **lambda is the only true dial** — it controls ||z||_1
via the trace steady state `||z||_1 ≈ g/(1-lam·gamma)`.

### 3e. Per-step housekeeping
```
utility[p] = decay·utility[p] + (1-decay)·|delta·trace[p]|  # per-param gate
if utility[p] < tau_low: skip update on p this step          # plasticity
global_step += 1
_update_epsilon()                                             # decay ε
every 25k steps: regenerate dead trunk units (ReDo-style)
threshold.update(event_rate)  # homeostat (no-op for per-pixel)
```

---

## STAGE 4 — Episode boundary

```
on done:
  agent.learn(pending, r, v_next=0.0, done=True)   # terminal: v_next=0
  agent.reset_episode():
    encoder.reset_cache()    # all EventConv/EventLinear: _initialized=False
                             #   → next frame is full recompute (re-seeds out_prev)
    opt.reset()              # zero all eligibility traces
  env.reset()
  raw_ep_return logged as pong=  (the real Pong score)
```

---

## How to read the log line

```
[EP 131] f=27337 pong=-21 ret20=-20.8 corrVr=+0.003 |d|=3.043 V=+2.62
  ent=0.000 act=0.28 |h|=0.33 z=188 a_eff=1.4e-05 FLOPs=3.1x
  rates=[0.022, 0.037, 0.031, 0.008]
```

| field | meaning | want |
|-------|---------|------|
| `pong` | raw Pong score this episode (-21..+21) | climbing toward 0 |
| `ret20` | 20-ep avg of raw Pong score | climbing toward 0 |
| `corrVr` | corr(V, return) over last 50 eps | >0.3 (V tracks return) |
| `\|d\|` | abs TD error | finite, not exploding |
| `V` | value/Q of current state | tracking return scale |
| `ent` | policy entropy (0 in Q mode) | n/a for Q |
| `z` | trace L1 sum (ObGD denominator) | bounded (~hundreds-thousands) |
| `a_eff` | effective step size | >1e-5 (not collapsing) |
| `FLOPs` | dense/event compute ratio | 3-15× savings |
| `rates` | per-layer event rates (4 layers) | 1-10%, no 0% (dead) |

---

## The two halves and where each fix lives

```
EVENT HALF (encoder)              STREAMING HALF (RL loop)
────────────────────              ────────────────────────
EMAWrapper (input)                Stream Q (off-policy TD)
EventConv2d (delta·mask)          epsilon-greedy exploration
PerPixelThreshold (move A)        trace reset on exploration
  ↑ fixed Bug 4 (dead layers)       ↑ broke the -20.5 plateau
LayerNorm + LeakyReLU             ObGD (normalized step)
aux task (ball position)          eligibility traces (lam=0.8)
                                    ↑ trace clip (Bug 5 safety net)
                                  utility gate + ReDo regen
```

Move A (per-pixel theta) fixed the event half. Stream Q + epsilon-greedy
fixed the streaming half. ScaleReward was tried and disabled (distorted
Pong's clean ±1 rewards). Move C (event-gated traces) was rejected by
diagnostic (traces already gated by forward mask).
