# SEAL — Progress & Findings

## Current status: EMA + trace clipping run in progress

Architecture: event-driven encoder (EventConv2d + EventLinear) + EMA temporal
input (alpha=0.3) + ObGD with trace clipping (max_z_sum=10,000) + eligibility
traces (lambda=0.8) + adaptive entropy + utility gate + aux task.

Run: `python watch.py --frames 5000000 --seed 0 --fps 60`

---

## What we've built and verified

### Stage 0 — Environment harness
- ALE Pong, 84x84 grayscale, MaxAndSkip(4), EpisodicLife, FireReset
- Streaming Welford observation normalization (clip ±5, no buffer)
- No reward scaling (Pong rewards are already ±1)
- Checkpointing (every 50k frames + best + interrupt save)
- Live Pygame GUI with per-episode diagnostic logging
- **Status: complete, passing tests**

### Stage 1 — Event-driven encoder
- EventConv2d: incremental delta-conv with straight-through thresholded mask
- EventLinear: same pattern for fully-connected layers
- HomeostaticThreshold: adapts theta per layer to keep event rate in 0.5-3% band
- Dead-layer recovery: cuts theta by 0.3x after 100 silent steps
- **Exactness verified**: theta=0 matches dense conv to 1e-6
- **Sparsity verified**: homeostat settles event rates into target band
- **Heatmap**: event mask fires on ball and paddles only
- **Reconstruction drift**: <2e-6 over 300 frames
- **Status: complete, passing tests**

### Stage 2/3 — Streaming RL
- ObGD optimizer (paper Algorithm 3, verbatim)
- Paper-exact loss: no delta in loss (ObGD supplies it), sign(delta) on entropy
- Adaptive entropy: |delta| x tau x grad(H) — always ascent, scaled by surprise
- UtilityTracker: per-parameter gating + dead-unit regeneration
- Aux task: ball position from event mask centroid
- 90% sparse init + LayerNorm at every layer
- **Status: architecture complete, learning behavior under investigation**

---

## Architecture evolution (what we tried and learned)

### Version 1: GRU trunk (single frame input)
- **Architecture**: 1-frame input → EventConv stack → GRUCell(256,256) → heads
- **Temporal mechanism**: GRU hidden state (recurrent memory, no BPTT, detached each step)
- **Result**: ran 1.45M frames. ret20 oscillated between -20.00 and -20.85.
  corrVr swung 0.03 to 0.58. Agent found -15 episodes (6 points) but couldn't
  sustain gains. Entropy stable (no collapse after the sign-flip fix).
- **Diagnosis**: GRU's temporal representation was unstable under trace-only
  learning (no BPTT). The hidden state would learn a velocity representation,
  it would work for a while, the traces would shift, it would "forget," relearn,
  repeat — a limit cycle.
- **Finding**: a trace-trained GRU cannot stably replace frame stacking for
  temporal perception in streaming RL on Pong.

### Version 2: Frame stacking (4-frame input, no GRU)
- **Architecture**: 4-frame stack → EventConv stack (in_ch=4) → EventLinear → heads
- **Temporal mechanism**: 4 stacked frames as 4 input channels (velocity in input)
- **Result**: ran 1.36M frames. ret20 oscillated identically to the GRU version
  (-20.35 to -20.85). BUT z_sum exploded to 800 million (4x more parameters in
  layer 0 → 4x faster trace accumulation). Step size collapsed to 7.6e-19.
  Agent entered permanent coma — entropy → 0.001, features → 0.13, all
  learning stopped.
- **Diagnosis**: the oscillation was NOT caused by the GRU — it's in the
  ObGD + sparse-reward interaction. Frame stacking's 4x parameter count
  accelerated the trace explosion, killing the run faster than the GRU did.
- **Finding**: the bottleneck in streaming RL on Pong is not temporal
  representation but trace accumulation + credit assignment under sparse rewards.

### Version 3: EMA + trace clipping (current)
- **Architecture**: 1 EMA channel (alpha=0.3) → EventConv stack (in_ch=1) → EventLinear → heads
- **Temporal mechanism**: exponential moving average — single accumulated frame
  that captures recent motion as a smooth trail
- **Trace clipping**: caps ||z||_1 at 10,000 to prevent trace explosion
- **Why EMA over frame stacking**:
  1. 4x fewer parameters (1 channel vs 4) → 4x slower trace growth
  2. No wasted channels (1 clean delta vs 4 mostly-empty deltas)
  3. Better FLOP sparsity (7.7x vs 5x in smoke test)
  4. Architecturally consistent with event layers' own temporal accumulation
  5. Smoother homeostat signal (1 stable channel vs 4 noisy ones)
  6. Closer to real event-camera philosophy
- **Why trace clipping**: without it, z_sum grows unboundedly → step size → 0
  → learning stops permanently. Standard technique in eligibility trace
  literature. The paper didn't need it (simpler architecture, more frequent
  resets), but our architecture has more parameters and longer accumulation.
- **Status: launched, monitoring**

---

## Bugs found and fixed

### Bug 1: ObGD normalization (max-abs instead of L1)
- **Symptom**: V inflated to ±72,000 (diverged) in the dense baseline
- **Cause**: implemented ObGD with per-element max-abs normalization instead of
  the paper's global L1 (z_sum) normalization
- **Fix**: matched paper Algorithm 3 verbatim — step_size = alpha / (delta_bar
  x z_sum x kappa)
- **Discovery**: the streaming-RL 50k diagnostic probe

### Bug 2: Alpha cancels in ObGD
- **Symptom**: alpha=1 and alpha=10 produced identical step sizes
- **Investigation**: re-derived ObGD from the paper's Algorithm 3 + official code
- **Finding**: alpha cancels in the bound-active regime (step_size = 1/(kappa x
  delta_bar x ||z||_1)). This is BY DESIGN — it's what enables the paper's
  "single set of hyperparameters" claim. Alpha is NOT a tuning knob; lambda is
  the only true dial (controls ||z||_1 via trace steady state).
- **Fix**: no code change needed — understanding corrected, documented in
  optimizers.py

### Bug 3: Entropy sign flip
- **Symptom**: entropy collapsed to 0.03, agent locked into losing 21-0
- **Cause**: loss had `-entropy_coeff * abs(delta) * entropy` → after ObGD
  multiplied by delta, the entropy push became `delta * abs(delta)` which is
  NEGATIVE when losing (delta < 0) → entropy was pushed DOWN when the agent
  was doing badly (the opposite of what we want)
- **Also**: the policy loss had delta in it, but ObGD also multiplies by delta
  internally → delta was double-counted → policy updates were weighted by
  delta^2 (81x too strong when delta=9)
- **Fix**: paper-exact loss — no delta in value/policy loss (ObGD supplies it),
  entropy term uses sign(delta) so ObGD×delta gives |delta|×ascent (always
  pushes entropy UP, scaled by surprise)
- **Discovery**: reading the official stream_ac code line-by-line

### Bug 4: Homeostat dead-layer overshoot
- **Symptom**: deeper event layers (2, 3) went to 0% event rate and never
  recovered
- **Cause**: adapt_rate=1e-3 was too slow to recover from theta overshoot.
  Layers overshot to theta >> delta, went silent, and the 0.1%/step recovery
  was too gradual to bring them back within the run.
- **Fix**: adapt_rate 1e-3 → 1e-2 (10x faster) + dead-layer safety net (cut
  theta by 0.3x after 100 silent steps, repeat until layer wakes)
- **Status**: partially fixed — layers flicker but don't permanently die

### Bug 5: Trace explosion
- **Symptom**: z_sum grew to 800 million, step size → 7.6e-19, agent entered
  permanent coma (entropy → 0, features → 0, learning stopped)
- **Cause**: eligibility traces accumulate every step (lambda*gamma = 0.792
  decay per step). With 4-channel frame stacking (4,096 params in layer 0),
  traces grew 4x faster than with 1-channel input. Across 1.36M frames,
  z_sum exploded and the ObGD denominator killed the step size.
- **Fix**: trace clipping — cap ||z||_1 at 10,000. If exceeded, scale all
  traces proportionally. Preserves relative credit, bounds the denominator.
- **Status**: implemented, testing in current run

### Bug 6: Checkpoint resume crash
- **Symptom**: resuming from checkpoint crashed with "modified by an inplace
  operation"
- **Cause**: the pending transition's computation graph was built with the
  random init weights (before load_state_dict replaced them) → graph
  referenced old param versions → backprop through stale graph failed
- **Fix**: do a fresh forward pass after load_state_dict to build a new
  pending transition that references the loaded weights

---

## Key metrics logged per episode

```
[EP N] f=frames ret=return ret20=running_avg corrVr=V-return-correlation
  |d|=TD_error V=value ent=entropy act=trunk_activation |h|=feature_magnitude
  z=trace_L1_sum a_eff=effective_step_size
  FLOPs=savings_ratio rates=[per-layer event rates]
```

### What each metric means
- **ret20**: 20-episode running average return. Want it climbing from -21
  toward 0. Scale: -21 (lost 21-0) → 0 (tie) → +21 (won 21-0).
- **corrVr**: correlation between V and actual return. Want positive and
  stable (>0.3). Near 0 = V not tracking return. Negative = V is backwards.
- **entropy**: policy randomness. 1.79 = pure random (exploring). 0 = fully
  deterministic (stuck). Want it stable above ~1.0 while learning.
- **z_sum**: total eligibility trace magnitude. The ObGD denominator. Want it
  bounded (trace clipping caps at 10,000). If it explodes → step size → 0 →
  learning dies.
- **a_eff**: effective step size. How much the weights actually change per
  step. Want it stable above ~1e-5. If it drops below 1e-10 → learning stops.
- **FLOPs**: analytic compute savings vs dense (event-driven vs full conv).
  Want 10-50x.
- **rates**: per-layer event rates (fraction of pixels/units that fired).
  Want all layers in 0.5-3% band. 0% = dead layer. 100% = no sparsity.

---

## Open questions

1. **Will EMA + trace clipping learn Pong?** The architecture is stable and
   efficient, but the oscillation (ret20 swinging -20.0 to -20.85) appeared in
   both the GRU and frame-stack versions. If it persists with EMA, it's a
   fundamental limit of streaming TD(lambda) on sparse-reward Pong, not an
   architecture problem.

2. **Is the oscillation fixable?** The oscillation is in the ObGD + sparse-
   reward interaction (policy improves → visits new states → V has to relearn
   → V degrades → policy gradient degrades → policy regresses → repeat).
   Possible fixes if it persists: lower lambda (shorter traces → less
   oscillation but less credit), trace clipping (already added), or a
   separate value-step-size (decouple V learning rate from policy).

3. **Does the event-driven encoder actually save compute in practice?** The
   analytic FLOP savings are 3-7.7x. But LayerNorm costs O(n) per layer per
   step even on sparse input — it may eat the savings at Stage 3. This is a
   known open problem (documented in README.md).

4. **Multi-scale EMA as a future upgrade?** If 1 EMA shows promise, stacking
   2 EMAs (fast alpha=0.5 + slow alpha=0.1) would give 2 temporal scales
   (recent velocity + trajectory context) with only 2x the parameters. This
   is a known technique (multi-scale temporal features) used in RL, time-
   series, and neuroscience. Not needed now, but a natural upgrade path.
