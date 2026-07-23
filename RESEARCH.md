# Event-Driven NN + Streaming RL — Literature Validation & Variant Selection

Research notes: validating SEAL's **two halves** — the event-driven encoder
AND the streaming RL pipeline — against the literature, and selecting a
canonical variant that fits the design + task. **No code changes — for
review/validation only.**

SEAL is the *combination* of (i) an event-driven incremental encoder and
(ii) a streaming RL pipeline (batch=1, no replay, no BPTT, eligibility traces,
ObGD, per-parameter utility gate, dead-unit regeneration). Both halves are
validated separately below, then the fit is assessed jointly.

---

## 1. What in SEAL we are validating

### 1a. Event-driven encoder (`model/event_layers.py`)

The incremental delta-conv encoder:

```
delta = x - x_prev
mask  = |delta| > theta                      (hard gate, forward)
mask_st = mask + (sigmoid(delta) - sigmoid(delta).detach())   (straight-through)
d     = delta * mask_st
out   = out_prev + W(d)                       (incremental, bias applied once)
```

with:
- a **per-layer homeostatic theta** keeping event rate in a band (`thresholds.py`)
- an **exactness invariant**: theta=0 ⇒ output == dense conv to 1e-5
- **analytic FLOPs** counted as *active output spatial locations* × k² × C_in × C_out
- **reset_cache()** on episode boundaries (forces full recompute, re-seeds state)
- used in **streaming RL** (batch=1, no replay, eligibility traces, ObGD) on ALE Pong

Four encoder claims to validate:
  (C1) the incremental delta formulation is mathematically sound & precedented
  (C2) the thresholded "fire-on-change" gate is a known, correct neuron model
  (C3) the exactness invariant (sparse/async == dense) is a proven result
  (C4) the active-output-location FLOP model is the right sparse-conv cost model

### 1b. Streaming RL pipeline (`model/agent.py`, `optimizers.py`, `utility.py`, `traces.py`)

```
# one observation, one forward, one update, sample discarded (batch=1, no replay)
trace[p] <- λγ · trace[p] + grad[p]           # accumulating eligibility trace
δ       <- r + γ·V_next·(1-done) - V           # TD error
M       <- α·κ·δ̄·‖z‖_1                        # ObGD overshoot bound
α_eff   <- min(α/M, α) = 1/(κ·δ̄·‖z‖_1)        # (α cancels in bound regime)
W       <- W - α_eff·δ·z                       # fixed-budget normalized step
utility[p] <- decay·utility[p] + (1-decay)·|δ·trace[p]|   # per-param gate
gate[p]    <- utility[p] > τ_low                           # skip dead params
dead units <- silent > N_steps AND bottom-quartile utility  # ReDo-style regen
```

with: no BPTT (one-step backward, graph dropped each step), no GRU/hidden
state (4-frame stack puts velocity in the input), traces reset on episode
boundary, episode-boundary encoder cache reset.

Four streaming claims to validate:
  (S1) eligibility traces + deep nets without BPTT is precedented & sound
  (S2) the ObGD normalized fixed-budget step is a recognized online-RL update
  (S3) the per-parameter utility gate is a recognized plasticity mechanism
  (S4) dead-unit regeneration (ReDo-style) is precedented & correct

Plus: find a canonical variant that fits input-event sparsity + streaming RL +
exact-output.

---

## 2. Literature found (with arXiv IDs)

### Encoder validation matches

| # | Paper | arXiv | What it validates |
|---|-------|-------|-------------------|
| A | **Sigma-Delta Quantized Networks** (O'Connor et al.) | 1611.02024 | C1, C2 — "each layer sends a discretized form of its change in activation to the next layer; computation scales with the amount of change, not the size of the network." This IS the delta·mask incremental formulation. |
| B | **Delta Networks for Optimized Recurrent Network Computation** | 1612.05571 | C2 — "each neuron transmits its value only when the change in its activation exceeds a threshold." This IS the homeostatic event gate, in a recurrent setting. |
| C | **Submanifold Sparse Convolutional Networks** (Graham) | 1706.01307 | C4 — "operates strictly on submanifolds, rather than dilating the observation with every layer." Active output sites stay aligned with active inputs; no dilation. This is exactly the cost model `flops()` reports. |
| D | **Event-based Asynchronous Sparse Convolutional Networks** (Messikommer et al.) | 2003.09148 | C3 + the variant — "framework for converting models trained on synchronous image-like event representations into asynchronous models with **identical output**"; proven identical theoretically + experimentally; up to 20× compute reduction; architecture/task-agnostic; **no train-time change** (compatible with standard backprop training). |

### Task-domain matches (event-stream + RL)

| # | Paper | arXiv | Relevance |
|---|-------|-------|-----------|
| E | **CERiL: Continuous Event-based Reinforcement Learning** (Walters) | 2302.07667 | Same task framing: "specialised network layers which operate directly on an event stream, rather than aggregating events into quantised image frames." Validates event-stream + RL as a combination. Differs: CERiL uses a true event-camera stream; SEAL synthesizes events from frame deltas on ALE Pong. |
| F | **Streaming Deep RL Finally Works** (Elsayed et al.) | 2410.14606 | SEAL's RL base. Confirmed. (= K below.) |

### Streaming pipeline validation matches

| # | Paper | arXiv | What it validates |
|---|-------|-------|-------------------|
| K | **Streaming Deep RL Finally Works** (Elsayed et al.) | 2410.14606 | S1, S2 — the base. Eligibility traces + ObGD + batch=1, no replay, no BPTT. SEAL's `optimizers.py` is Algorithm 3 verbatim. |
| L | **Intentional Updates for Streaming RL** | 2604.19033 | S2 — "specify the intended outcome and solve for the step size." The conceptual basis for ObGD's fixed-budget normalized step (α cancels → step governed by κ·δ̄·‖z‖_1, not nominal α). Confirms the α-cancels finding documented in `optimizers.py`. |
| M | **Revisiting Adam for Streaming RL** | 2605.06764 | S2 — follow-up showing Adam-style adaptive optimizers also work in the streaming regime; positions ObGD as one member of a family of normalized streaming updates. |
| N | **Adaptive & Multiple Time-scale Eligibility Traces for Online Deep RL** | 2008.10040 | S1 — eligibility traces integrated with deep nets online; explicitly addresses the deep-net parameter-dependency that breaks naive traces (SEAL's trace-clipping concern). Directly motivates 4-B below. |
| O | **The Dormant Neuron Phenomenon in Deep RL (ReDo)** | 2302.12902 | S4 — "increasing number of inactive neurons affects expressivity; ReDo Recycles Dormant neurons throughout training." This IS SEAL's `regenerate_dead_units` / bottom-quartile-utility reinit (utility.py). |
| P | **Plasticity Loss in Deep RL: A Survey** | 2411.04832 | S3, S4 — taxonomy of 50+ plasticity-loss mitigations; utility gating + dead-unit regen are standard members. Validates SEAL's `UtilityTracker` as a recognized mechanism, not an ad-hoc hack. |
| Q | **Streaming RL under Partial Observability with Real-Time Recurrent Learning** | 2605.24709 | S1 (context) — streaming RL with POMDPs via RTRL (not BPTT). Relevant to SEAL's GRU-vs-frame-stack finding (PROGRESS V1→V2): the paper confirms truncated-BPTT collapses to one-step under streaming, which is exactly why the GRU trunk couldn't stably learn — validates SEAL's diagnosis that frame stacking > recurrent memory under streaming-RL constraints. |

### Directly relevant to SEAL's open problems (both halves)

| # | Paper | arXiv | Relevance to SEAL |
|---|-------|-------|-------------------|
| G | **Intentional Updates for Streaming RL** | 2604.19033 | (= K/L) step-size-collapse / trace-explosion (Bug 5). |
| H | **Adaptive & Multiple Time-scale Eligibility Traces** | 2008.10040 | (= N) trace-clipping (Bug 5) + multi-scale EMA future upgrade. Literature-endorsed alternative to hard ‖z‖_1 clipping. |
| I | **Region Masking to Accelerate Video Processing on Neuromorphic HW** | 2503.16775 | sigma-delta still wastes compute on insignificant events; region masking. Relates to homeostatic θ (region-level vs per-pixel gating). |
| J | **Transferring dense object detection models to event-based data** | 2210.02607 | Honest caveat: sparse-conv theoretical gains don't always become wall-clock gains. Supports SEAL's analytic-FLOP-only scoping (spec §0). |

---

## 3. Validation verdict

**Both halves of SEAL are correct and well-precedented.**

### 3a. Encoder — mapping

- **C1 (incremental delta)** ✓ — Sigma-Delta Networks (A) is the same formulation:
  `out_t = out_{t-1} + W(Δx_t)`. SEAL's `out_prev + W(delta·mask)` is the masked
  version of exactly this.
- **C2 (fire-on-change gate)** ✓ — Delta Networks (B) is the same neuron model:
  transmit only when |Δactivation| > θ. SEAL applies it at layer inputs.
- **C3 (exactness invariant)** ✓ — ESS (D) *proves* async-sparse == dense with
  identical output, theoretically and experimentally. SEAL's "theta=0 ⇒ dense
  to 1e-5" test is the empirical instance of ESS's theorem. The bias-applied-once
  trick in `_init_buffers` is the correct way to preserve exactness (bias is a
  constant; re-applying it incrementally would double-count).
- **C4 (active-output-location FLOPs)** ✓ — Submanifold Sparse Conv (C) is
  precisely this cost model: count output sites whose receptive field contains
  ≥1 active input, do not dilate. SEAL's `flops()` implements this verbatim.

### 3b. Streaming pipeline — mapping

- **S1 (traces + deep, no BPTT)** ✓ — Elsayed et al. (K) is the base; paper N
  (2008.10040) addresses the deep-net parameter-dependency that makes naive
  traces fragile (exactly SEAL's Bug 5). Paper Q confirms truncated-BPTT
  collapses to one-step under streaming — validating SEAL's PROGRESS V1→V2
  finding that the GRU trunk couldn't stably learn and frame stacking was the
  right call.
- **S2 (ObGD normalized step)** ✓ — Intentional Updates (L) is the conceptual
  basis: solve for the step that yields an intended outcome. SEAL's
  `optimizers.py` documents this exact reframe ("α cancels → fixed-budget
  normalized step, only κ is the true constant, λ is the only dial"). Paper M
  (Revisiting Adam) confirms this is one member of a family of normalized
  streaming updates.
- **S3 (per-parameter utility gate)** ✓ — Plasticity Loss survey (P) places
  utility/statistic-based gating in the standard taxonomy of plasticity
  mitigations. SEAL's `UtilityTracker` (decay EMA of |δ·trace|, gate below τ_low)
  is a recognized member, not an ad-hoc hack.
- **S4 (dead-unit regeneration)** ✓ — ReDo (O) is *the* canonical mechanism:
  recycle dormant neurons throughout training. SEAL's `regenerate_dead_units`
  (bottom-quartile-utility + long-silence detection, reinit incoming, zero
  outgoing) is ReDo-faithful. The survey (P) confirms it is standard practice.

### 3c. The combination

So the design is not a novel-from-scratch construction; it is a coherent
combination of **two established lineages**:
  - **Event half:** Sigma-Delta incremental computation (A,B) + ESS submanifold
    async-sparse propagation (C,D), applied to a synthetic event stream (E).
  - **Streaming half:** ObGD normalized updates (K,L,M) + eligibility traces
    without BPTT (K,N) + utility gating & ReDo regeneration (O,P).

No single paper combines all of these. The novelty of SEAL is the *joint*
combination: an event-driven encoder whose incremental state interacts with
streaming eligibility traces — which is exactly where SEAL's open problems
live (the trace-accumulation-vs-event-sparsity coupling in Bugs 4–5).

### One correctness nuance to flag (encoder, not a bug — a design choice to validate)

Sigma-Delta (A) applies the delta gate **between every layer** — each layer
sends its *output delta* to the next. SEAL currently gates only the **input
delta per layer** (`delta = x - x_prev` at the layer input, where x_prev is the
*previous frame's input to that same layer*). This is the "input-event
sparsity" variant — it is valid (it is what ESS (D) does: events live at the
input, propagation is sparse) and it preserves the exactness invariant. The
fuller Sigma-Delta variant (per-layer output deltas) is a possible upgrade,
discussed in §4-A.

A consequence already observed in SEAL: deeper layers go dead (Bug 4) because
their *input* deltas (the previous layer's output deltas) become tiny after
LayerNorm + LeakyReLU. The homeostat dead-layer hack (θ *= 0.3) is a symptom
fix; the structural fix is per-layer delta propagation (§4-A) — exactly what
Sigma-Delta prescribes.

---

## 4. Recommended variant that fits SEAL's design + task

### Primary recommendation: the **ESS (Event-based Asynchronous Sparse CNN)** variant

Paper D (Messikommer et al., 2003.09148) is the canonical fit because it
matches every SEAL constraint simultaneously:

| SEAL constraint | ESS match |
|---|---|
| Input-event-driven (events at the sensor/input) | ✓ events live at the input |
| Exact output (sparse == dense) | ✓ proven identical output |
| Architecture-agnostic | ✓ any CNN backbone |
| Task-agnostic | ✓ detection, recognition, … (and CERiL (E) extends to RL) |
| No train-time change (backprop, not SNN) | ✓ "compatible with standard NN training" |
| 10–50× compute savings target | ✓ up to 20× measured |
| Episode-boundary reset | ✓ ESS reset/state handling |
| Streaming RL (batch=1, one update/step) | ✓ ESS is per-step incremental by construction; pairs naturally with the streaming pipeline (no batching assumption anywhere) |

**What "adopting the ESS variant" concretely means for SEAL** (for validation,
not implementation):

- Keep the current math: `out = out_prev + W(delta·mask)` is already ESS-correct.
- Replace the *dense* masked-delta conv (`F.conv2d(d, ...)` over the full delta
  tensor) with the **active-output-location gather**: maintain `out_prev`, find
  output sites whose receptive field contains ≥1 active input (submanifold,
  no dilation), and recompute *only those* output sites, accumulating into
  `out_prev`. This makes **execution match the analytic FLOP count** that
  `flops()` already reports — with bit-exact identical output (ESS theorem).
- The current `flops()` (active-output-location count via dilation of the mask
  by the kernel at the stride) is already the ESS cost formula; the variant
  just makes the forward pass use it.
- `reset_cache()` on episode boundary is already ESS-correct state handling.
- **Streaming-pipeline interaction:** because ESS is incremental per-step with
  no batching, it composes cleanly with ObGD's batch-1 update and trace
  accumulation. The only coupling to watch is that ESS's per-step `out_prev`
  cache is *forward state*, while ObGD's traces are *learning state* — they
  reset on the same episode boundary but must not be confused. SEAL already
  keeps them separate (`encoder.reset_cache()` vs `opt.reset()`), which is
  correct.

This is **not a redesign** — it is the sparse-gather execution path the code's
own docstring already flags as a "later optimization," now grounded in a
proven-identical-output framework. The exactness invariant (Stage-1 Test A) is
the ESS theorem's empirical check, so adoption is safe *if and only if* the
sparse gather uses bit-exact accumulation (the docstring already warns of this).

### Streaming-pipeline refinements (both halves interact — order matters)

#### 4-B. Multiple time-scale eligibility traces — fix for trace explosion
Replace the hard `||z||_1 ≤ 10,000` clip (Bug 5 hack) with **multiple λ
timescales** (paper N/H). This is the literature-endorsed approach and unifies
with SEAL's own "multi-scale EMA as future upgrade" idea — but moves the
multi-scale structure into the *trace* domain (credit assignment) rather than
the *input* domain (perception), which is where the actual bottleneck lives
(PROGRESS.md Open Question 1). **Recommendation: validate as the principled
alternative to trace clipping if EMA + clip still oscillates.**

#### 4-C. RTRL-style streaming under partial observability — only if revisiting recurrence
Paper Q shows real-time recurrent learning (RTRL, not BPTT) can give streaming
RL memory under partial observability. SEAL abandoned the GRU trunk (PROGRESS
V1) because truncated-BPTT collapses to one-step under streaming — which paper
Q confirms. If SEAL ever returns to recurrence, RTRL (full forward-mode) is the
literature-correct alternative to frame stacking; it costs O(params) per step
in memory but avoids BPTT entirely. **Recommendation: keep frame stacking; hold
RTRL in reserve only if 4-frame temporal context proves insufficient.**

### Two encoder refinements (optional, ordered by ROI)

#### 4-A. Per-layer delta propagation (Sigma-Delta) — fix for dead deeper layers
Instead of gating only the input delta, threshold **each layer's output delta**
before it becomes the next layer's input. This keeps sparsity alive through
depth (the root cause of Bug 4) and is the canonical Sigma-Delta design (A).
Tradeoff: changes what "event" means (now inter-layer, not just input);
exactness invariant must be re-verified for the propagated-delta variant.
**Recommendation: validate this as the structural fix for Bug 4 before adopting
more homeostat heuristics.**

### Note on the interaction between the two halves
SEAL's hardest bugs (4 and 5) are *not* in either half alone — they are at the
seam where the event half meets the streaming half: sparse event outputs feed
into accumulating traces, and trace statistics feed back into ObGD's step size
which updates the very weights that produce the next event deltas. 4-A fixes
the event side of that seam (keeps events alive through depth); 4-B fixes the
trace side (keeps credit assignment bounded). They are complementary and
could be validated together.

---

## 4-alt. Task-driven custom variant — SEAL-EVT (recommended over ESS)

**Honest assessment of the §4 ESS pick:** ESS was matched on *mechanism
constraints* (exact output, sparse, backprop-compatible, no train-time change,
10–50× savings), NOT on task. ESS was built for event-camera *vision* tasks
(object detection/recognition on DVS sensor data) where the goal is inference
latency reduction on truly sparse, asynchronous sensor events. SEAL's task is
materially different:

| | ESS's task | SEAL's task (Pong) |
|---|---|---|
| Event source | real DVS sensor, async, temporally sparse | **synthetic** frame-deltas, temporally dense for the ball |
| Sparsity pattern | arbitrary, sensor-driven | background ~90% static; ball+paddles are the *entire* task signal |
| Goal | inference latency / wall-clock | **learning stability under streaming RL** |
| Real bottleneck | forward-pass FLOPs | **event↔trace coupling** (Bugs 4–5), not FLOPs |
| Reward | dense supervision (labels) | sparse ±1 at point scored |

No paper targets SEAL's exact intersection (synthetic frame-delta events +
streaming online RL + sparse reward + event-trace coupling as the bottleneck).
Building a task-driven variant is well-justified. **Recommendation: build
SEAL-EVT rather than adopt ESS.**

### 4-alt.0 What it is, in one sentence
Keep the validated incremental delta encoder + validated streaming
ObGD/traces/ReDo, but make **two task-driven replacements** that attack the
actual bottleneck (event↔trace coupling, Bugs 4–5) instead of transplanting a
vision-paper mechanism.

### 4-alt.1 The four moves

| | Move | Replaces | Task fact that forces it |
|---|---|---|---|
| **A** | **Per-pixel variance θ** (Welford — already in your normalizer) | per-layer homeostat scalar | Pong is bimodal: ~90% static background, ball+paddles are the whole signal. A global θ can't tell "dead layer" from "correctly sees only the ball." Per-pixel θ kills Bug 4 at the root. |
| **B** | **Drop rate-band targeting** | the 0.5–3% event-rate target | The correct rate is *whatever the object pixels are* (~1–5%), as a consequence of (A), not a target. Gating events away to hit a rate prior is anti-task. |
| **C** | **Event-gated trace accumulation** | the hard `‖z‖₁ ≤ 10k` clip (Bug 5 hack) | The real bottleneck: traces accumulate on *all* params every step while events fire on *few* pixels → background weights earn credit they never got → z_sum explodes → step collapses. Gate the trace increment per-param by "did this param's receptive field contain an event this step?" → ‖z‖₁ grows with event activity, not param count. **This is the piece no paper has.** |
| **D** | **Keep + extend the aux task** (ball position from event centroid) | (nothing — keep it) | Already task-driven, free supervision from the mechanism. Optional extension: predict ball *velocity* from the event mask's per-pixel delta direction — what the events literally encode and what the policy needs. |

### 4-alt.2 What stays (validated, keep as-is)
- Incremental delta math `out = out_prev + W(delta·mask)` — Sigma-Delta / ESS correct
- Exactness invariant (θ=0 ⇒ dense to 1e-5) — safety check
- Active-output-location FLOP model — right cost model regardless
- The whole streaming half: ObGD, traces, ReDo, utility gate — validated

### 4-alt.3 What gets thrown away
- Per-layer homeostat → (A)
- Rate-band targeting → (B)
- Hard ‖z‖₁ clip → (C)
- Homeostat dead-layer hack (θ *= 0.3) → disappears; (A) prevents the failure
  mode structurally

### 4-alt.4 Why this over ESS
ESS optimizes inference latency on *real event-camera data* — a goal SEAL
doesn't have. SEAL-EVT optimizes *stable online learning of Pong* where the
events are the task and the bottleneck is event-trace coupling. Every move is
forced by a Pong + streaming-RL property, not by a vision paper. And (C) is the
genuinely novel piece — no paper sits at SEAL's intersection (synthetic
frame-delta events + streaming online RL + sparse reward + event-trace
coupling as the bottleneck), so building it is justified, not a workaround.

### 4-alt.5 Risks / things to validate before building
1. **(A) per-pixel θ** needs a variance *floor* so a pixel that's been static
   then suddenly sees the ball wakes up fast — Welford variance is slow to
   grow from 0. Mitigation: θ = max(per_pixel_variance × k, global_floor), or
   an EWMA variance with a short window.
2. **(C) event-gated traces** must not kill learning on the value head (which
   reads the trunk, not pixels). Gate applies to *encoder* params via
   receptive-field membership; trunk Linear + heads keep standard trace
   accumulation. Validate the gating map is computed correctly (which conv
   output locations are active → which weight rows earned trace).
3. **Exactness invariant** must still hold under (A) — per-pixel θ is still a
   forward gate; θ=0 ⇒ dense should still pass. (B) and (C) are training-side,
   don't touch forward exactness.
4. **(C) changes credit-assignment semantics** — params that saw no event get
   no trace. This is *intended* (they contributed nothing to the output
   change) but is a real departure from standard accumulating traces; worth a
   small ablation (gated vs clipped) to confirm it learns faster, not just more
   stably.

### 4-alt.6 How to validate before committing (laid-out plan, no code yet)
1. Re-verify Stage-1 Test A (exactness) holds with per-pixel θ — should pass,
   it's still a forward gate.
2. Instrument: under the current per-layer homeostat, log *per-pixel* event
   rate over an episode → confirm the bimodal distribution (background ~0%,
   object ~100%) that motivates (A).
3. Prototype (C) as a *diagnostic first*: log, per step, what fraction of
   params would be gated under "receptive field saw an event" → confirm it
   tracks event activity, not param count, and that it bounds ‖z‖₁ without a
   hard clip.
4. Only then implement A+C together and run the standard smoke + 50k probe.

### 4-alt.7 ACTUAL OUTCOME OF THE DIAGNOSTIC-FIRST PLAN (implemented)

The plan was run. The result is **asymmetric**: move A was validated and
adopted; move C was falsified by its diagnostic and rejected. This is exactly
why the diagnostic-first plan was the right call — it avoided implementing a
fix for a non-problem.

#### Move A — VALIDATED & IMPLEMENTED (`model/thresholds.py: PerPixelThreshold`)

Premise confirmed empirically (`tests/diag_perpixel_eventrate.py`):
  - per-pixel event rate is bimodal: 72.4% background (rate<1%), 8.2% object
    (rate>20%); per-element delta std varies 128× in layer 0, 3–6× deeper.

Two threshold formulas were tried and REJECTED before the working one:
  1. `θ = k·√(EWMA(Δ²))` (variance): rejected — in deeper layers the delta
     distribution is heavy-tailed, so variance is dominated by the active
     minority and θ is too high for the static majority → layer still goes dead.
  2. per-element homeostat (each element adapts θ to a target firing rate):
     rejected — after warmup θ starts uniform across elements and the
     multiplicative adapt rule keeps it uniform, so it degenerates to the
     per-layer homeostat (no differentiation emerges).

The working formula is **per-element scale-following**:
  `adelta[e] = β·adelta[e] + (1-β)·|Δ[e]|`;  `θ[e] = clip(k·adelta[e], floor, ceil)`
An element fires when `|Δ[e]| > k·(its own typical |Δ|)`. This is per-element
from the first step (adelta differs), robust to heavy tails (mean |Δ|, not
variance), and self-calibrating (static elements → θ=floor, ARMED to fire the
instant real signal arrives; active elements → stable tail rate).

Results (`tests/diag_perpixel_exactness.py` + 3000-frame encoder run + 4000-step
RL loop):
  - Exactness invariant holds: θ=0 ⇒ dense, max err 3e-7 (< 1e-5). ✓
  - No dead layers over 3000 frames: L0=1.4%, L1=6.6%, L2=8.6%, L3=9.1% event
    rate. ✓
  - **Bonus win not in the original premise:** the per-layer homeostat left
    deeper layers HYPERACTIVE (L1=77%, L2=90%, L3=99% firing — not event-driven
    at all). Per-element θ makes deeper layers actually sparse (6–9%) for the
    first time. The encoder is event-driven through its full depth.
  - In the real RL loop: z_sum bounded at ~1500–1600 (vs 800M in PROGRESS),
    α_eff stable ~1e-8 (not collapsing), entropy 1.6–1.7 (healthy), V tracking.
  - Config flag: `cfg.threshold_kind = "perpixel"` (new default); legacy
    `"homeostat"` preserved. ObGD now reads `cfg.max_z_sum`.
  - Original Stage-1, Stage-2/3 smoke tests all still pass.

#### Move C — REJECTED by diagnostic (`tests/diag_event_gated_traces.py`)

Premise was: traces accumulate the FULL gradient (incl. the straight-through
estimator's non-zero-everywhere contribution) on ALL params, so z_sum scales
with total param count, not active-param count → explodes. Gating the trace
increment by the hard event mask should make z_sum scale with event activity.

The diagnostic falsified this. Rigorous check on a frame with 1 active / 127
inactive fc input columns:
  - mean |grad_W| at the active column:   7.2e-2
  - mean |grad_W| at the 127 inactive columns: **0.000 (exactly zero)**

Reason: `grad_W = (forward d)ᵀ · upstream`, and forward `d = Δ·hard_mask` is
*already* zero at inactive locations. The straight-through term
`sigmoid(Δ) - sigmoid(Δ).detach()` is zero in the FORWARD (so d is exactly
Δ·hard_mask forward) and only propagates gradient to `grad_Δ` in BACKWARD — it
does NOT leak into `grad_W`. Therefore traces accumulating `grad_W` are
**already event-gated by the forward hard mask**. Gating the trace again gates
nothing (diagnostic ratio gated/ungated = 1.00×).

**Conclusion:** Bug 5 (trace explosion) is NOT an event-sparsity problem. It is
standard accumulating-trace growth (`trace → grad/(1-λγ)` per active param,
summed over active params and episode length). The correct fixes are the
existing hard `‖z‖₁` clip (kept) or 4-B (multi-timescale traces). Move C is
NOT implemented.

#### What this means for SEAL
  - The encoder is now event-driven through its full depth (move A), fixing
    Bug 4 structurally AND restoring sparsity at depth.
  - Bug 5 remains handled by the hard clip (the right tool, per the rejection
    of C). 4-B (multi-timescale traces) is still on the table as a principled
    future upgrade if the clip proves lossy.
  - The event↔trace seam, hypothesized as the core bottleneck, turned out to be
    only HALF real: the event side (sparse activations) was broken (move A
    fixes it); the trace side was already self-gated by the forward mask (move
    C unnecessary). The real trace problem is accumulation, not coupling.

---

## 5. What I did NOT find / caveats

- No paper combines *all* of {Sigma-Delta incremental, submanifold sparse,
  exact-output, streaming-RL with ObGD+traces+ReDo} in one system. SEAL's
  combination is novel as a whole, but each component in *each half* is
  precedented. Closest single-system matches: CERiL (E) for {event-stream + RL},
  ESS (D) for {Sigma-Delta + submanifold + exact-output}, Elsayed (K) for the
  streaming-RL half.
- Sparse-conv wall-clock speedups are not guaranteed (paper J). SEAL already
  correctly scopes FLOPs as analytic-only (spec §0).
- I did not verify the ESS "identical output" proof line-by-line against
  SEAL's straight-through mask; the straight-through gradient estimator is a
  *training* addition on top of the (forward-exact) ESS framework. Forward
  exactness holds; the ST trick is orthogonal and precedented (it is standard
  for hard-gate differentiation). Validate that ST does not break the exactness
  test (Stage-1 Test A checks forward only, so it should pass).
- ReDo (O) recycles neurons by *activation* dormancy; SEAL's regen combines
  activation dormancy *and* utility (|δ·trace|). This is a stricter criterion
  than ReDo and is arguably an improvement, but it is a SEAL-specific variant —
  validate it doesn't over-prune (the bottom-25%-among-silent threshold is the
  guardrail).

---

## 6. Decision points for you to validate

**Top-level choice (pick one path):**

- **Path 1 — ESS (§4):** adopt the literature variant as-is. No math change,
  grounds the sparse-gather execution in a proven-identical-output framework.
  Best if you want a defensible, precedented mechanism and are OK optimizing
  for inference efficiency, not task-specific learning stability.
- **Path 2 — SEAL-EVT (§4-alt, RECOMMENDED):** build the task-driven custom
  variant. Attacks the actual bottleneck (event↔trace coupling, Bugs 4–5)
  with two task-forced moves (per-pixel variance θ + event-gated traces).
  Best if you want the design to serve Pong+streaming-RL, not a vision task.
  The novel piece (event-gated traces) is justified by the gap in the
  literature, not a workaround.

**If Path 2 (SEAL-EVT), validate in this order (§4-alt.6):**

1. Re-verify Stage-1 Test A (exactness) holds with per-pixel θ.
2. Instrument per-pixel event rate under the current homeostat → confirm the
   bimodal (background ~0% / object ~100%) distribution that motivates move A.
3. Prototype move C as a diagnostic first → confirm gated-param fraction
   tracks event activity, not param count, and bounds ‖z‖₁ without a hard clip.
4. Only then implement A+C together and run smoke + 50k probe.

**Either path — housekeeping:**

5. **Acknowledge the streaming half's lineage explicitly** (ObGD/Intentional
   Updates K,L,M + traces-without-BPTT K,N + ReDo/utility-gate O,P)? Currently
   the README cites only the streaming-RL base paper (K).
6. **Hold 4-C (RTRL) in reserve** only if 4-frame temporal context proves
   insufficient (paper Q validates the V1→V2 GRU diagnosis meanwhile).
7. **Cite full lineage** in the README — encoder: Sigma-Delta (1611.02024) +
   Submanifold Sparse Conv (1706.01307) + ESS (2003.09148) + CERiL (2302.07667);
   streaming: Elsayed (2410.14606) + Intentional Updates (2604.19033) +
   Multi-timescale traces (2008.10040) + ReDo (2302.12902) + Plasticity survey
   (2411.04832) + RTRL streaming (2605.24709)?
   If SEAL-EVT: add a note that the event-gated trace accumulation (move C) is
   a SEAL-specific co-design not found in the literature, motivated by the
   event↔trace coupling bottleneck.
