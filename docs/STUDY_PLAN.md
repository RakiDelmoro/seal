# SEAL Architecture — Weekly Study Plan

A 10-week plan to understand the research behind SEAL's current architecture
(reward-based e-prop on an LSNN with adaptive feedback weights, Bellec et al.
2020) from the ground up — origins, math, biological motivation, and the code
that implements it — so you can eventually modify it with confidence.

Each week has: **Readings** (papers/sections), **Math to master** (equations
you should be able to derive, not just recognize), **Code to map** (which
SEAL files implement that week's ideas), and **Checkpoint** (a question you
should be able to answer from memory, plus a suggested exercise).

The plan is cumulative: each week builds on the last. Don't skip the neuron
dynamics weeks even though they feel elementary — the eligibility trace math
in Week 3 is *literally* the derivative of the Week 1 dynamics.

---

## Week 1 — Spiking neurons: LIF and the spike response model

**Goal:** Understand the atomic unit — a spiking neuron — and why it's
non-differentiable (the problem everything else solves).

**Readings:**
- Gerstner, Kistler, Naud & Paninski, *Neuronal Dynamics* (free online),
  Ch. 1 (spike as event) + Ch. 3 §3.1 (integrate-and-fire).
- Bellec et al. 2020, Methods §"LIF neurons" (Eqs. 6–7).

**Math to master:**
- The discrete-time LIF update: `v_{t+1} = α·v_t + Σ W_ji·z_i − z_j·v_th`,
  with `α = exp(−dt/τ_m)`. Derive α from the continuous ODE
  `τ_m dv/dt = −v + I`.
- Why the spike `z = H(v − v_th)` makes `∂z/∂v` undefined (Heaviside step).
- Subtractive reset (`−z·v_th`) vs. reset-to-zero — why the paper's
  spike-response-model variant uses subtractive.

**Code to map:** `model/neurons.py` — `LIFNeurons.step()`, `pseudo_derivative()`.
Match each line to an equation.

**Checkpoint:** Why is `v_{t+1} = α·v_t + ...` a *discretization* of a leaky
integrator, and what does `τ_m = 20 ms` imply about how long a neuron "remembers"
its input? *Exercise:* plot `v` over 100 ms for a constant input just above and
just below threshold; observe the regular firing and the reset.

---

## Week 2 — Adaptation: ALIF neurons and slow hidden variables

**Goal:** Understand why the paper insists on ALIF (adapting threshold) — it's
not a detail, it's the reason LSNNs rival LSTMs.

**Readings:**
- Bellec et al. 2020, Methods §"LSNNs" (Eqs. 8–10).
- Bellec et al. 2018 (NeurIPS), "Long short-term memory and learning-to-learn
  in networks of spiking neurons" — the LSNN origin paper. Read §3 (SFA
  enables LSTM-like memory) + Fig. 2.
- Pozzorini et al. 2013, *Nature Neuroscience* (the experimental basis: SFA
  lasts seconds in neocortex).

**Math to master:**
- The 2-D hidden state `h = [v, a]`: threshold `A = v_th + β·a`,
  adaptation `a_{t+1} = ρ·a_t + z`, `ρ = exp(−dt/τ_a)`.
- Why `τ_a` (seconds) >> `τ_m` (20 ms) creates a *second, slow timescale* —
  this is the biological analog of an LSTM cell state.
- The Jacobian `∂h_t/∂h_{t-1}` for ALIF (the 2×2 matrix in
  `ALIFNeurons.dh_dh_prev`): diagonal `[α, ρ−ψβ]`, off-diagonal `∂a/∂v = ψ`.

**Code to map:** `model/neurons.py` — `ALIFNeurons.step()`, `dh_dh_prev()`.
`model/eligibility.py` — the 2-D structure starts here.

**Checkpoint:** Explain in one sentence why a *population* of LIF neurons
cannot solve the cue-counting task (Fig. 3, red curve) but adding ALIF can.
*Exercise:* drive an ALIF neuron with constant input; observe the firing rate
drop over ~1 s as `a` accumulates — that's adaptation.

---

## Week 3 — Eligibility traces: the locally-computable gradient

**Goal:** The heart of e-prop. Understand *exactly* what an eligibility trace
is and is not, and why it's the forward-computable part of the loss gradient.

**Readings:**
- Bellec et al. 2020, Methods §"Mathematical basis for e-prop" (Eqs. 1, 3,
  13–14) — read this three times; it's the whole paper.
- Gerstner et al. 2018, *Frontiers in Neural Circuits* ("Eligibility traces
  and plasticity on behavioral time scales") — the biological notion of
  eligibility traces e-prop formalizes.

**Math to master (the central theorem, Eq. 1/3):**
- `dE/dW_ji = Σ_t L_j^t · ε_ji^t`, where `ε_ji^t = [∂z_j/∂h_j · ε_vec]_local`.
- The eligibility-vector recursion (Eq. 14):
  `ε_vec^t = (∂h/∂h_{t-1})·ε_vec^{t-1} + ∂h/∂W`.
- LIF specialization (Eq. 22–23): `ε_vec` is just a low-pass of the
  *presynaptic* spike train; `ε = ψ·ε_vec`.
- ALIF specialization (Eq. 24–25): the 2-D `ε_vec = [ε_v, ε_a]`; `ε_a` decays
  with the *slow* `ρ` — this is the "highway into the future."
- **Crucial distinction:** `dE/dz_j` (total derivative, needs BPTT) vs
  `∂E/∂z_j` (partial, direct influence only, online-computable). E-prop
  approximates the first with the second.

**Code to map:** `model/eligibility.py` — `LIFEligibility.step()`,
`ALIFEligibility.step()`. Run `tests/test_eligibility.py` and *read the
finite-difference check* — it proves `Σ_t L·ε = dE/dW` for a feedforward layer.

**Checkpoint:** Why is `ε_ji` computable *without knowing the loss*? (Answer:
it only depends on the neuron's own dynamics + presynaptic spikes, not on E.)
*Exercise:* modify `test_eligibility.py` to use a *recurrent* layer (add a
self-connection) and show the e-prop gradient now *diverges* from autograd on
the indirect (route ii) contributions — this is what the L_j approximation
gives up.

---

## Week 4 — From BPTT to e-prop: the re-factorization

**Goal:** Understand what e-prop gives up vs. BPTT, and why that's the right
trade for online/biological learning.

**Readings:**
- Bellec et al. 2020, Methods §"Mathematical basis for e-prop" (Eqs. 15–21,
  the *proof*). Work through the index manipulation that turns Eq. 15 into
  Eq. 21.
- Lillicrap & Santoro 2019, *Curr. Opin. Neurobiol.* ("Backpropagation through
  time and the brain") — why BPTT is biologically implausible.

**Math to master:**
- The classical BPTT factorization (Eq. 15): `dE/dW = Σ_t' (dE/dh_{t'})·(∂h_{t'}/∂W)`.
- The re-factorization (Eqs. 16–21): expand `dE/dh_{t'}` recursively into a
  *sum of learning signals* `L_j^t = dE/dz_j^t` times local Jacobian products,
  then swap the summation indices to pull `L_j^t` out and absorb the rest into
  `ε_ji^t`.
- Route (i) vs route (ii) (Discussion, ¶after Fig. 3): route (i) = through the
  neuron's *own* slow hidden variables (kept by e-prop); route (ii) = through
  *other* neurons' future spikes (blocked by the `∂E/∂z` approximation).

**Code to map:** none new — but re-read `eligibility.py` knowing *why* the
trace is forward-only. The `L_j` (next week) is where route (ii) is lost.

**Checkpoint:** Name a task where symmetric e-prop should *fail* to match BPTT
(paper gives one: Fig. Suppl. 8 — hidden feedforward layers block route (i)).
*Exercise:* trace through which terms in Eq. 19 correspond to route (i) and
which to route (ii).

---

## Week 5 — Learning signals & feedback weights: symmetric, random, adaptive

**Goal:** Understand the learning signal `L_j` and the three ways to generate
it — this is where SEAL's "adaptive e-prop" choice lives.

**Readings:**
- Bellec et al. 2020, §"Mathematical basis" (Eq. 4) + §"Learning phoneme
  recognition" (Fig. 2a) + Supplementary Note 2 (adaptive e-prop).
- Lillicrap et al. 2016, *Nature Comm.* ("Random synaptic feedback weights
  support error backpropagation") — random feedback alignment for
  feedforward nets (the ancestor of random e-prop).
- Nøkland 2016 (Direct Feedback Alignment) + Samadi et al. 2017 — the
  feedforward alignment family.

**Math to master:**
- `L_j^t = Σ_k B_jk·(y_k − y*_k)` (Eq. 4, supervised) and the RL form
  `L_j^t = Σ_k B_jk·(π_k − 1_{a=k}) + c_V·B_j^V` (Eq. 37).
- **Symmetric:** `B = Wout^T` (exact `∂E/∂z`, but weight transport —
  biologically implausible).
- **Random:** `B` fixed random (no weight transport; works because B and
  Wout^T align *implicitly* during training, as in feedback alignment).
- **Adaptive:** `B` random init + local mirror rule
  `ΔB_jk = η_fb·err_k·z_j` (mimics the Wout update; B slowly aligns with
  Wout^T *without* transport). This is SEAL's default.

**Code to map:** `model/broadcast.py` — `FeedbackWeights.learning_signal()`,
`mirror_update()`. The `mask` gates which synapses the mirror rule touches.

**Checkpoint:** Why does *random* feedback alignment work at all, given B is
not the true transpose? (Answer sketch: during SGD, Wout rotates to align
with the fixed B, not the other way around.) *Exercise:* in
`tests/test_eprop_pong.py`, toggle `feedback_mirror` and compare B_drift —
confirm adaptive drifts, random doesn't.

---

## Week 6 — Reward-based e-prop: actor-critic & policy gradient

**Goal:** Understand the RL specialization — how e-prop approximates BPTT-based
deep RL (A3C) with a biologically plausible online rule.

**Readings:**
- Bellec et al. 2020, §"Reward-based e-prop" + Methods §"Reward-based e-prop:
  application of e-prop to deep RL" (Eqs. 5, 30–37).
- Sutton & Barto, *Reinforcement Learning* §13.2–13.4 (policy gradient +
  actor-critic) — the foundation the paper builds on. Eq. 13.8 and 13.11 map
  directly onto the paper's Eqs. 30–34.
- Mnih et al. 2016, *Asynchronous Methods for Deep RL* (A3C) — the
  BPTT-based baseline the paper approximates on Atari.

**Math to master:**
- Policy gradient: `∇E[R_0] ∝ E[Σ_n R_{t_n} ∇log π(a_{t_n}|y)]`.
- Actor-critic variance reduction: replace `R_{t_n}` with advantage
  `(R_{t_n} − V_{t_n})` (Eq. 34).
- The e-prop plasticity rule (Eq. 5/36):
  `ΔW_ji = η·δ_t·F_γ(L_j^t · ε̄_ji^t)`, with
  `δ_t = r_t + γV_{t+1} − V_t` (reward prediction error).
- The `F_γ` low-pass filter: a *second* per-synapse trace that discounts the
  eligibility×learning-signal product by γ — this is the RL eligibility trace
  (Sutton & Barto §12), distinct from the e-prop eligibility trace `ε`.

**Code to map:** `model/eprop_optimizer.py` (`accumulate`, `step` — the
`tag` is the `F_γ`-filtered product), `model/agent.py` `SEALAgent.learn()`
(where δ, L_j, and the tag come together).

**Checkpoint:** What are the *two* eligibility traces in reward-based e-prop,
and what does each carry? (Answer: `ε̄_ji` = the e-prop trace, how W affected
z; `F_γ(L·ε̄)` = the RL trace, discounted credit for past actions.)
*Exercise:* trace the sign conventions — why do we *add* `η·δ·tag` (ascent)
rather than subtract?

---

## Week 7 — Deep RL stability: the deadly triad & the episode schedule

**Goal:** Understand *why* naïve online deep RL is unstable, and the paper's
unusual fix (episode-length schedule) — SEAL's main stability mechanism.

**Readings:**
- Sutton & Barto §11.3–11.4 (the deadly triad: function approximation +
  bootstrapping + off-policy) and §13.6 (deep RL instabilities).
- Bellec et al. 2020, §"Reward-based e-prop" — the paragraph on "additional
  mechanisms to avoid well-known instabilities" and the episode-length +
  inverse-η schedule.
- Mnih et al. 2016 (A3C) — the *parallel-agents* solution SEAL replaces.

**Math to master:**
- Why `max_a' Q(s',a')` bootstrapping + nonlinear function approx + off-policy
  diverges (Sutton & Barto §11.4 examples).
- Why SEAL uses *on-policy* actor-critic (no max-bootstrap) — sidesteps one
  leg of the triad by construction.
- The episode-length schedule: short early episodes → diverse, uncorrelated
  experience → useful skills; then longer episodes → fine-tune. η scaled by
  `1/√(max_len)` to keep total update magnitude bounded as episodes grow.

**Code to map:** `config.py` `episode_schedule` + `eta_length_scale`;
`model/eprop_optimizer.py` `set_episode_length()`, `_length_factor`.

**Checkpoint:** SEAL has no replay buffer and no parallel agents — two of the
three standard deep-RL stabilizers. What replaces them? (Answer: on-policy
actor-critic + the episode-length/η schedule.) *Exercise:* run a short SEAL
training with `eta_length_scale=False` and a fixed long episode length from
step 0 — watch it diverge faster than the scheduled version.

---

## Week 8 — LSNNs ↔ LSTMs, and the spiking-CNN front-end

**Goal:** Two related threads. (a) Why LSNNs are "spiking LSTMs" — the deep
analogy that explains their power. (b) How pixels become spikes (the part of
SEAL not strictly in the paper).

**Readings:**
- Bellec et al. 2018 (NeurIPS), §3 + Fig. 2 — the LSNN≈LSTM demonstration.
- Bellec et al. 2020, Supplementary Note 4 (LSTM eligibility traces) — e-prop
  works on LSTMs too; the math is the same shape.
- Esser et al. 2016, *PNAS* ("Convolutional networks for fast, energy-efficient
  neuromorphic computing") — spiking CNNs (the front-end family SEAL adopts).
- Optional: Roy et al. 2019, *Nature* review on neuromorphic computing —
  context for why spike-coding matters.

**Math to master:**
- The gate analogy: LIF membrane ≈ LSTM input gate; ALIF adaptation ≈ LSTM
  forget gate (the slow `a` variable *gates* future firing like a cell state).
- Rate coding vs. temporal coding — SEAL's `SpikingCNN` uses rectified-
  proportional rate coding (Bernoulli with `p = clip(gain·relu(feats))`).
- Why the front-end is *trainable* — the paper learns it (Fig. 4b: error fed
  back to the spiking CNN). SEAL injects L_in = Winᵀ·L_j at the spike rates
  and backprops through the feedforward conv stack (no BPTT); the δ-gating
  and γλ traces live in the ObGD optimizer. `train_cnn=False` ablates it.

**Code to map:** `model/spiking_conv.py`, `model/lsnn.py` (the Win/Wrec
structure), `config.py` `conv_layers`, `sim_ms_per_step`.

**Checkpoint:** If ALIF ≈ forget gate, what's missing vs. a full LSTM?
(Largely: no output gate, no multiplicative input gate — just additive
integration.) *Exercise:* crank `n_alif` to 0 and re-run a short training —
observe the LSNN degenerates to a plain RSNN (paper's red curve in Fig. 3).

---

## Week 9 — Reading the SEAL codebase end-to-end

**Goal:** Map every equation to a code line. By now the math is familiar; this
week is about fluency in the implementation.

**Readings:**
- `docs/ARCHITECTURE_FLOW.md` (the one-page dataflow).
- `README.md` (the component index).
- Every file in `model/`, in this order, with the paper open:

  | file | paper section | equations |
  |---|---|---|
  | `neurons.py` | Methods §LIF/LSNN | 6–10, pseudo-deriv |
  | `eligibility.py` | Methods §Math basis | 13–14, 22–25 |
  | `lsnn.py` | (composition of the above) | 6–10 + traces |
  | `spiking_conv.py` | §Reward-based e-prop (spiking CNN) | Fig. 4b |
  | `readout.py` | Methods §Network output | 11 |
  | `broadcast.py` | §Math basis (Eq. 4) + Suppl. Note 2 | 4, 37, mirror |
  | `eprop_optimizer.py` | §Reward-based e-prop | 5, 36 |
  | `agent.py` | (the wiring) | 5, 36, 37 |
  | `utility.py` | (engineering; ReDo-style) | — |

**Checkpoint:** Without looking, write down the dataflow from a raw Pong frame
to a weight update, naming every equation involved. *Exercise:* add a
`--symmetric` flag that sets `B = Wout^T` at each step (bypassing the mirror
rule) and run a short comparison — this is the paper's symmetric-e-prop
baseline.

---

## Week 10 — Modification project: pick one extension and implement it

**Goal:** Prove your understanding by changing something nontrivial and
measuring the effect. Pick **one** from the menu below (or propose your own).

**Menu of modifications (roughly increasing difficulty):**

1. **Symmetric e-prop baseline.** Set `B = Wout^T` (frozen, weight transport).
   Compare learning curves vs adaptive. *Tests:* the finite-diff eligibility
   check is unchanged; add a test that `B` tracks `Wout^T`.

2. **Sparse + Dale's-law LSNN.** Implement E/I split (config already has
   `dale`, `rec_sparsity` flags as stubs). Integrate stochastic rewiring
   (Kappel et al. 2018, ref. 24) so the sparse net stays functional.
   *Why:* paper's Fig. 3c orange curve — biological plausibility upgrade.

3. **Adaptive e-prop's mirror rule: compare schedules.** Sweep
   `feedback_lr` ∈ {1e-5, 1e-4, 1e-3}; measure how fast `B_drift` grows and
   whether final return tracks it. *Why:* the paper leaves this as a knob;
   SEAL defaults to a guess.

4. **Learned spiking CNN.** Make the front-end conv weights learnable (via
   e-prop or autograd). *Why:* the paper learns it; SEAL freezes it for
   simplicity. This is the biggest perf lever.

5. **Second Atari game (Fishing Derby).** Paper's Fig. 5 — requires
   recurrent memory (LSTM-level). Add the env preset and run. *Why:* tests
   whether SEAL's LSNN genuinely rivals LSTM on temporal tasks, not just Pong.

6. **Supervised e-prop sanity task (the cue-counting task, Fig. 3).** Build
   the task generator, train with symmetric e-prop, reproduce the BPTT-vs-
   e-prop learning curves. *Why:* the cleanest possible validation that your
   eligibility traces are *exactly* right, in a controlled setting.

**Deliverable for any project:** a short write-up with (a) what you changed,
(b) which equations/code moved, (c) a before/after metric (return curve,
B_drift, spike rate, or the supervised loss), (d) what you'd do next.

---

## Reference: the paper stack at a glance

| # | Paper | Role | Week |
|---|---|---|---|
| 1 | Bellec et al. 2020, *Nat. Comm.* 11:3625 | **the e-prop paper** | 3–7, 9 |
| 2 | Bellec et al. 2018, NeurIPS (LSNN) | LSNN ≈ LSTM origin | 2, 8 |
| 3 | Gerstner et al. 2018, *Front. Neural Circuits* | biological eligibility traces | 3 |
| 4 | Gerstner et al., *Neuronal Dynamics* (book) | LIF/SRM foundations | 1 |
| 5 | Lillicrap et al. 2016, *Nat. Comm.* | feedback alignment | 5 |
| 6 | Sutton & Barto, *RL: An Introduction* | policy gradient, deadly triad | 6, 7 |
| 7 | Mnih et al. 2016 (A3C) | the deep-RL baseline | 6, 7 |
| 8 | Esser et al. 2016, *PNAS* | spiking CNNs | 8 |
| 9 | Pozzorini et al. 2013, *Nat. Neurosci.* | SFA experimental basis | 2 |
| 10 | Kappel et al. 2018, *eNeuro* | stochastic rewiring (Dale's law) | 10 |

All papers are on arXiv or open access; *Neuronal Dynamics* and *Sutton &
Barto* are free online.

---

## How to use this plan

- **Pace:** ~5–8 hrs/week. If a week feels light, pull the next one forward.
  Weeks 3 and 6 are the densest — budget extra time there.
- **Don't read passively.** Every week has a *code-to-map* and an *exercise*.
  Open the SEAL source alongside the paper; the implementation is small enough
  to hold in your head.
- **The tests are your ground truth.** `test_eligibility.py` proves the central
  theorem numerically — re-derive what it checks each time you revisit Week 3.
- **Keep a notebook** of equations in your own notation; the paper's notation
  (rounded ∂ vs. straight d) is deliberate and easy to lose.
