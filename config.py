"""SEAL configuration — OaK-aligned architecture.

Perception → transition model (A,B) → value function (V) + policy (π) + imagination.
The agent learns online from every frame without backpropagation or replay.
Imagination uses the transition model to generate plans, scored by the learned
value function V (blended with a geometric goal). A learned policy π provides
fast reactive action selection and learns by imitating imagination's plans and
from per-step actor-critic updates (streaming TD(λ), no episode buffering).
"""
from __future__ import annotations

# ─── Perception: frozen CNN encoder ───────────────────────────────
# The frozen CNN produces a 9×9 grid × 16 channels = 1296 features, flattened
# (py, px, channel) so each spatial position maps to a contiguous 16-dim
# block. This is the locality structure the banded A needs. N_STATE matches
# the CNN output (no separate encoder E — the CNN IS the encoder).
# The GCML/CML papers' core claim: HIGH-D state spaces enable planning
# (Stöckl et al. 2024; GCML Fig. S3b: performance improves 25→3000 dims).
N_STATE = 1296            # 81 positions × 16 channels
CNN_GRID = 9              # 9×9 spatial grid from the frozen CNN
CNN_CHANNELS = 16         # channels per grid position (= block size)
N_POSITIONS = CNN_GRID * CNN_GRID   # 81

# ─── Dynamics A (banded) — the transition model ───────────────────
# "if I do nothing, where does the world go?"  Learns from prediction error
# every frame (self-supervised; no reward needed). Banded (±160) so each row
# only connects to its nearest neighbors — covers both px shifts (±16) and
# py shifts (±144) in the flattened (py, px, channel) CNN state, keeping
# params cheap (K=321 entries/row) and the delta-rule signal-to-noise viable.
# Spectral clip keeps ‖A‖_op ≈ 1.
A_HALF_BAND = 160
A_SEED = 42
A_DIAG_INIT = 1.0
A_DIAG_NOISE_STD = 0.01
A_OFFDIAG_NOISE_STD = 0.001
A_DIAG_CLIP = (0.95, 1.0)   # keep state magnitude near 1 (no collapse)
A_OFFDIAG_ROW_L2_CLIP = 1.0  # allow meaningful shift transfer
A_SPECTRAL_RADIUS_MAX = 1.0  # rescale A if ρ(A) exceeds this
A_SPECTRAL_CHECK_INTERVAL = 100  # check every N updates
ETA_A = 5e-4

# ─── Action effect B — part of the transition model ───────────────
# "what does each button do to the world?"
N_ACTIONS = 3               # NOOP, UP, DOWN
B_INIT_STD = 0.01
ETA_B = 1e-3
B_WEIGHT_DECAY = 1e-5       # keep B from growing large on residual errors
B_NORM_UPDATE = True        # normalize B update by ‖s_t‖²
B_SEED = 43

# ─── Direction D (inverse model / the paper's W) ──────────────────
# "to move toward the goal, which action has utility?"  The paper calls W a
# "universal value function" — it assigns utility to actions for reaching any
# goal. Learned error-driven (self-supervised, no reward needed):
#   ΔD = η·(a_onehot − softmax(D·Δs)) ⊗ Δs / ‖Δs‖²
# Trained on the prediction residual so D maps the action-induced part of the
# world change onto the action. Weight decay is the safety net.
D_INIT_STD = 0.01
ETA_D = 5e-3            # stronger inverse model guidance
D_WEIGHT_DECAY = 1e-5   # softer decay so D can reach useful magnitude
D_SEED = 44

# ─── Value function V (critic) — streaming TD(λ) ──────────────────
# Stream-AC(λ) style: per-step TD(λ) with an eligibility trace, NO episode
# buffering. The drift that naive one-step TD shows on sparse rewards is
# handled by (a) a long λ trace (→ near-MC credit assignment, still online),
# (b) an overshooting-bounded effective step size (V_ALPHA_MAX), and
# (c) weight decay. This keeps SEAL strictly streaming (O(1)/step, no buffer)
# in line with Elsayed et al. 2024 ("Streaming Deep RL Finally Works").
V_INIT_STD = 0.01
ETA_V = 1e-4                # nominal value learning rate
GAMMA = 0.99                # discount factor
LAMBDA = 0.95               # eligibility-trace decay (high → near-MC, still online)
V_WEIGHT_DECAY = 0.0        # MUST stay ~0 on Pong: ±1 rewards arrive every
                            # ~5000 frames, so any per-step decay drains V back
                            # to zero between rewards (measured: 1e-4 erased
                            # learned value 5× faster than sparse rewards could
                            # rebuild it). Drift is already guarded by
                            # V_ALPHA_MAX + V_TRACE_CLIP.
V_TRACE_CLIP = 10.0         # max eligibility trace magnitude
V_ALPHA_MAX = 0.25          # overshooting bound: cap effective step so each update
                             # corrects at most this fraction of the TD error.
                             # Stops the bootstrap-driven blow-ups that naive TD
                             # shows on sparse rewards (the "stream barrier").
V_SEED = 45

# ─── Average-reward (RVI) critic ───────────────────────────────
# Pong's ±1 arrive at a near-steady rate (~one loss every ~9 frames), but the
# score is NOT in the CNN state, so a discounted V(s)=w·s cannot tell states
# apart and converges to a negative constant ("everything equally bad"; v_norm
# climbs linearly). The average-reward / RVI critic subtracts the agent's own
# running reward rate ρ from every reward, so quiet frames become small
# positives ("I survived") and loss frames become sharp negatives ("lost the
# ball here"). The residual TD error varies per state again — restoring the
# contrast the planner needs. (Yu, Wan, Sutton 2025, "Average-reward RL in
# semi-MDPs via relative value iteration", arXiv:2512.06218.)
#
# ρ is updated ONLY from real rewards (imagined TD must not let the model talk
# to itself) and is saved/restored in checkpoints.
RVI_ENABLE = True
ETA_RHO = 5e-3            # step size for the running reward-rate estimate ρ

# ─── Successor-feature value V_sf — "where does this state lead?" ─
# The successor-features decomposition (Dayan 1993; Barreto et al. 2017):
# V^π(s; w_R) = ψ^π(s)·w_R, where ψ is expected discounted future state
# visitation. Learning the composition ψ·w_R directly costs one vector:
# run TD(λ) on the REWARD-PREDICTOR stream r̂(s) instead of the raw reward.
# r̂(s') is available nearly every frame, so credit propagates densely, and
# the fixed point is the forward-looking landscape "how much reward the
# future is likely to bring from here" — discriminative across states where
# the raw V collapses to a constant. Imagination scores rollouts with V_sf
# when SF_ENABLE (see SEALCore.scorer_value). Average-reward form with its
# own ρ tracker (core/successor.py) so V_sf stays centred.
#
# Default OFF: A/B with `train.py --sf on` before flipping the default.
SF_ENABLE = False
ETA_SF = 1e-3               # critic lr for the auxiliary (r̂) TD stream —
                            # 1e-4 froze V_sf at init over 120k frames: the
                            # r̂ stream's TD errors (~±0.05) are 10x smaller
                            # than the main critic's, so it needs the same
                            # rate as the other small-signal readouts B/D/r̂
SF_SEED = 49

# ─── Policy π (actor) — streaming actor-critic ────────────────────
# Learns by (1) imitating imagination's chosen first action every frame and
# (2) a per-step actor-critic update: reinforce the taken action by the TD(λ)
# error δ (the advantage). No episode buffering — δ is computed online from the
# same trace V uses.
PI_INIT_STD = 0.01
ETA_PI_IMIT = 1e-3          # imitate imagination / chosen action
ETA_PI_AC = 1e-4            # per-step actor-critic rate (scaled by TD error δ)
PI_WEIGHT_DECAY = 1e-5
PI_CONFIDENCE_THRESHOLD = 0.8  # max softmax prob above which π is "confident"
PI_FORCE_IMAGINATION = 0.2     # probability to still run imagination when π confident
PI_SEED = 46

# ─── Imagination engine (System 2 planning) ───────────────────────
# Samples 40 noisy rollouts on the cognitive map (using A,B + D), scores
# each by a blend of predicted value V and geometric distance to the goal.
N_TRAJECTORIES = 40
IMAGINATION_HORIZON = 5
NOISE_SIGMA_FLOOR = 0.25   # paper Fig. S1: 0.15 traps in obstacles, 0.25+ escapes
NOISE_SIGMA_SCALE = 0.3     # σ = max(floor, scale × ‖u‖)
DANGER_PENALTY = 2.0
GOAL_WINDOW = 100           # rolling window of recent states for goal selection

# Blend between value-based and geometric scoring. At 0.0 only geometry is
# used; at 1.0 only the learned value function scores rollouts. Start low
# because V is random, and increase as V learns signal.
IMAGINATION_ALPHA_V = 0.0
IMAGINATION_ALPHA_V_GROWTH = 1e-6  # per-frame increase toward 0.5
IMAGINATION_ALPHA_V_MAX = 0.5

# ─── Geometric goal (paper's o* / J = −‖s−o*‖, adapted to Pong) ───
# The goal is read directly from the state (not learned): the ball's
# horizontal column px (0=our side .. 8=opponent side on the 9×9 CNN grid),
# from the energy peak across the 81 positions. s* = recent state with the
# ball most on the opponent's side; rollouts scored by −‖ŝ_H − s*‖₁ +
# danger penalty. This is a ruler, not a predictor — it cannot drift.

# ─── Pre-score memory (the +1 reward as a goal label, not a learning signal) ─
# When the agent scores (+1), the states from the PRE_SCORE_WINDOW frames
# before the +1 are saved as *proven* goal states. The goal selector prefers
# these over the crude geometric proxy. No replay, no weight updates — just a
# small deque of "states that worked" to aim at. Cold start: empty until the
# first +1, then the geometric proxy fills in (self-improving goal).
PRE_SCORE_WINDOW = 12          # frames before a +1 to save as goal states
PRE_SCORE_MEMORY = 30          # max number of pre-score states to remember

# ─── Reward model r̂(s) — learned reward predictor for imagined TD ─
# One more linear readout: r̂(s) = w_R · s, trained by a normalized LMS delta
# rule on every real (arrival-state, reward) pair. Needed because imagined
# states have no real reward — the model must predict "would arriving here
# bring reward?" during imagined rollouts.
R_INIT_STD = 0.01
ETA_R = 1e-3
R_WEIGHT_DECAY = 1e-5
R_SEED = 47

# ─── Imagined TD — streaming Dyna without a replay buffer ─────────
# After each real transition, imagine K short rollouts from the new state
# using the learned dynamics (A, B), stepping with the current policy π,
# and give the critic a one-step TD update per imagined step using the
# predicted reward r̂. Nothing is stored — rollouts are generated fresh from
# the CURRENT weights and consumed immediately: O(1) memory, no replay, no
# off-policy correction. That is what makes it *streaming*.
#
# Model error compounds with depth, so each imagined update's learning rate
# is scaled by a confidence κ that decays geometrically per imagined step.
# Early in training (A/B still inaccurate) imagined updates are tiny; they
# gain weight automatically as the world model improves.
IMAGINED_TD_ENABLE = True
IMAGINED_TD_K = 2              # imagined rollouts per real step
IMAGINED_TD_HORIZON = 3        # imagined steps per rollout
IMAGINED_TD_ETA = 5e-5         # critic lr per imagined update (before κ scaling)
IMAGINED_TD_KAPPA_DECAY = 0.5  # confidence decay per imagined step
IMAGINED_TD_EXPLORE = 0.2      # uniform-mix for imagined action selection
IMAGINED_TD_SEED = 48

# From-memory variant: additionally rehearse from PROVEN-GOOD states — the
# pre-score memory holds the frames that preceded an actual +1. Rolling out
# from them repeatedly re-enters the rewarded region, keeping V sharp around
# success patterns across episodes (focused rehearsal instead of uniform
# replay; still no buffer — the deque already exists for goal selection).
# OFF by default: enable only after the base A/B is settled.
IMAGINED_TD_FROM_MEMORY_ENABLE = False
IMAGINED_TD_FROM_MEMORY_K = 1  # memory rollouts per real step (when memory nonempty)

# ─── Exploration — ε-greedy exploration rate ──────────────────────
# The ±1 reward adjusts ε: losing → explore more. The value function V and
# policy π receive the raw reward via streaming TD(λ) credit assignment.
EPSILON_BASE = 0.3
EPSILON_FLOOR = 0.05
TOP5_SAMPLING_PROB = 0.10
SUCCESS_EMA_EPISODES = 20

# ─── Environment ──────────────────────────────────────────────────
FRAME_SKIP = 4
ENV_SEED = 0
