"""SEAL configuration — the four-box dopamine-free architecture.

Perception → transition model (A,B,b) → value/direction (D) → imagination (policy).
The game reward steers exploration (ε), not skill learning. No learned value
function, no learned policy, no eligibility traces, no TD.
"""
from __future__ import annotations

# ─── Perception: input ────────────────────────────────────────────
FRAME_SIZE = 84
DOG_SIGMA_ON = 1.0          # center Gaussian
DOG_SIGMA_OFF = 3.0         # surround Gaussian

# ─── Perception: Gabor banks ──────────────────────────────────────
GABOR_ORIENTATIONS = [0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345]  # 0°, 45°, 90°, 135°
GABOR_GAMMA = 1.0            # aspect ratio
GABOR_WAVELENGTH_FACTOR = 2.0  # λ = factor × σ
GABOR_SPATIAL_SCALES = [2.0, 4.0, 8.0]
GABOR_MOTION_SCALES = [2.0, 4.0]

SPATIAL_GRID = 4             # 4×4 = 16 positions
MOTION_GRID = 8             # 8×8 = 64 positions

# ─── Perception: computed feature dims ────────────────────────────
N_SPATIAL_POSITIONS = SPATIAL_GRID * SPATIAL_GRID          # 16
N_MOTION_POSITIONS = MOTION_GRID * MOTION_GRID             # 64
SPATIAL_FEAT_PER_POS = len(GABOR_ORIENTATIONS) * len(GABOR_SPATIAL_SCALES)  # 12
MOTION_FEAT_PER_POS = len(GABOR_ORIENTATIONS) * len(GABOR_MOTION_SCALES)    # 8
SPATIAL_FEATURE_DIM = SPATIAL_FEAT_PER_POS * N_SPATIAL_POSITIONS  # 192
MOTION_FEATURE_DIM = MOTION_FEAT_PER_POS * N_MOTION_POSITIONS    # 512
FEATURE_DIM = SPATIAL_FEATURE_DIM + MOTION_FEATURE_DIM           # 704

# ─── State space ──────────────────────────────────────────────────
# The frozen CNN produces a 9×9 grid × 32 channels = 2592 features, flattened
# (py, px, channel) so each spatial position maps to a contiguous 32-dim
# block. This is the locality structure the banded A needs. N_STATE matches
# the CNN output (no separate encoder E — the CNN IS the encoder).
# The GCML/CML papers' core claim: HIGH-D state spaces enable planning
# (Stöckl et al. 2024; GCML Fig. S3b: performance improves 25→3000 dims).
N_STATE = 1296            # 81 positions × 16 channels
CNN_GRID = 9              # 9×9 spatial grid from the frozen CNN
CNN_CHANNELS = 16         # channels per grid position (= block size)
N_POSITIONS = CNN_GRID * CNN_GRID   # 81
SPATIAL_STATE_OFFSET = 0  # (legacy compat; all positions are unified now)
MOTION_STATE_OFFSET = 0

# ─── Encoder E (fixed random, block-sparse) ───────────────────────
E_SEED = 42

# ─── Dynamics A (banded) — the transition model ───────────────────
# "if I do nothing, where does the world go?"  Learns from prediction error
# every frame (self-supervised; no reward needed). Banded (±16) so each row
# only connects to its nearest neighbors — keeps params cheap (33K) and the
# delta-rule signal-to-noise viable. Spectral clip keeps ‖A‖_op ≈ 1.
A_HALF_BAND = 16             # ±16 → K = 33 entries/row. Block size is now
                             # 16 dims (one CNN channel block); ±16 spans
                             # exactly one block.
A_SEED = 42
A_DIAG_INIT = 1.0
A_DIAG_NOISE_STD = 0.01
A_OFFDIAG_NOISE_STD = 0.001
A_DIAG_CLIP = (0.0, 1.0)    # after each update — allow full decay
A_OFFDIAG_ROW_L2_CLIP = 0.5  # allow meaningful shift transfer
A_SPECTRAL_RADIUS_MAX = 1.0  # rescale A if ρ(A) exceeds this
A_SPECTRAL_CHECK_INTERVAL = 100  # check every N updates
ETA_A = 5e-4

# ─── Action effect B — part of the transition model ───────────────
# "what does each button do to the world?"
N_ACTIONS = 3               # NOOP, UP, DOWN
B_INIT_STD = 0.01
ETA_B = 1e-3
B_SEED = 43

# ─── Direction D (inverse model / the paper's W) — the value function ─
# "to move toward the goal, which action has utility?"  The paper calls W a
# "universal value function" — it assigns utility to actions for reaching any
# goal. Learned error-driven (self-supervised, no reward needed):
#   ΔD = η·(a_onehot − softmax(D·Δs)) ⊗ Δs / ‖Δs‖²
# Self-limiting: when D·Δs predicts the action, error → 0 and D stops growing.
# Weight decay is the safety net (replaces the paper's eq-21 min(W,1) which
# assumes nonneg grid-cell states; Pong Gabor features are signed).
D_INIT_STD = 0.01
ETA_D = 1e-3
D_WEIGHT_DECAY = 1e-4   # Oja-style self-limiting (half-life ~7k steps)
D_SEED = 44

# ─── Gate F (frozen all-ones for Pong) ───────────────────────────
# "which actions are even possible?" All three always are, in Pong.

# ─── Imagination engine (the policy — a fixed planning procedure) ─
# Samples 40 noisy rollouts on the cognitive map (using A,B,b + D), scores
# each by geometric distance to the goal, picks the best. Learns NOTHING —
# it's a fixed procedure, like the paper's imagination.
N_TRAJECTORIES = 40
IMAGINATION_HORIZON = 5
NOISE_SIGMA_FLOOR = 0.25   # paper Fig. S1: 0.15 traps in obstacles, 0.25+ escapes
NOISE_SIGMA_SCALE = 0.3     # σ = max(floor, scale × ‖u‖)
DANGER_PENALTY = 2.0
GOAL_WINDOW = 100           # rolling window of recent states for goal selection

# ─── Geometric goal (paper's o* / J = −‖s−o*‖, adapted to Pong) ───
# The goal is read directly from the state (not learned): the ball's
# horizontal column px (0=our side .. 8=opponent side on the 9×9 CNN grid),
# from the energy peak across the 81 positions. s* = recent state with the
# ball most on the opponent's side; rollouts scored by −‖ŝ_H − s*‖₁ +
# danger penalty. This is a ruler, not a predictor — it cannot drift.
GEO_MIN_MOTION_ENERGY = 1e-3   # below this, the ball isn't visible
GEO_OUR_SIDE_PX = 1            # columns 0..1 count as our side (danger)
GEO_GOAL_MIN_EPISODES_DATA = 10  # need ≥ this many recent states to set a goal

# ─── Exploration — the only place the game reward enters ──────────
# The ±1 reward adjusts ε: losing → explore more (see more situations → A and
# D learn the physics better → imagination plans better → score more). This is
# arousal/motivation, not skill learning. No credit assignment, no TD, no
# eligibility traces.
EPSILON_BASE = 0.3
EPSILON_FLOOR = 0.05
TOP5_SAMPLING_PROB = 0.10
SUCCESS_EMA_EPISODES = 20

# ─── Environment ──────────────────────────────────────────────────
FRAME_SKIP = 4
EPISODIC_LIFE = True
ENV_SEED = 0
