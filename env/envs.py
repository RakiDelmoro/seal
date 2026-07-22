"""SEAL environment harness (spec §1, §2.4; Stage 0).

Single entry point: make_env(preset_id, seed) -> (env, EnvSpec).

EnvSpec carries everything the agent needs to build the right encoder + heads
without re-inspecting the env: obs shape (after preprocessing), n_actions,
action_kind, obs_kind, done semantics.

FrameRecorder records `n` raw post-preprocessing frames to disk as a .npy
file for the Stage-1 encoder tests (exactness/sparsity/heatmap/reconstruction).
It stores nothing else and holds at most `n` frames in memory (a fixed
fixture, not a replay buffer -- used only offline for Stage-1 tests).
"""
from __future__ import annotations
import os
import collections
import numpy as np
import gymnasium as gym

from config import PRESETS, EnvPreset
from env.envs_atari import NoopResetEnv, FireResetEnv, EpisodicLifeEnv
from env.norm_wrappers import NormalizeObservation, ScaleReward

# Register ALE + (if available) MinAtar envs with gymnasium.
import ale_py
gym.register_envs(ale_py)
try:  # MinAtar is optional (and currently unused -- no Pong in v1.0.15)
    import minatar.gym as _minatar_gym
    _minatar_gym.register_envs()
except Exception:
    pass


class EnvSpec:
    """Static description of the env as seen by the agent after wrapping."""
    def __init__(self, preset: EnvPreset, obs_shape, n_actions, dtype=np.float32):
        self.env_id = preset.id
        self.domain = preset.domain
        self.obs_kind = preset.obs_kind
        self.action_kind = preset.action_kind
        self.frame_skip = preset.frame_skip
        self.episodic_life = preset.episodic_life
        self.obs_shape = tuple(obs_shape)   # e.g. (1, 84, 84) CHW
        self.n_actions = int(n_actions)
        self.dtype = dtype


class FrameStackWrapper(gym.Wrapper):
    """Stack last `stack_size` frames as channels. Custom version because
    gymnasium's FrameStackObservation has a dtype mismatch with float32
    normalized obs. Output shape: (H, W, stack_size). O(stack_size·H·W)
    memory, fixed — not a replay buffer, just input representation."""
    def __init__(self, env: gym.Env, stack_size: int = 4):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = collections.deque(maxlen=stack_size)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=obs_shape[:-1] + (stack_size,),
            dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(np.array(obs, dtype=np.float32))
        stacked = np.concatenate(list(self.frames), axis=-1)
        return stacked, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        f = np.array(obs, dtype=np.float32)
        for _ in range(self.stack_size):
            self.frames.append(f)
        stacked = np.concatenate(list(self.frames), axis=-1)
        return stacked, info


def _build_atari(preset: EnvPreset, seed: int, frame_stack: int = 4):
    """ALE Pong pipeline: NoopReset -> MaxAndSkip -> EpisodicLife? -> FireReset
    -> Resize(84) -> GrayScale -> NormalizeObservation(clip=5) -> FrameStack(4).
    No ScaleReward (Pong rewards are already ±1). Frame stacking replaces the
    GRU: velocity is in the input (4 stacked frames), no recurrent memory
    needed, no BPTT, no trace-teaching-recurrent-network problem."""
    env = gym.make(preset.id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = gym.wrappers.MaxAndSkipObservation(env, skip=preset.frame_skip)
    if preset.episodic_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    env = NormalizeObservation(env, clip=5.0)
    # Custom frame stacker (gymnasium's FrameStackObservation has a dtype
    # mismatch with our float32 normalized obs). Stacks the last `frame_stack`
    # frames as channels. This puts velocity directly in the input — the event
    # deltas become 4 channels of frame-to-frame motion. No GRU, no BPTT.
    env = FrameStackWrapper(env, stack_size=frame_stack)
    obs, _ = env.reset(seed=seed)
    # obs shape is (84,84,4) after stacking -> agent wants (C,H,W) = (4,84,84)
    obs_chw = np.moveaxis(np.asarray(obs), -1, 0) if obs.ndim == 3 else obs
    spec = EnvSpec(preset, obs_chw.shape, env.action_space.n)
    return env, spec


def make_env(env_id: str, seed: int = 0, frame_stack: int = 4):
    """Build the wrapped env + EnvSpec for one preset id."""
    preset = PRESETS[env_id]
    if preset.domain == "atari":
        env, spec = _build_atari(preset, seed, frame_stack)
    else:
        raise ValueError(f"Unknown domain for env {env_id}: {preset.domain}")
    return env, spec


def obs_to_chw(obs) -> np.ndarray:
    """Convert a gym obs (H,W,C) to agent-facing (C,H,W) float32."""
    obs = np.asarray(obs)
    if obs.ndim == 3 and obs.shape[-1] in (1, 3, 4):
        return np.moveaxis(obs, -1, 0).astype(np.float32)
    if obs.ndim == 2:
        return obs[None].astype(np.float32)
    return obs.astype(np.float32)


def warmup(env, agent, n_frames: int = 1000, seed: int = 0):
    """Fix #2: warm up the Welford normalizer + homeostatic threshold for
    `n_frames` random-policy frames BEFORE any learning. Learning is frozen;
    only env normalization stats and encoder theta adapt. On episode boundaries
    the agent's encoder caches reset (so theta settles on clean deltas). After
    warmup, call agent.reset_after_warmup() for a clean learning start."""
    obs, _ = env.reset(seed=seed)
    agent.reset_episode()
    for _ in range(n_frames):
        a = env.action_space.sample()
        next_obs, r, term, trunc, info = env.step(a)
        agent.warmup_forward(next_obs)        # advance caches + homeostat, no learn
        if term or trunc:
            agent.reset_episode()
            obs, _ = env.reset()
        else:
            obs = next_obs
    agent.reset_after_warmup()


# ---------------------------------------------------------------------------
# FrameRecorder -- Stage-1 fixture only. NOT a replay buffer.
# ---------------------------------------------------------------------------
class FrameRecorder:
    """Record the first `n` post-preprocessing frames to a .npy file.

    Used only offline for Stage-1 encoder tests (exactness, sparsity,
    heatmap, reconstruction). It holds at most `n` frames in memory and
    writes once; it is never read by the training loop. This is a fixed
    test fixture, not a replay buffer.
    """
    def __init__(self, n: int = 10_000, obs_shape=None):
        self.n = int(n)
        self.obs_shape = obs_shape
        self.frames = []
        self.done_flags = []
        self._full = False

    def add(self, obs, done: bool):
        if self._full:
            return
        self.frames.append(obs_to_chw(obs).copy())
        self.done_flags.append(bool(done))
        if len(self.frames) >= self.n:
            self._full = True

    @property
    def full(self) -> bool:
        return self._full

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arr = np.stack(self.frames, axis=0) if self.frames else np.zeros((0,), dtype=np.float32)
        np.save(path, arr)
        # episode boundaries saved alongside
        meta_path = os.path.splitext(path)[0] + "_done.npy"
        np.save(meta_path, np.asarray(self.done_flags, dtype=bool))
        return path

    def episode_boundaries(self):
        """Indices where an episode ended (done=True)."""
        return [i for i, d in enumerate(self.done_flags) if d]
