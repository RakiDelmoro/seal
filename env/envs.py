"""SEAL environment harness.

Single entry point: make_env(preset_id, seed) -> (env, EnvSpec).

4-frame stacking provides temporal context (velocity is in the 4 stacked
frames), matching the streaming-RL paper's proven Atari architecture. The
event-driven encoder then processes frame-to-frame deltas across these
channels.
"""
from __future__ import annotations
import os
from collections import deque
import numpy as np
import gymnasium as gym

from config import PRESETS, EnvPreset
from env.envs_atari import NoopResetEnv, FireResetEnv, EpisodicLifeEnv
from env.norm_wrappers import NormalizeObservation

import ale_py
gym.register_envs(ale_py)


class EnvSpec:
    """Static description of the env as seen by the agent after wrapping."""
    def __init__(self, preset: EnvPreset, obs_shape, n_actions, dtype=np.float32):
        self.env_id = preset.id
        self.domain = preset.domain
        self.obs_kind = preset.obs_kind
        self.action_kind = preset.action_kind
        self.frame_skip = preset.frame_skip
        self.episodic_life = preset.episodic_life
        self.obs_shape = tuple(obs_shape)
        self.n_actions = int(n_actions)
        self.dtype = dtype


class FrameStackWrapper(gym.Wrapper):
    """Stack the last N frames as separate channels. Replaces EMA.

    Output is (H, W, N) — N copies of the grayscale frame at t, t-1, ..., t-N+1.
    This gives the conv exact positions to difference (velocity), matching the
    streaming-RL paper's proven Atari input. On reset, the first frame is
    stacked N times (no zero-padding — avoids a discontinuity delta).
    """
    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = int(num_stack)
        self.frames = deque(maxlen=self.num_stack)
        obs_shape = env.observation_space.shape
        # output: (H, W, N)
        stacked_shape = obs_shape[:-1] + (self.num_stack,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=stacked_shape, dtype=np.float32)

    def _stack(self, obs):
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim == 2:
            obs = obs[:, :, None]
        self.frames.append(obs)
        # pad cold start with copies of the first frame
        while len(self.frames) < self.num_stack:
            self.frames.appendleft(obs.copy())
        return np.concatenate(list(self.frames), axis=-1)  # (H, W, N)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._stack(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.frames.clear()
        obs, info = self.env.reset(**kwargs)
        return self._stack(obs), info


def _build_atari(preset: EnvPreset, seed: int, frame_stack: int = 4):
    """ALE Pong pipeline:
    NoopReset -> MaxAndSkip(4) -> EpisodicLife -> FireReset
    -> Resize(84) -> GrayScale -> NormalizeObservation(clip=5) -> FrameStack(4).
    No reward scaling (Pong rewards are already ±1)."""
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
    env = FrameStackWrapper(env, num_stack=frame_stack)
    obs, _ = env.reset(seed=seed)
    obs_chw = np.moveaxis(np.asarray(obs), -1, 0) if obs.ndim == 3 else obs
    spec = EnvSpec(preset, obs_chw.shape, env.action_space.n)
    return env, spec


def make_env(env_id: str, seed: int = 0, frame_stack: int = 4):
    """Build the wrapped env + EnvSpec for one preset id."""
    preset = PRESETS[env_id]
    if preset.domain == "atari":
        env, spec = _build_atari(preset, seed, frame_stack=frame_stack)
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
    """Warm up the normalizer + per-element thresholds before learning starts."""
    obs, _ = env.reset(seed=seed)
    agent.reset_episode()
    for _ in range(n_frames):
        a = env.action_space.sample()
        next_obs, r, term, trunc, info = env.step(a)
        agent.warmup_forward(next_obs)
        if term or trunc:
            agent.reset_episode()
            obs, _ = env.reset()
        else:
            obs = next_obs
    agent.reset_after_warmup()


class FrameRecorder:
    """Record frames for Stage-1 tests. NOT a replay buffer."""
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
        meta_path = os.path.splitext(path)[0] + "_done.npy"
        np.save(meta_path, np.asarray(self.done_flags, dtype=bool))
        return path

    def episode_boundaries(self):
        return [i for i, d in enumerate(self.done_flags) if d]
