"""SEAL environment harness.

Single entry point: make_env(preset_id, seed) -> (env, EnvSpec).

ONE normalized 84x84 grayscale frame per env step. Temporal context is
carried by the recurrent LSNN core (e-prop), not by frame stacking — so no
FrameStackWrapper. The Welford streaming normalizer is warmed up for ~1k
frames before learning starts.
"""
from __future__ import annotations
import os
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


def _build_atari(preset: EnvPreset, seed: int, render: bool = False):
    """ALE Pong pipeline:
    NoopReset -> MaxAndSkip(4) -> EpisodicLife -> FireReset
    -> Resize(84) -> GrayScale -> NormalizeObservation(clip=5).
    No frame stacking — the recurrent LSNN core carries temporal context.
    No reward scaling (Pong rewards are already ±1)."""
    render_mode = "rgb_array" if render else None
    env = gym.make(preset.id, render_mode=render_mode)
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
    obs, _ = env.reset(seed=seed)
    obs_chw = np.moveaxis(np.asarray(obs), -1, 0) if obs.ndim == 3 else obs
    spec = EnvSpec(preset, obs_chw.shape, env.action_space.n)
    return env, spec


def make_env(env_id: str, seed: int = 0, render: bool = False):
    """Build the wrapped env + EnvSpec for one preset id.

    render=True builds the env with render_mode='rgb_array' so a GUI can
    blit the frames via env.render().
    """
    preset = PRESETS[env_id]
    if preset.domain == "atari":
        env, spec = _build_atari(preset, seed, render=render)
    else:
        raise ValueError(f"Unknown domain for env {env_id}: {preset.domain}")
    return env, spec


def obs_to_chw(obs) -> np.ndarray:
    """Convert a gym obs (H,W,C) to agent-facing (C,H,W) float32."""
    obs = np.asarray(obs)
    if obs.ndim == 3 and obs.shape[-1] in (1, 3):
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


def find_norm_stats(env):
    """Walk the wrapper stack to find the streaming NormalizeObservation stats."""
    e = env
    while e is not None:
        if hasattr(e, "stats"):
            return e.stats
        e = getattr(e, "env", None)
    return None


def restore_norm_stats(env, mean, var, count):
    """Restore Welford normalization stats into the env's normalizer."""
    norm = find_norm_stats(env)
    if norm is None or mean is None:
        return
    norm.mean = np.asarray(mean, dtype=np.float64)
    norm.var = np.asarray(var, dtype=np.float64)
    norm.count = int(count)
    if norm.count > 1:
        norm._p = norm.var * (norm.count - 1)


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
