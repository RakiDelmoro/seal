"""Pong environment wrapper — raw rewards, general-purpose preprocessing.

Pipeline (D7-style Atari, adapted for SEAL):
  ALE/Pong-v5
    → NoopReset (1..30 no-ops on reset)
    → MaxAndSkip(4)           — 2-frame max, frame skip 4
    → EpisodicLife            — done on life loss (trace clears here)
    → FireReset               — press FIRE to start rallies
    → Resize(84×84)
    → Grayscale
    → NormalizeObservation    — streaming Welford, clip ±5
    (rewards are raw ±1 — no scaling; they steer exploration ε only)

Returns a single (84, 84) grayscale normalized frame per step. Temporal
context is carried by the perception pipeline's frame differencing, NOT by
frame stacking.

Action mapping: SEAL uses 3 abstract actions [NOOP, UP, DOWN] → ALE [0, 2, 3].
In ALE Pong: action 2 = RIGHT (paddle up), action 3 = LEFT (paddle down).
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym

import ale_py
gym.register_envs(ale_py)

from env.envs_atari import NoopResetEnv, FireResetEnv, EpisodicLifeEnv
from env.norm_wrappers import NormalizeObservation

from config import FRAME_SKIP, ENV_SEED

PONG_ID = "ALE/Pong-v5"
# SEAL abstract actions → ALE actions
ACTION_MAP = [0, 2, 3]  # [NOOP, UP, DOWN]
N_SEAL_ACTIONS = 3


class PongEnv:
    """Wrapped Pong environment producing single (84,84) normalized frames."""

    def __init__(self, seed: int = ENV_SEED, render: bool = False):
        render_mode = "rgb_array" if render else None
        env = gym.make(PONG_ID, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = gym.wrappers.MaxAndSkipObservation(env, skip=FRAME_SKIP)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = NormalizeObservation(env, clip=5.0)
        # Rewards are raw ±1 (no scaling) — they steer exploration ε only.

        self.env = env
        self.n_actions = N_SEAL_ACTIONS
        self._seed = seed

    def reset(self, seed: int | None = None):
        """Reset and return the first (84,84) frame."""
        if seed is None:
            seed = self._seed
        obs, info = self.env.reset(seed=seed)
        return self._to_frame(obs), info

    def step(self, action_idx: int):
        """Execute a SEAL action (0,1,2) → ALE action.

        Returns:
            frame: (84,84) float32 normalized grayscale.
            reward: raw ±1 or 0 (NOT scaled).
            terminated: True if life lost (EpisodicLife) or game over.
            truncated: True if time limit hit.
            info: dict (may contain episode statistics).
        """
        ale_action = ACTION_MAP[action_idx]
        obs, reward, terminated, truncated, info = self.env.step(ale_action)
        frame = self._to_frame(obs)
        # Reward is raw (no scaling) — use directly
        return frame, float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @staticmethod
    def _to_frame(obs) -> np.ndarray:
        """Convert gym obs (84,84,1) → (84,84) float32."""
        obs = np.asarray(obs)
        if obs.ndim == 3 and obs.shape[-1] == 1:
            return obs[:, :, 0].astype(np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            return obs[0].astype(np.float32)
        return obs.astype(np.float32)
