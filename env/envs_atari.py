"""Vendored Atari wrappers (D7).

Gymnasium 1.3 already provides MaxAndSkipObservation (with the 2-frame
elementwise max that removes Atari flicker), ResizeObservation,
GrayscaleObservation, and NormalizeObservation. We vendor only the three
wrappers gymnasium lacks: NoopResetEnv, FireResetEnv, EpisodicLifeEnv.

These are adapted from stable_baselines3.common.atari_wrappers (MIT). Kept
here so SEAL's runtime deps stay torch + gymnasium + ale-py (no SB3).

All wrappers are buffer-free and compatible with SEAL's hard constraints
(no replay, no frame stacking, batch-dim-1). MaxAndSkip is a 1-step max-pool
over the two most recent frames producing a *single* observation -- it is
NOT frame stacking and NOT a replay buffer.
"""
from __future__ import annotations
import gymnasium as gym


class NoopResetEnv(gym.Wrapper):
    """Sample initial state by taking 1..noop_max no-ops on reset."""
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        assert noop_max > 0
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP", \
            "Start action must be NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = int(self.unwrapped.np_random.integers(1, self.noop_max + 1))
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Take FIRE on reset for environments fixed until firing (Pong needs it)."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """End episode (report done) when a life is lost; reset on true done.

    This interacts with SEAL's step loop: caches reset on
    `done` (spec §2.8). With episodic_life=True, that happens on each life loss
    (~5x per Pong episode). Toggle via config.episodic_life (D3).
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True        # life lost => report done
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)  # no-op to advance life
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info
