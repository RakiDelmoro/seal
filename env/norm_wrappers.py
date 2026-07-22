"""Streaming normalization wrappers (no buffer).

These are online Welford / discounted-trace estimators. They keep a few
running statistics in-place and NEVER store raw samples, satisfying SEAL's
hard constraints. Adapted from the streaming-RL literature (Elsayed et al.
2024) but reduced to the minimal spec-compatible form.

- NormalizeObservation: per-pixel running mean/std over the stream.
- ScaleReward: divides reward by the running std of the discounted reward
  trace r_t + gamma * trace. This is the standard streaming reward scaling
  that keeps ObGD's overshooting bound well-conditioned on raw Atari rewards
  (which are {-1,0,+1} in Pong but can drift in scale across environments).
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym


class SampleMeanStd:
    """Welford running mean/var, single pass, O(1) memory."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self._p = np.zeros(shape, "float64")
        self.count = 0

    def update(self, x):
        x = x.astype("float64")
        if self.count == 0:
            self.mean = x.copy()
            self._p = np.zeros_like(x)
        new_count = self.count + 1
        new_mean = self.mean + (x - self.mean) / new_count
        self._p = self._p + (x - self.mean) * (x - new_mean)
        self.mean = new_mean
        self.var = 1.0 if new_count < 2 else self._p / (new_count - 1)
        self.count = new_count


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize obs to ~zero mean / unit variance, online, no buffer.

    Fix #2 (diagnose findings): clip output to ±`clip` to suppress early
    blow-up when the running variance is still underestimated (a bright frame
    after many dark ones can otherwise produce obs_absmax ~19 and destabilize
    the encoder). The normalizer is also warmed up for ~1k frames before
    learning starts (see agent.warmup_forward / train.py).
    """
    def __init__(self, env: gym.Env, epsilon: float = 1e-8, clip: float = 5.0):
        super().__init__(env)
        self.stats = SampleMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon
        self.clip = float(clip) if clip is not None else None

    def observation(self, obs):
        obs = np.asarray(obs)
        self.stats.update(obs)
        out = (obs - self.stats.mean) / np.sqrt(self.stats.var + self.epsilon)
        if self.clip is not None:
            out = np.clip(out, -self.clip, self.clip)
        return out.astype(np.float32)


class ScaleReward(gym.Wrapper):
    """Scale reward by running std of the discounted reward trace.

    trace_t = gamma * trace_{t-1} * (1 - done) + r_t
    reward_scaled = r_t / sqrt(Var(trace) + eps)
    """
    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        super().__init__(env)
        self.stats = SampleMeanStd(shape=())
        self.trace = 0.0
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["raw_reward"] = float(reward)        # diagnostic only (suspect #3)
        done = bool(terminated or truncated)
        self.trace = self.trace * self.gamma * (0.0 if done else 1.0) + float(reward)
        self.stats.update(np.array(self.trace))
        scaled = float(reward) / float(np.sqrt(self.stats.var + self.epsilon))
        info["scaled_reward"] = scaled
        return obs, scaled, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
