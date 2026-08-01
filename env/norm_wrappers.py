"""Streaming observation normalization (no buffer).

Online Welford estimator: keeps a few running statistics in-place and NEVER
stores raw samples, satisfying SEAL's hard constraints (online, no replay).

NormalizeObservation: per-pixel running mean/std over the stream, clipped to
±`clip`. Used by env/pong_wrapper.py. (SEAL preserves raw ±1 rewards, so there
is no reward-scaling wrapper here.)
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

    Clip output to ±`clip` to suppress early blow-up when the running variance
    is still underestimated (a bright frame after many dark ones can otherwise
    produce obs_absmax ~19 and destabilize the encoder).
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
