"""Streaming observation normalization (no buffer).

Online Welford estimator: keeps a few running statistics in-place and NEVER
stores raw samples, satisfying SEAL's hard constraints.

NormalizeObservation: per-pixel running mean/std over the stream, clipped to
±`clip`. Warmed up for ~1k frames before learning starts (see agent.warmup).

ScaleReward: divide rewards by the running std of the DISCOUNTED RETURN TRACE
(stream-x ingredient, Elsayed et al. 2024). Shrinks return magnitudes so the
critic fits at small weights and δ becomes reward-dominated rather than
critic-noise-dominated (observed failure: V oscillating ±40 on a ±21 game).
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


class ScaleReward(gym.Wrapper):
    """Scale rewards by the running std of the discounted return trace.

    trace_t = γ·trace_{t-1}·(1−done) + r_t ;  r_scaled = r / std(trace)

    The trace approximates the discounted return, so rewards are normalized
    by the scale of the RETURNS the critic must represent — the critic then
    fits at small weight magnitudes instead of drifting to large ones.

    Deviation from stream-x: the scale is floored at 1.0. Pong rewards are
    ±1 sparse; before the trace variance grows (first rallies), dividing by
    a near-zero std would blow rewards up to ±10⁴. The floor means rewards
    pass through unscaled until return variance is established.
    """
    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        super().__init__(env)
        self.reward_stats = SampleMeanStd(shape=())
        self.reward_trace = 0.0
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        done = float(term or trunc)
        self.reward_trace = self.reward_trace * self.gamma * (1.0 - done) + r
        self.reward_stats.update(np.asarray(self.reward_trace))
        scale = max(float(np.sqrt(self.reward_stats.var + self.epsilon)), 1.0)
        # expose the RAW reward so logging/model-selection stays in true game
        # points (train.py/play.py read info["raw_reward"]); without this the
        # scaled reward would masquerade as learning progress
        info = dict(info)
        info["raw_reward"] = r
        return obs, r / scale, term, trunc, info

    def reset(self, **kwargs):
        self.reward_trace = 0.0
        return self.env.reset(**kwargs)


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
