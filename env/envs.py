"""SEAL environment harness.

Single entry point: make_env(preset_id, seed) -> (env, EnvSpec).

EMA temporal wrapper replaces frame stacking. The EMA maintains a single
accumulated frame that captures recent motion as a smooth trail — aligned with
the event-driven encoder's own temporal accumulation (out_prev). This is
simpler (1 channel vs 4), accumulates fewer eligibility traces (4× slower
z_sum growth), and produces cleaner event deltas (1 smooth channel vs 4
mostly-empty channels).
"""
from __future__ import annotations
import os
import numpy as np
import gymnasium as gym

from config import PRESETS, EnvPreset
from env.envs_atari import NoopResetEnv, FireResetEnv, EpisodicLifeEnv
from env.norm_wrappers import NormalizeObservation, ScaleReward

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


class EMAWrapper(gym.Wrapper):
    """Exponential moving average of frames. Replaces frame stacking.

    Maintains a single accumulated frame:
        ema_t = alpha * frame_t + (1 - alpha) * ema_{t-1}

    This captures recent motion as a smooth trail (the ball appears as a
    streak whose length encodes velocity). Aligned with the event-driven
    encoder's own temporal accumulation (out_prev = running state updated
    incrementally). Simpler than frame stacking: 1 channel, 4× fewer
    parameters in the first conv, 4× slower trace accumulation.

    alpha=0.3 gives ~3 frames of effective memory (geometric decay:
    0.3 + 0.7*0.3 + 0.7^2*0.3 ≈ 1.0).
    """
    def __init__(self, env: gym.Env, alpha: float = 0.3):
        super().__init__(env)
        self.alpha = float(alpha)
        self.ema = None
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    def _update(self, obs):
        obs = np.array(obs, dtype=np.float32)
        if self.ema is None:
            self.ema = obs.copy()
        else:
            self.ema = self.alpha * obs + (1.0 - self.alpha) * self.ema
        return self.ema

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._update(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.ema = None
        obs, info = self.env.reset(**kwargs)
        return self._update(obs), info


class MultiScaleEMAWrapper(gym.Wrapper):
    """Multi-scale EMA with lagged stacking (replaces single EMA + frame stacking).

    Maintains N EMAs at different alphas (timescales). For each, keeps L lagged
    copies (t, t-1, ..., t-L+1). Output is N×L channels, ordered EMA-first:
        [ema0_t, ema0_{t-1}, ..., ema0_{t-L+1}, ema1_t, ema1_{t-1}, ...]

    This gives the conv access to:
      - position at each timescale (the EMA values themselves)
      - velocity at each timescale (differences between adjacent lags)
      - multi-scale temporal context (short trails = precise position, long
        trails = trajectory context)

    Why this over single EMA: a single EMA smears the ball into one streak;
    the conv can't easily extract precise position AND velocity. Multi-scale +
    lags gives explicit velocity (lag differences) at multiple timescales in
    one input — like frame stacking but with smooth, event-friendly trails.

    Why this over frame stacking: smooth EMA trails produce sparser deltas than
    raw frames (better for the event encoder), and the multi-scale structure
    captures more temporal context than a fixed 4-frame window.

    Effective memory of each EMA ≈ (1-alpha)/alpha frames:
      alpha=0.5  → ~1 frame  (sharp current position)
      alpha=0.25 → ~3 frames  (short trail)
      alpha=0.125→ ~7 frames  (medium trail)
      alpha=0.0625→ ~15 frames (long trajectory context)
    """
    def __init__(self, env: gym.Env, alphas, lags: int = 3):
        super().__init__(env)
        self.alphas = [float(a) for a in alphas]
        self.lags = int(lags)
        self.n_ema = len(self.alphas)
        self._emas = [None] * self.n_ema       # current EMA per timescale
        import collections
        self._hist = collections.deque(maxlen=self.lags)  # list of [N,H,W] snapshots
        # output shape: (H, W, N*L)
        obs_shape = env.observation_space.shape  # (H, W, 1) grayscale
        H, W = obs_shape[0], obs_shape[1]
        out_ch = self.n_ema * self.lags
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(H, W, out_ch), dtype=np.float32)

    def _update(self, obs):
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim == 2:
            obs = obs[:, :, None]
        # update each EMA
        snap = np.zeros((self.n_ema, obs.shape[0], obs.shape[1]), dtype=np.float32)
        for i, a in enumerate(self.alphas):
            if self._emas[i] is None:
                self._emas[i] = obs[:, :, 0].copy()
            else:
                self._emas[i] = a * obs[:, :, 0] + (1.0 - a) * self._emas[i]
            snap[i] = self._emas[i]
        self._hist.append(snap)
        # build output: for each EMA, the L lags (pad cold start with earliest)
        channels = []
        for i in range(self.n_ema):
            for j in range(self.lags):
                # j=0 is newest, j=lags-1 is oldest
                idx = len(self._hist) - 1 - j
                if idx < 0:
                    idx = 0  # pad cold start
                channels.append(self._hist[idx][i])
        out = np.stack(channels, axis=-1)  # (H, W, N*L)
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._update(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._emas = [None] * self.n_ema
        self._hist.clear()
        obs, info = self.env.reset(**kwargs)
        return self._update(obs), info


def _build_atari(preset: EnvPreset, seed: int, ema_alpha: float = 0.3,
                 scale_reward: bool = True, ema_alphas=None, ema_lags: int = 1):
    """ALE Pong pipeline: ... -> EMA.

    If ema_alphas is given (multi-scale), uses MultiScaleEMAWrapper (N EMAs ×
    L lags = N*L channels). Otherwise uses single EMAWrapper (1 channel).
    ScaleReward applied if scale_reward=True.
    """
    from env.norm_wrappers import NormalizeObservation, ScaleReward
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
    if scale_reward:
        env = ScaleReward(env, gamma=0.99)
    if ema_alphas is not None and len(ema_alphas) > 1 or (ema_alphas is not None and ema_lags > 1):
        env = MultiScaleEMAWrapper(env, alphas=ema_alphas, lags=ema_lags)
    else:
        env = EMAWrapper(env, alpha=ema_alpha)
    obs, _ = env.reset(seed=seed)
    obs_chw = np.moveaxis(np.asarray(obs), -1, 0) if obs.ndim == 3 else obs
    spec = EnvSpec(preset, obs_chw.shape, env.action_space.n)
    return env, spec


def make_env(env_id: str, seed: int = 0, ema_alpha: float = 0.3,
            scale_reward: bool = True, ema_alphas=None, ema_lags: int = 1):
    """Build the wrapped env + EnvSpec for one preset id."""
    preset = PRESETS[env_id]
    if preset.domain == "atari":
        env, spec = _build_atari(preset, seed, ema_alpha, scale_reward=scale_reward,
                                 ema_alphas=ema_alphas, ema_lags=ema_lags)
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
    """Warm up the normalizer + homeostat before learning starts."""
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
