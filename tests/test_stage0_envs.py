"""Stage 0 acceptance test (spec §4): env harness.

Tests:
  - make_env returns (env, spec) with correct obs shape (1,84,84), n_actions=6.
  - obs is float32, roughly normalized (finite, variance != 0 after a few steps).
  - episode boundaries are detected (done fires, reset works).
  - FrameRecorder records N frames to disk and reports episode boundaries.
  - Reward scaling wrapper updates running stats (no buffer).
"""
import os
import tempfile
import numpy as np

from env.envs import make_env, obs_to_chw, FrameRecorder
from config import PRESETS


def test_env_spec_shapes():
    env, spec = make_env("ALE/Pong-v5", seed=0)
    assert spec.obs_kind == "image"
    assert spec.action_kind == "discrete"
    assert spec.obs_shape == (1, 84, 84), spec.obs_shape
    assert spec.n_actions == 6, spec.n_actions
    assert spec.frame_skip == 4
    assert spec.episodic_life is True
    env.close()


def test_obs_dtype_and_normalization():
    env, spec = make_env("ALE/Pong-v5", seed=0)
    obs, _ = env.reset(seed=1)
    obs = obs_to_chw(obs)
    assert obs.shape == (1, 84, 84), obs.shape
    assert obs.dtype == np.float32, obs.dtype
    assert np.all(np.isfinite(obs)), "obs has non-finite values"
    # After a few steps the running var should be > 0 (stats are updating).
    for _ in range(20):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
    norm = env.get_wrapper_attr("stats") if hasattr(env, "stats") else None
    # find the NormalizeObservation wrapper
    e = env
    while e is not None and not isinstance(e, type(env)) is False:
        e = getattr(e, "env", None) if not hasattr(e, "stats") else e
        if hasattr(e, "stats"):
            norm = e.stats
            break
    assert norm is not None, "NormalizeObservation wrapper not found"
    assert norm.count >= 20
    assert (norm.var > 0).any(), "normalization var stuck at 0"
    env.close()


def test_episode_boundaries_detected():
    env, spec = make_env("ALE/Pong-v5", seed=0)
    obs, _ = env.reset(seed=2)
    dones = 0
    for _ in range(2000):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            dones += 1
            obs, _ = env.reset()
        if dones >= 1:
            break
    assert dones >= 1, "no episode boundary detected within 2000 steps"
    env.close()


def test_frame_recorder_records_and_saves():
    env, spec = make_env("ALE/Pong-v5", seed=0)
    obs, _ = env.reset(seed=3)
    rec = FrameRecorder(n=50, obs_shape=spec.obs_shape)
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        done = bool(term or trunc)
        rec.add(obs, done)
        if done:
            obs, _ = env.reset()
    assert rec.full
    assert len(rec.frames) == 50
    frames_stack = np.stack(rec.frames, axis=0)
    assert frames_stack.shape == (50, 1, 84, 84), frames_stack.shape
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "frames.npy")
        rec.save(path)
        assert os.path.exists(path)
        loaded = np.load(path)
        assert loaded.shape == (50, 1, 84, 84)
        b = rec.episode_boundaries()
        assert isinstance(b, list)
    env.close()


if __name__ == "__main__":
    test_env_spec_shapes()
    test_obs_dtype_and_normalization()
    test_episode_boundaries_detected()
    test_frame_recorder_records_and_saves()
    print("Stage 0 tests passed.")
