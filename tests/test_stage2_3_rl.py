"""Stage 2 / Stage 3 acceptance tests.

Smoke tests: run a few thousand steps and check the streaming loop is
non-degenerate (no NaN, finite TD errors, FLOPs accounting sane). The full
acceptance ("positive return within 5M frames") requires the long runs
launched by `python train.py --frames 5000000`.
"""
import numpy as np
import torch

from config import config_from_preset
from env.envs import make_env
from model.agent import SEALAgent


def _run(cfg, n_steps):
    env, spec = make_env(cfg.env_id, seed=0)
    agent = SEALAgent(cfg, spec.n_actions)
    agent.reset_episode()
    obs, _ = env.reset(seed=0)
    a, pending = agent.act(obs)
    returns, tds, flops_ev = [], [], []
    ep_ret = 0.0
    for t in range(n_steps):
        next_obs, r, term, trunc, info = env.step(a)
        done = bool(term or trunc)
        ep_ret += float(r)
        if done:
            agent.learn(pending, float(r), next_pending=None, done=True)
            returns.append(ep_ret); ep_ret = 0.0
            agent.reset_episode(); obs, _ = env.reset(); a, pending = agent.act(obs)
        else:
            a2, np_ = agent.act(next_obs)
            td = agent.learn(pending, float(r), next_pending=np_, done=False)
            tds.append(td); flops_ev.append(agent.event_flops())
            pending = np_; a = a2
    env.close()
    return returns, tds, flops_ev, agent


def test_stage2_smoke():
    cfg = config_from_preset("ALE/Pong-v5", total_frames=3000, run_name="smoke")
    returns, tds, flops_ev, agent = _run(cfg, 3000)
    assert np.all(np.isfinite(tds)), "TD errors diverged (NaN/inf)"
    assert agent.dense_flops() > 0
    assert all(f > 0 for f in flops_ev[-100:])
    print(f"  [Stage2] smoke: {len(returns)} episodes, "
          f"mean|td|={np.mean(np.abs(tds)):.2f}")


def test_stage3_seal_smoke():
    cfg = config_from_preset("ALE/Pong-v5", total_frames=6000, run_name="seal_smoke")
    returns, tds, flops_ev, agent = _run(cfg, 6000)
    assert np.all(np.isfinite(tds)), "TD errors diverged (NaN/inf)"
    dense = agent.dense_flops()
    late = np.mean(flops_ev[-500:])
    assert late < dense, f"event flops {late} >= dense {dense} after warmup"
    ratio = dense / late
    print(f"  [Stage3] seal smoke: dense/event ratio={ratio:.1f}x, "
          f"rates={[round(l.last_event_rate,3) for l in agent.encoder.event_layers]}")
    assert float((agent.since_active > cfg.dormant_silence_steps).mean()) < 0.5


if __name__ == "__main__":
    test_stage2_smoke()
    test_stage3_seal_smoke()
    print("Stage 2/3 smoke tests passed.")
