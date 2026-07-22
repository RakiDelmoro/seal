"""Stage 2 / Stage 3 acceptance tests (spec §4).

These are SMOKE tests: they run a few thousand steps and check the streaming
loop is non-degenerate (no NaN, finite returns, FLOPs accounting sane, dormant
tracking active). The full Stage-2 acceptance ("positive return within 5M
frames") and Stage-3 acceptance ("within 20% of baseline; 10-50x FLOPs lower;
dormant flat through 10M") require the long runs launched by
`python -m seal.train --stage dense/seal --frames 10000000`, whose results are
plotted by `seal.plotting`. The asserts here guard the preconditions for those
long runs to be meaningful.
"""
import os
import numpy as np
import torch

from config import config_from_preset
from env.envs import make_env
from model.agent import StreamingActorCritic


def _run(cfg, n_steps):
    env, spec = make_env(cfg.env_id, seed=0)
    agent = StreamingActorCritic(cfg, spec.n_actions)
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
            agent.learn(pending, float(r), v_next=0.0, done=True)
            returns.append(ep_ret); ep_ret = 0.0
            agent.reset_episode(); obs, _ = env.reset(); a, pending = agent.act(obs)
        else:
            a2, np_ = agent.act(next_obs)
            td = agent.learn(pending, float(r), v_next=np_.v.detach(), done=False)
            tds.append(td); flops_ev.append(agent.event_flops())
            pending = np_; a = a2
    env.close()
    return returns, tds, flops_ev, agent


def test_stage2_dense_smoke():
    cfg = config_from_preset("ALE/Pong-v5", total_frames=3000,
                             use_events=False, use_utility_gate=False,
                             use_aux=False, run_name="dense_smoke")
    returns, tds, flops_ev, agent = _run(cfg, 3000)
    assert np.all(np.isfinite(tds)), "TD errors diverged (NaN/inf)"
    assert agent.dense_flops() > 0
    # in dense mode (theta=0) event_flops is the full dense-ish compute
    assert all(f > 0 for f in flops_ev[-100:])
    print(f"  [Stage2] dense smoke: {len(returns)} episodes, "
          f"mean|td|={np.mean(np.abs(tds)):.2f}")


def test_stage3_seal_smoke():
    cfg = config_from_preset("ALE/Pong-v5", total_frames=6000,
                             use_events=True, use_utility_gate=True,
                             use_aux=True, run_name="seal_smoke")
    returns, tds, flops_ev, agent = _run(cfg, 6000)
    assert np.all(np.isfinite(tds)), "TD errors diverged (NaN/inf)"
    dense = agent.dense_flops()
    # after warmup, event flops must be BELOW dense (sparsity kicks in)
    late = np.mean(flops_ev[-500:])
    assert late < dense, f"event flops {late} >= dense {dense} after warmup"
    ratio = dense / late
    print(f"  [Stage3] seal smoke: dense/event ratio={ratio:.1f}x, "
          f"rates={[round(l.last_event_rate,3) for l in agent.encoder.event_layers]}, "
          f"dormant_frac={float((agent.since_active>cfg.dormant_silence_steps).mean()):.3f}")
    # dormant fraction should be small early (agents haven't had time to die)
    assert float((agent.since_active > cfg.dormant_silence_steps).mean()) < 0.5


def test_idbd_optimizer_runs():
    cfg = config_from_preset("ALE/Pong-v5", total_frames=500, use_events=True,
                             use_utility_gate=False, use_aux=False,
                             optimizer="idbd", run_name="idbd_smoke")
    _, tds, _, _ = _run(cfg, 500)
    assert np.all(np.isfinite(tds)), "IDBD diverged"


if __name__ == "__main__":
    test_stage2_dense_smoke()
    test_stage3_seal_smoke()
    test_idbd_optimizer_runs()
    print("Stage 2/3 smoke tests passed.")
