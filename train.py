"""SEAL streaming training loop (spec §2.8, §5).

One sample per env step, used once, then discarded (spec §0). No replay, no
target net, no frame stacking, no minibatches. Logs one CSV row per
`log_every` steps + periodic utility histograms / feature ranks.

This is the SEAL architecture -- always event-driven, always aux+utility-gate
on. 

Usage:
  python -m seal.train --frames 10000000 --seed 0
"""
from __future__ import annotations
import os
import time
import argparse
import numpy as np
import torch

from config import config_from_preset
from env.envs import make_env, obs_to_chw, warmup
from model.agent import SEALAgent
from model.metrics import CSVLogger, feature_rank


def run(cfg, seed: int = 0, debug: bool = False):
    torch.manual_seed(seed); np.random.seed(seed)
    env, spec = make_env(cfg.env_id, seed=seed, ema_alpha=cfg.ema_alpha)
    agent = SEALAgent(cfg, n_actions=spec.n_actions, device="cpu")
    # Warm up normalizer + homeostat before any learning (so theta settles on a
    # stable normalization and the Welford stats converge before weight updates).
    warmup(env, agent, n_frames=1000, seed=seed)
    agent.reset_episode()

    os.makedirs(cfg.out_dir, exist_ok=True)
    tag = cfg.run_name
    cols = ["step", "episode", "return", "td_err", "event_flops", "dense_flops",
            "event_rate_mean", "frac_weights_updated", "dormant_frac",
            "feat_rank", "alpha_eff", "theta_mean"]
    logger = CSVLogger(os.path.join(cfg.out_dir, f"{tag}.csv"), cols)

    obs, _ = env.reset(seed=seed)
    a, pending = agent.act(obs)
    ep_return = 0.0
    raw_ep_return = 0.0
    ep_idx = 0
    start = time.time()
    last_log = 0

    for t in range(1, cfg.total_frames + 1):
        next_obs, r, term, trunc, info = env.step(a)
        done = bool(term or trunc)
        ep_return += float(r)
        raw_ep_return += float(info.get("raw_reward", r))

        if done:
            # terminal: v_next = 0, no forward wasted on the terminal frame
            td_err = agent.learn(pending, float(r), v_next=0.0, done=True)
            ep_idx += 1
            if debug and ep_idx % 10 == 0:
                fps = t / max(1e-6, time.time() - start)
                print(f"ep {ep_idx} step {t} pong={raw_ep_return:.0f} "
                      f"td {td_err:.3f} flops {agent.event_flops()} "
                      f"fps {fps:.0f}")
            _maybe_log(logger, t, ep_idx, raw_ep_return, td_err, agent, cfg)
            agent.reset_episode()
            obs, _ = env.reset()
            a, pending = agent.act(obs)
            ep_return = 0.0
            raw_ep_return = 0.0
        else:
            # forward next_obs -> next action AND bootstrap value v_next
            a_next, next_pending = agent.act(next_obs)
            td_err = agent.learn(pending, float(r),
                                 v_next=agent.bootstrap(next_pending, done=False),
                                 done=False)
            pending = next_pending
            a = a_next
            if t - last_log >= cfg.log_every:
                _maybe_log(logger, t, ep_idx, raw_ep_return, td_err, agent, cfg, running=True)
                last_log = t

    env.close()
    return os.path.join(cfg.out_dir, f"{tag}.csv")


def _theta_mean(threshold) -> float:
    """Mean theta across elements (scalar diagnostic)."""
    th = threshold.theta
    if isinstance(th, torch.Tensor):
        return round(float(th.mean().item()), 6)
    return round(float(th), 6)


def _maybe_log(logger, t, ep_idx, ep_return, td_err, agent, cfg, running=False):
    rates = agent.encoder.event_rates()
    rate_mean = float(np.mean(rates)) if rates else 0.0
    # fraction of weights updated this step (utility gate, spec §2.7)
    frac = float(np.mean([bool(u.item() > cfg.utility_tau_low)
                          for u in agent.utility.utility]))
    dormant = float((agent.since_active > cfg.dormant_silence_steps).mean())
    h_np = np.array(agent.encoder.last_acts[-1] if agent.encoder.last_acts else [0.0])
    z = getattr(agent.opt, "last_z_sum", 0.0)
    a_eff = getattr(agent.opt, "last_step_size", 0.0) * abs(td_err) / max(z, 1e-12)
    logger.log({
        "step": t, "episode": ep_idx,
        "return": round(ep_return, 3) if not running else "",
        "td_err": round(td_err, 5),
        "event_flops": agent.event_flops(),
        "dense_flops": agent.dense_flops(),
        "event_rate_mean": round(rate_mean, 5),
        "frac_weights_updated": round(frac, 4),
        "dormant_frac": round(dormant, 4),
        "feat_rank": feature_rank(h_np),
        "alpha_eff": round(a_eff, 8),
        "theta_mean": _theta_mean(agent.encoder.event_layers[0].threshold),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--env", default="ALE/Pong-v5")
    args = p.parse_args()
    cfg = config_from_preset(args.env, total_frames=args.frames,
                             run_name=f"seal_s{args.seed}")
    run(cfg, seed=args.seed, debug=args.debug)


if __name__ == "__main__":
    main()
