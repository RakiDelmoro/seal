"""Watch the agent learn Pong in real time (500k run, lambda=0.8).

This is the 500k 'is it actually learning?' experiment, with a live GUI:
  - ALE Pong frames are rendered to a Pygame window (we drive the window
    ourselves via render_mode='rgb_array', because ALE's built-in human
    renderer has an SDL quirk in this environment).
  - A side panel shows the live metrics that answer 'is it learning?':
      corrVr   (V-return correlation; the leading indicator -- want it rising)
      return   (running mean episode return; want it not worsening, later rising)
      |delta|, V, entropy, step, frame, episode.

The learning loop is IDENTICAL to seal.train (same agent, same streaming TD
updates, same ObGD). Rendering is just a view onto the same process; it does
not add a buffer or change any hard constraint.

Usage:
  python -m seal.watch --frames 500000 --seed 0
  python -m seal.watch --frames 500000 --seed 0 --fps 60     # cap display fps
  python -m seal.watch --frames 500000 --seed 0 --no-render  # headless (fast)

If corrVr stays ~0 for a long time, the agent is not learning to track value
-> stop (Ctrl-C); more frames won't help. If corrVr starts climbing, that's
the green light for the full 10M run.
"""
from __future__ import annotations
import os, time, argparse, collections
import numpy as np
import torch

import pygame

from config import config_from_preset
from env.envs import make_env, warmup
from model.agent import SEALAgent


def run(frames, seed, lam, alpha, kappa, fps_cap, render, log_every, resume_path="", ckpt_every_arg=50_000):
    torch.manual_seed(seed); np.random.seed(seed)
    # SEAL full: event-driven encoder + aux task + utility gate (always on).
    cfg = config_from_preset("ALE/Pong-v5", total_frames=frames,
                             run_name=f"seal_l{int(lam*100)}_s{seed}",
                             alpha=alpha, lam=lam, kappa=kappa)
    if render:
        import gymnasium as gym, ale_py
        gym.register_envs(ale_py)
        env = gym.make(cfg.env_id, render_mode="rgb_array")
        from env.envs_atari import NoopResetEnv, FireResetEnv, EpisodicLifeEnv
        from env.norm_wrappers import NormalizeObservation
        from env.envs import EMAWrapper
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = gym.wrappers.MaxAndSkipObservation(env, skip=4)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = NormalizeObservation(env, clip=5.0)
        env = EMAWrapper(env, alpha=cfg.ema_alpha)
        spec_obs, _ = env.reset(seed=seed)
        from env.envs import EnvSpec
        from config import PRESETS
        spec = EnvSpec(PRESETS[cfg.env_id], np.moveaxis(np.asarray(spec_obs),-1,0).shape, env.action_space.n)
    else:
        env, spec = make_env(cfg.env_id, seed=seed, ema_alpha=cfg.ema_alpha)

    agent = SEALAgent(cfg, spec.n_actions)
    agent.encoder.record_acts = True   # per-layer activation magnitude for diagnostics
    warmup(env, agent, n_frames=1000, seed=seed)
    agent.encoder.record_acts = True   # re-enable after warmup resets it via reset_episode path
    agent.reset_episode()
    obs, _ = env.reset(seed=seed)
    a, pending = agent.act(obs)

    # ---- checkpoint config ----
    # Saves: model weights, optimizer+traces, normalization stats, counters,
    # recent returns/V (so corrVr is continuous across resume). Keeps only the
    # latest + the best (highest ret20) so disk doesn't fill.
    ckpt_dir = cfg.out_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_latest = os.path.join(ckpt_dir, "seal-pong_latest.pt")
    ckpt_best   = os.path.join(ckpt_dir, "seal-pong_best.pt")
    ckpt_every  = ckpt_every_arg
    best_ret20  = -1e9

    def _find_norm_stats(env):
        e = env
        while e is not None:
            if hasattr(e, "stats"):
                return e.stats
            e = getattr(e, "env", None)
        return None

    def save_checkpoint(tag="latest"):
        path = ckpt_latest if tag == "latest" else ckpt_best
        norm = _find_norm_stats(env)
        torch.save({
            "step": t,
            "n_episodes": len(ep_returns),
            "model_state": agent.state_dict(),
            "opt_traces": [tr.detach().clone() for tr in agent.opt.traces],
            "opt_z_sum": getattr(agent.opt, "last_z_sum", 0.0),
            "opt_step_size": getattr(agent.opt, "last_step_size", 0.0),
            "since_active": agent.since_active.copy(),
            "global_step": agent.global_step,
            "norm_mean": None if norm is None else np.array(norm.mean, dtype=np.float64),
            "norm_var":  None if norm is None else np.array(norm.var,  dtype=np.float64),
            "norm_count": 0 if norm is None else norm.count,
            "ep_returns": list(ep_returns[-200:]),
            "v_at_ep": list(v_at_ep[-200:]),
            "recent_ret": list(recent_ret),
            "recent_v": list(recent_v),
            "corrVr": corrVr,
            "best_ret20": best_ret20,
            "cfg": {"lam": lam, "alpha": alpha, "kappa": kappa, "seed": seed,
                    "env_id": cfg.env_id, "lam": lam, "alpha": alpha,
                    "kappa": kappa, "seed": seed},
        }, path)
        return path

    def load_checkpoint(path):
        """Restore full state. Returns (step, n_episodes, best_ret20)."""
        nonlocal best_ret20, corrVr, ep_ret
        ck = torch.load(path, map_location="cpu")
        agent.load_state_dict(ck["model_state"])
        # restore optimizer traces
        for i, tr in enumerate(ck["opt_traces"]):
            if i < len(agent.opt.traces):
                agent.opt.traces[i].copy_(tr)
        if hasattr(agent.opt, "last_z_sum"):
            agent.opt.last_z_sum = ck.get("opt_z_sum", 0.0)
        if hasattr(agent.opt, "last_step_size"):
            agent.opt.last_step_size = ck.get("opt_step_size", 0.0)
        agent.since_active[:] = ck["since_active"]
        agent.global_step = int(ck["global_step"])
        # restore normalization stats
        norm = _find_norm_stats(env)
        if norm is not None and ck["norm_mean"] is not None:
            norm.mean = ck["norm_mean"].astype(np.float64)
            norm.var  = ck["norm_var"].astype(np.float64)
            norm.count = int(ck["norm_count"])
            # recompute the Welford helper p from mean/var/count so future
            # updates stay consistent (SampleMeanStd uses _p internally)
            if norm.count > 1:
                norm._p = norm.var * (norm.count - 1)
        # restore episode history (cap at the deques/windows)
        ep_returns.clear(); ep_returns.extend(ck["ep_returns"])
        v_at_ep.clear(); v_at_ep.extend(ck["v_at_ep"])
        recent_ret.clear(); recent_ret.extend(ck["recent_ret"])
        recent_v.clear(); recent_v.extend(ck["recent_v"])
        corrVr = float(ck.get("corrVr", 0.0))
        best_ret20 = float(ck.get("best_ret20", -1e9))
        # need a fresh forward to repopulate pending (the held transition)
        obs, _ = env.reset()
        a, pending = agent.act(obs)
        ep_ret = 0.0
        raw_ep_ret = 0.0
        print(f"[ckpt] resumed from {path}: step={ck['step']} episodes={ck['n_episodes']} "
              f"best_ret20={best_ret20:.2f} corrVr={corrVr:.3f}", flush=True)
        return int(ck["step"]), int(ck["n_episodes"]), best_ret20

    # ---- pygame window ----
    screen = None
    if render:
        pygame.init()
        # ALE frame is 210x160; scale up 3x for visibility. Side panel 320px.
        scale = 3
        game_w, game_h = 160 * scale, 210 * scale
        panel_w = 340
        screen = pygame.display.set_mode((game_w + panel_w, game_h))
        pygame.display.set_caption(f"SEAL learning Pong  (lambda={lam}, seed={seed})")
        font = pygame.font.SysFont("monospace", 16)
        font_sm = pygame.font.SysFont("monospace", 13)

    # ---- live stats ----
    ep_returns = []          # all completed episode returns (scaled)
    raw_ep_returns = []      # raw Pong scores (sum of info[raw_reward])
    v_at_ep = []             # V at the moment each episode ended
    recent_ret = collections.deque(maxlen=20)
    recent_v = collections.deque(maxlen=20)
    ep_ret = 0.0
    raw_ep_ret = 0.0
    corrVr = 0.0
    log_rows = []            # for the periodic CSV
    csv_path = os.path.join(cfg.out_dir, f"{cfg.run_name}.csv")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("step,episode,return,abs_td,v,entropy,act_med,h_mean,z_sum,a_eff,corrVr\n")

    start = time.time()
    last_log = 0
    frame_period = 1.0 / fps_cap if fps_cap else 0.0
    last_frame_time = time.time()

    t = 0
    if resume_path and os.path.exists(resume_path):
        t, _, _ = load_checkpoint(resume_path)
        start = time.time()  # reset fps clock after resume
        # CRITICAL: rebuild a fresh pending transition AFTER load_state_dict.
        # load_checkpoint restores the saved weights (bumping param version
        # counters), but the `pending` from the initial act() at the top of
        # run() was built with the random-init weights -- its graph references
        # the OLD param versions. Backprop through that stale graph crashes
        # with "modified by an inplace operation". Do a fresh forward here so
        # pending's graph references the LOADED weights.
        obs, _ = env.reset()
        a, pending = agent.act(obs)
        ep_ret = 0.0
        raw_ep_ret = 0.0

    try:
        while t < frames:
            # pygame event pump (let window close cleanly)
            if render:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt

            next_obs, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            ep_ret += float(r)
            raw_ep_ret += float(info.get("raw_reward", r))

            if done:
                agent.learn(pending, float(r), v_next=0.0, done=True)
                ep_returns.append(ep_ret)
                raw_ep_returns.append(raw_ep_ret)
                raw_run_ret = float(np.mean(raw_ep_returns[-20:])) if raw_ep_returns else 0.0
                v_at_ep.append(agent.last_v)
                recent_ret.append(ep_ret); recent_v.append(agent.last_v)
                if len(ep_returns) >= 5 and len(v_at_ep) == len(ep_returns):
                    arr_r = np.array(ep_returns[-50:]); arr_v = np.array(v_at_ep[-50:])
                    if arr_r.std() > 1e-9 and arr_v.std() > 1e-9:
                        corrVr = float(np.corrcoef(arr_v, arr_r)[0, 1])
                # ---- per-episode terminal log (diagnostic) ----
                ep_len = len(ep_returns)  # not exact length but a proxy; use counter below
                run_ret = float(np.mean(recent_ret)) if recent_ret else 0.0
                acts = agent.encoder.last_acts + [0.0]*(4-len(agent.encoder.last_acts))
                h_mean = float(np.mean(np.abs(agent.encoder.last_acts[-1] if agent.encoder.last_acts else [0.0])))
                a_eff = getattr(agent.opt, "last_step_size",0)*abs(agent.last_td_err)/max(getattr(agent.opt,"last_z_sum",1),1)
                # event-driven FLOP savings (the whole point of SEAL)
                ef = agent.event_flops(); df = agent.dense_flops()
                flop_ratio = (df / ef) if ef > 0 else 0.0
                erates = [round(l.last_event_rate,3) for l in agent.encoder.event_layers]
                if corrVr > 0.15 and run_ret > -20.5:
                    flag = "  V-tracking + policy-holding (promising)"
                elif corrVr > 0.15:
                    flag = "  V tracks return (value learning), policy not improving yet"
                elif corrVr < 0.05:
                    flag = "  corrVr flat (V not tracking -> not learning; stop if persists)"
                else:
                    flag = ""
                print(f"[EP {len(ep_returns):4d}] f={t:6d} pong={raw_ep_ret:+3.0f} ret20={raw_run_ret:+5.1f} "
                      f"corrVr={corrVr:+.3f} |d|={abs(agent.last_td_err):.3f} V={agent.last_v:+.2f} "
                      f"ent={agent.last_entropy:.3f} act={float(np.median(acts)):.2f} "
                      f"|h|={h_mean:.2f} z={getattr(agent.opt,'last_z_sum',0):.0f} "
                      f"a_eff={a_eff:.1e} "
                      f"FLOPs={flop_ratio:.1f}x rates={erates}{flag}", flush=True)
                agent.reset_episode()
                obs, _ = env.reset()
                a, pending = agent.act(obs)
                ep_ret = 0.0
                raw_ep_ret = 0.0
            else:
                a2, np_ = agent.act(next_obs)
                agent.learn(pending, float(r), v_next=agent.bootstrap(np_, False), done=False)
                pending = np_; a = a2

            t += 1

            # ---- render ----
            if render and screen is not None:
                img = env.render()   # (210,160,3) uint8
                if img is not None:
                    surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
                    surf = pygame.transform.scale(surf, (game_w, game_h))
                    screen.blit(surf, (0, 0))
                    # side panel
                    panel_x = game_w
                    screen.fill((20, 20, 25), (panel_x, 0, panel_w, game_h))
                    acts = agent.encoder.last_acts + [0.0]*(4-len(agent.encoder.last_acts))
                    h_mean = float(np.mean(np.abs(agent.encoder.last_acts[-1] if agent.encoder.last_acts else [0.0])))
                    a_eff = getattr(agent.opt, "last_step_size", 0) * abs(agent.last_td_err) / max(getattr(agent.opt,"last_z_sum",1),1)
                    run_ret = float(np.mean(recent_ret)) if recent_ret else 0.0
                    lines = [
                        (f"SEAL  (streaming + event-driven)", (220,220,220), font),
                        (f"lambda={lam} alpha={alpha} kappa={kappa}", (160,160,180), font_sm),
                        (f"", (0,0,0), font_sm),
                        (f"frame   {t}/{frames}  ({100*t/frames:.1f}%)", (200,200,200), font),
                        (f"episode {len(ep_returns)}", (200,200,200), font),
                        (f"fps     {t/(time.time()-start):.0f}", (160,160,160), font_sm),
                        (f"", (0,0,0), font_sm),
                        (f"--- IS IT LEARNING? ---", (255,220,80), font),
                        (f"corrVr  {corrVr:+.3f}   <- want rising", ( (80,255,120) if corrVr>0.15 else (255,120,120) if corrVr<0.05 else (255,220,80)), font),
                        (f"return  {run_ret:+.2f}  (last20 ep)", (200,200,200), font),
                        (f"", (0,0,0), font_sm),
                        (f"--- health ---", (180,180,200), font_sm),
                        (f"|delta| {abs(agent.last_td_err):.3f}", (170,170,170), font_sm),
                        (f"V       {agent.last_v:+.2f}", (170,170,170), font_sm),
                        (f"entropy {agent.last_entropy:.3f}", (170,170,170), font_sm),
                        (f"act_med {float(np.median(acts)):.2f}", (170,170,170), font_sm),
                        (f"|h|     {h_mean:.2f}", (170,170,170), font_sm),
                        (f"z_sum   {getattr(agent.opt,'last_z_sum',0):.0f}", (170,170,170), font_sm),
                        (f"a_eff   {a_eff:.1e}", (170,170,170), font_sm),
                    ]
                    y = 8
                    for txt, col, fnt in lines:
                        if txt:
                            screen.blit(fnt.render(txt, True, col), (panel_x + 12, y))
                        y += fnt.get_height() + 2
                    pygame.display.flip()
                # fps cap
                if frame_period:
                    dt = time.time() - last_frame_time
                    if dt < frame_period:
                        time.sleep(frame_period - dt)
                    last_frame_time = time.time()

            # ---- periodic CSV log (terminal is per-episode; CSV keeps 5k cadence) ----
            if t - last_log >= log_every:
                last_log = t
                run_ret = float(np.mean(recent_ret)) if recent_ret else 0.0
                acts = agent.encoder.last_acts + [0.0]*(4-len(agent.encoder.last_acts))
                h_mean = float(np.mean(np.abs(agent.encoder.last_acts[-1] if agent.encoder.last_acts else [0.0])))
                a_eff = getattr(agent.opt, "last_step_size",0)*abs(agent.last_td_err)/max(getattr(agent.opt,"last_z_sum",1),1)
                row = (f"{t},{len(ep_returns)},{run_ret:.3f},{abs(agent.last_td_err):.4f},"
                       f"{agent.last_v:.3f},{agent.last_entropy:.4f},{float(np.median(acts)):.3f},"
                       f"{h_mean:.3f},{getattr(agent.opt,'last_z_sum',0):.0f},{a_eff:.2e},{corrVr:.4f}")
                with open(csv_path, "a") as f:
                    f.write(row + "\n")
                # ---- checkpoint: latest every ckpt_every, best on ret20 improve ----
                if t % ckpt_every == 0:
                    p = save_checkpoint("latest")
                    cur_ret20 = float(np.mean(recent_ret)) if recent_ret else -1e9
                    if cur_ret20 > best_ret20:
                        best_ret20 = cur_ret20
                        save_checkpoint("best")
                    print(f"[ckpt] saved latest@{t} -> {os.path.basename(p)}  "
                          f"best_ret20={best_ret20:.2f}", flush=True)

    except KeyboardInterrupt:
        print(f"\nInterrupted at frame {t}. CSV saved to {csv_path}")
        # save a final checkpoint on interrupt so progress isn't lost
        p = save_checkpoint("latest")
        print(f"[ckpt] final save on interrupt -> {p}", flush=True)
    finally:
        env.close()
        if render and screen is not None:
            pygame.quit()
    # final checkpoint on normal completion too
    if t >= frames:
        p = save_checkpoint("latest")
        print(f"[ckpt] final save on completion -> {p}", flush=True)
    print(f"\nDone. CSV: {csv_path}")
    print(f"Final: episodes={len(ep_returns)} mean_return(last20)="
          f"{(float(np.mean(recent_ret)) if recent_ret else 0.0):.2f} corrVr={corrVr:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.8, help="trace decay (0.8 = paper value)")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument("--fps", type=int, default=0, help="cap display fps (0 = as fast as possible)")
    p.add_argument("--no-render", action="store_true", help="headless (no pygame window)")
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--resume", type=str, default="",
                   help="path to a seal-pong_*.pt checkpoint to resume from")
    p.add_argument("--ckpt-every", type=int, default=50_000,
                   help="save checkpoint every N frames (default 50000)")
    args = p.parse_args()
    run(args.frames, args.seed, args.lam, args.alpha, args.kappa,
        args.fps, not args.no_render, args.log_every, args.resume, args.ckpt_every)


if __name__ == "__main__":
    main()
