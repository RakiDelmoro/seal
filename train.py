"""SEAL training entrypoint.

One sample per env step, used once, then discarded. No replay, no target net,
no minibatches. Streams online: event-driven encoder + Stream Q (AdaptiveObGD)
+ SPR auxiliary representation + utility gate + dead-unit regen.

Usage:
  # headless (fast)
  python train.py --frames 5000000 --seed 0

  # with live Pygame GUI
  python train.py --frames 5000000 --seed 0 --gui --fps 60

  # resume from checkpoint
  python train.py --frames 5000000 --seed 0 --resume results/seal-pong_latest.pt
"""
from __future__ import annotations
import os, time, argparse, collections
import numpy as np
import torch

from config import config_from_preset
from env.envs import make_env, warmup, find_norm_stats, restore_norm_stats
from model.agent import SEALAgent
from model.metrics import CSVLogger, feature_rank

CSV_COLUMNS = ["step", "episode", "return", "td_err", "v",
               "event_flops", "dense_flops", "event_rate_mean",
               "dormant_frac", "feat_rank", "alpha_eff", "theta_mean", "corrVr"]


def run(cfg, seed: int, gui: bool, fps_cap: int, resume_path: str,
        ckpt_every: int, debug: bool):
    torch.manual_seed(seed); np.random.seed(seed)
    env, spec = make_env(cfg.env_id, seed=seed, frame_stack=cfg.frame_stack,
                         render=gui)
    agent = SEALAgent(cfg, n_actions=spec.n_actions, device="cpu")
    agent.encoder.record_acts = True
    warmup(env, agent, n_frames=1000, seed=seed)
    agent.encoder.record_acts = True
    agent.reset_episode()

    # ---- checkpoint paths ----
    ckpt_dir = cfg.out_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_latest = os.path.join(ckpt_dir, "seal-pong_latest.pt")
    ckpt_best = os.path.join(ckpt_dir, "seal-pong_best.pt")
    best_ret20 = -1e9

    # ---- live stats ----
    ep_returns = []
    raw_ep_returns = []
    v_at_ep = []
    recent_ret = collections.deque(maxlen=20)
    recent_v = collections.deque(maxlen=20)
    ep_ret = 0.0
    raw_ep_ret = 0.0
    corrVr = 0.0

    logger = CSVLogger(os.path.join(cfg.out_dir, f"{cfg.run_name}.csv"), CSV_COLUMNS)

    # ----------------------------------------------------------------- ckpt
    def save_checkpoint(tag="latest"):
        path = ckpt_latest if tag == "latest" else ckpt_best
        norm = find_norm_stats(env)
        torch.save({
            "step": t, "n_episodes": len(ep_returns),
            "model_state": agent.state_dict(),
            "opt_traces": [tr.detach().clone() for tr in agent.opt.traces],
            "opt_v": [vi.detach().clone() for vi in agent.opt._v],
            "opt_counter": agent.opt.counter,
            "opt_z_sum": agent.opt.last_z_sum,
            "opt_step_size": agent.opt.last_step_size,
            "target_enc": agent.target_enc.state_dict(),
            "since_active": agent.since_active.copy(),
            "global_step": agent.global_step,
            "norm_mean": None if norm is None else np.array(norm.mean, dtype=np.float64),
            "norm_var": None if norm is None else np.array(norm.var, dtype=np.float64),
            "norm_count": 0 if norm is None else norm.count,
            "ep_returns": list(ep_returns[-200:]),
            "v_at_ep": list(v_at_ep[-200:]),
            "recent_ret": list(recent_ret),
            "recent_v": list(recent_v),
            "corrVr": corrVr,
            "best_ret20": best_ret20,
            "env_id": cfg.env_id,
        }, path)
        return path

    def load_checkpoint(path):
        nonlocal best_ret20, corrVr, ep_ret
        ck = torch.load(path, map_location="cpu")
        sd = ck["model_state"]
        model_sd = agent.state_dict()
        agent.load_state_dict(
            {k: v for k, v in sd.items()
             if k in model_sd and v.shape == model_sd[k].shape},
            strict=False)
        for i, tr in enumerate(ck["opt_traces"]):
            if i < len(agent.opt.traces):
                agent.opt.traces[i].copy_(tr)
        for i, vi in enumerate(ck["opt_v"]):
            if i < len(agent.opt._v):
                agent.opt._v[i].copy_(vi)
        agent.opt.counter = int(ck.get("opt_counter", 0))
        agent.opt.last_z_sum = ck.get("opt_z_sum", 0.0)
        agent.opt.last_step_size = ck.get("opt_step_size", 0.0)
        if "target_enc" in ck:
            agent.target_enc.load_state_dict(ck["target_enc"])
        agent.since_active[:] = ck["since_active"]
        agent.global_step = int(ck["global_step"])
        restore_norm_stats(env, ck.get("norm_mean"), ck.get("norm_var"),
                           ck.get("norm_count", 0))
        ep_returns.clear(); ep_returns.extend(ck["ep_returns"])
        v_at_ep.clear(); v_at_ep.extend(ck["v_at_ep"])
        recent_ret.clear(); recent_ret.extend(ck["recent_ret"])
        recent_v.clear(); recent_v.extend(ck["recent_v"])
        corrVr = float(ck.get("corrVr", 0.0))
        best_ret20 = float(ck.get("best_ret20", -1e9))
        print(f"[ckpt] resumed from {path}: step={ck['step']} "
              f"episodes={ck['n_episodes']} best_ret20={best_ret20:.2f} "
              f"corrVr={corrVr:.3f}", flush=True)
        return int(ck["step"])

    # ----------------------------------------------------------------- gui
    screen, font, font_sm = None, None, None
    if gui:
        import pygame
        pygame.init()
        scale = 3
        game_w, game_h = 160 * scale, 210 * scale
        panel_w = 340
        screen = pygame.display.set_mode((game_w + panel_w, game_h))
        pygame.display.set_caption(f"SEAL training  (seed={seed})")
        font = pygame.font.SysFont("monospace", 16)
        font_sm = pygame.font.SysFont("monospace", 13)

    # ----------------------------------------------------------------- loop
    obs, _ = env.reset(seed=seed)
    a, pending = agent.act(obs)
    t = 0
    if resume_path and os.path.exists(resume_path):
        t = load_checkpoint(resume_path)
        obs, _ = env.reset()
        a, pending = agent.act(obs)
        ep_ret = 0.0; raw_ep_ret = 0.0

    start = time.time()
    last_log = 0
    last_frame_time = time.time()
    frame_period = 1.0 / fps_cap if fps_cap else 0.0

    try:
        while t < cfg.total_frames:
            if gui:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt

            next_obs, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            ep_ret += float(r)
            raw_ep_ret += float(info.get("raw_reward", r))

            if done:
                td_err = agent.learn(pending, float(r), next_pending=None, done=True)
                ep_returns.append(ep_ret)
                raw_ep_returns.append(raw_ep_ret)
                v_at_ep.append(agent.last_v)
                recent_ret.append(ep_ret); recent_v.append(agent.last_v)
                if len(ep_returns) >= 5 and len(v_at_ep) == len(ep_returns):
                    arr_r = np.array(ep_returns[-50:])
                    arr_v = np.array(v_at_ep[-50:])
                    if arr_r.std() > 1e-9 and arr_v.std() > 1e-9:
                        corrVr = float(np.corrcoef(arr_v, arr_r)[0, 1])
                _log_episode(t, len(ep_returns), raw_ep_ret, corrVr, agent, debug)
                _maybe_log(logger, t, len(ep_returns), raw_ep_ret, td_err,
                           agent, cfg, corrVr)
                agent.reset_episode()
                obs, _ = env.reset()
                a, pending = agent.act(obs)
                ep_ret = 0.0; raw_ep_ret = 0.0
            else:
                a_next, next_pending = agent.act(next_obs)
                td_err = agent.learn(pending, float(r),
                                     next_pending=next_pending, done=False)
                pending = next_pending; a = a_next
                if t - last_log >= cfg.log_every:
                    _maybe_log(logger, t, len(ep_returns), raw_ep_ret, td_err,
                               agent, cfg, corrVr, running=True)
                    last_log = t

            t += 1

            # ---- render ----
            if gui and screen is not None:
                _render_gui(screen, font, font_sm, env, agent, t,
                            cfg.total_frames, len(ep_returns), corrVr,
                            recent_ret, start, game_w, game_h, panel_w)
                if frame_period:
                    dt = time.time() - last_frame_time
                    if dt < frame_period:
                        time.sleep(frame_period - dt)
                    last_frame_time = time.time()

            # ---- checkpoint ----
            if t % ckpt_every == 0:
                save_checkpoint("latest")
                cur_ret20 = float(np.mean(recent_ret)) if recent_ret else -1e9
                if cur_ret20 > best_ret20:
                    best_ret20 = cur_ret20
                    save_checkpoint("best")
                print(f"[ckpt] saved @{t} best_ret20={best_ret20:.2f}", flush=True)

    except KeyboardInterrupt:
        print(f"\nInterrupted at frame {t}.", flush=True)
        save_checkpoint("latest")
        print(f"[ckpt] final save on interrupt.", flush=True)
    finally:
        env.close()
        if gui and screen is not None:
            import pygame; pygame.quit()

    if t >= cfg.total_frames:
        save_checkpoint("latest")
        print(f"[ckpt] final save on completion.", flush=True)
    print(f"\nDone. episodes={len(ep_returns)} "
          f"mean_return(last20)={(float(np.mean(recent_ret)) if recent_ret else 0.0):.2f} "
          f"corrVr={corrVr:.3f}")


# ----------------------------------------------------------------- helpers
def _theta_mean(threshold) -> float:
    th = threshold.theta
    if isinstance(th, torch.Tensor):
        return round(float(th.mean().item()), 6)
    return round(float(th), 6)


def _maybe_log(logger, t, ep_idx, ep_return, td_err, agent, cfg, corrVr,
               running=False):
    rates = agent.encoder.event_rates()
    rate_mean = float(np.mean(rates)) if rates else 0.0
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
        "v": round(float(agent.last_v), 4),
        "event_flops": agent.event_flops(),
        "dense_flops": agent.dense_flops(),
        "event_rate_mean": round(rate_mean, 5),
        "dormant_frac": round(dormant, 4),
        "feat_rank": feature_rank(h_np),
        "alpha_eff": round(a_eff, 8),
        "theta_mean": _theta_mean(agent.encoder.event_layers[0].threshold),
        "corrVr": round(corrVr, 4),
    })


def _log_episode(t, n_eps, raw_ep_ret, corrVr, agent, debug):
    if not debug:
        return
    flag = ""
    if corrVr > 0.15:
        flag = "  V-tracking (promising)"
    elif corrVr < 0.05 and n_eps > 50:
        flag = "  corrVr flat (not learning; stop if persists)"
    mode = "greedy" if agent.epsilon < 0.05 else ""
    print(f"[EP {n_eps:4d}] f={t:6d} pong={raw_ep_ret:+3.0f} "
          f"ε={agent.epsilon:.3f} corrVr={corrVr:+.3f} "
          f"|d|={abs(agent.last_td_err):.3f} "
          f"V={agent.last_v:+.2f} z={getattr(agent.opt,'last_z_sum',0):.0f}{flag}"
          + (f"  ← POLICY MODE" if mode else ""),
          flush=True)


def _render_gui(screen, font, font_sm, env, agent, t, total_frames,
                n_eps, corrVr, recent_ret, start, game_w, game_h, panel_w):
    import pygame
    img = env.render()
    if img is None:
        return
    surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    surf = pygame.transform.scale(surf, (game_w, game_h))
    screen.blit(surf, (0, 0))
    screen.fill((20, 20, 25), (game_w, 0, panel_w, game_h))
    run_ret = float(np.mean(recent_ret)) if recent_ret else 0.0
    a_eff = getattr(agent.opt, "last_step_size", 0) * abs(agent.last_td_err) / \
            max(getattr(agent.opt, "last_z_sum", 1), 1)
    ef = agent.event_flops(); df = agent.dense_flops()
    flop_ratio = (df / ef) if ef > 0 else 0.0
    corr_color = (80, 255, 120) if corrVr > 0.15 else \
                 (255, 120, 120) if corrVr < 0.05 else (255, 220, 80)
    lines = [
        (f"SEAL  (streaming + event-driven)", (220, 220, 220), font),
        (f"", (0, 0, 0), font_sm),
        (f"frame   {t}/{total_frames}  ({100*t/total_frames:.1f}%)", (200, 200, 200), font),
        (f"episode {n_eps}", (200, 200, 200), font),
        (f"fps     {t/(time.time()-start):.0f}", (160, 160, 160), font_sm),
        (f"", (0, 0, 0), font_sm),
        (f"--- IS IT LEARNING? ---", (255, 220, 80), font),
        (f"corrVr  {corrVr:+.3f}   <- want rising", corr_color, font),
        (f"return  {run_ret:+.2f}  (last20 ep)", (200, 200, 200), font),
        (f"FLOPs   {flop_ratio:.1f}x savings", (160, 200, 160), font_sm),
        (f"", (0, 0, 0), font_sm),
        (f"--- health ---", (180, 180, 200), font_sm),
        (f"|delta| {abs(agent.last_td_err):.3f}", (170, 170, 170), font_sm),
        (f"V       {agent.last_v:+.2f}", (170, 170, 170), font_sm),
        (f"z_sum   {getattr(agent.opt,'last_z_sum',0):.0f}", (170, 170, 170), font_sm),
        (f"a_eff   {a_eff:.1e}", (170, 170, 170), font_sm),
        (f"eps     {agent.epsilon:.3f}", (170, 170, 170), font_sm),
    ]
    y = 8
    for txt, col, fnt in lines:
        if txt:
            screen.blit(fnt.render(txt, True, col), (game_w + 12, y))
        y += fnt.get_height() + 2
    pygame.display.flip()


def main():
    p = argparse.ArgumentParser(description="SEAL training (streaming, event-driven)")
    p.add_argument("--frames", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", default="ALE/Pong-v5")
    p.add_argument("--gui", action="store_true", help="live Pygame window")
    p.add_argument("--fps", type=int, default=0, help="cap display fps (0 = uncapped)")
    p.add_argument("--debug", action="store_true", help="per-episode console log")
    p.add_argument("--resume", type=str, default="", help="checkpoint path to resume from")
    p.add_argument("--ckpt-every", type=int, default=50_000, help="checkpoint interval (frames)")
    args = p.parse_args()
    cfg = config_from_preset(args.env, total_frames=args.frames,
                             run_name=f"seal_s{args.seed}")
    run(cfg, seed=args.seed, gui=args.gui, fps_cap=args.fps,
        resume_path=args.resume, ckpt_every=args.ckpt_every, debug=args.debug)


if __name__ == "__main__":
    main()
