"""SEAL training entrypoint — reward-based e-prop on an LSNN, ALE Pong.

Online, one frame per env step, sample used once and discarded. No BPTT, no
replay. Eligibility traces (forward) + neuron-specific learning signals
(symmetric feedback B_jk = Wout_kjᵀ) + reward prediction error δ implement e-prop.

Usage:
  # headless (fast)
  python train.py --frames 5000000 --seed 0

  # with live Pygame GUI
  python train.py --frames 5000000 --seed 0 --gui --fps 60

  # resume from a rotating checkpoint
  python train.py --frames 5000000 --seed 0 --resume results/seal-pong_latest.pt

Checkpoints: a rotating ring of the last --ckpt-keep (default 5) episode
checkpoints named seal-{n_episodes}.pt, saved every --ckpt-every-ep episodes.
A seal-pong_best.pt keeps the best-return snapshot.
"""
from __future__ import annotations
import os, re, time, argparse, collections
import numpy as np
import torch

from tqdm import tqdm

from config import config_from_preset
from env.envs import make_env, warmup, find_norm_stats, restore_norm_stats
from model.agent import SEALAgent
from model.metrics import CSVLogger, policy_entropy

CSV_COLUMNS = ["step", "episode", "return", "td_err", "v", "spike_rate_hz",
               "policy_entropy", "b_drift", "tag_norm_win", "tag_norm_wrec",
               "dormant_frac", "max_episode_len"]


def run(cfg, seed: int, gui: bool, fps_cap: int, resume_path: str,
        ckpt_every_ep: int, ckpt_keep: int, quiet: bool, log_every_ep: int):
    torch.manual_seed(seed); np.random.seed(seed)
    env, spec = make_env(cfg.env_id, seed=seed, render=gui)
    agent = SEALAgent(cfg, n_actions=spec.n_actions, device="cpu")
    warmup(env, agent, n_frames=cfg.warmup_frames, seed=seed)
    agent.reset_after_warmup()

    # ---- checkpoint paths ----
    ckpt_dir = cfg.out_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_best = os.path.join(ckpt_dir, "seal-pong_best.pt")
    best_ret20 = -1e9

    def _ep_num_of(name: str):
        m = re.match(r"^seal-(\d+)\.pt$", name)
        return int(m.group(1)) if m else None

    ckpt_ring = collections.deque(maxlen=ckpt_keep)
    if os.path.isdir(ckpt_dir):
        existing = []
        for f in os.listdir(ckpt_dir):
            n = _ep_num_of(f)
            if n is not None:
                existing.append((n, os.path.join(ckpt_dir, f)))
        for _n, path in sorted(existing)[:ckpt_keep]:
            ckpt_ring.append(path)

    # ---- live stats ----
    ep_returns = []
    raw_ep_returns = []
    recent_ret = collections.deque(maxlen=20)
    ep_ret = 0.0
    raw_ep_ret = 0.0
    corrVr = 0.0
    v_at_ep = []
    recent_v = collections.deque(maxlen=20)
    # ---- early critic-convergence probe (works from episode 1) ----
    # End-of-episode V should approach the terminal reward r_term, since the
    # TD target at the terminal step is r + γ·0 = r. A correct run drives
    # |mean_term_v − mean_term_r| → 0; a broken value channel leaves it stuck.
    # This catches critic-channel bugs in ~100 eps instead of ~1.5M frames.
    recent_term_v = collections.deque(maxlen=100)
    recent_term_r = collections.deque(maxlen=100)
    term_v_mean = 0.0
    term_r_mean = 0.0

    logger = CSVLogger(os.path.join(cfg.out_dir, f"{cfg.run_name}.csv"), CSV_COLUMNS)

    # ----------------------------------------------------------------- ckpt
    def _ckpt_payload():
        norm = find_norm_stats(env)
        return {
            "step": t, "n_episodes": len(ep_returns),
            "model_state": agent.state_dict(),
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
        }

    def save_ring_checkpoint():
        path = os.path.join(ckpt_dir, f"seal-{len(ep_returns)}.pt")
        if len(ckpt_ring) == ckpt_ring.maxlen:
            old = ckpt_ring.popleft()
            try: os.remove(old)
            except FileNotFoundError: pass
        torch.save(_ckpt_payload(), path)
        ckpt_ring.append(path)
        return path

    def save_best_checkpoint():
        torch.save(_ckpt_payload(), ckpt_best)
        return ckpt_best

    def load_checkpoint(path):
        nonlocal best_ret20, corrVr, ep_ret
        ck = torch.load(path, map_location="cpu")
        sd = ck["model_state"]
        model_sd = agent.state_dict()
        agent.load_state_dict(
            {k: v for k, v in sd.items()
             if k in model_sd and v.shape == model_sd[k].shape},
            strict=False)
        agent.global_step = int(ck.get("global_step", 0))
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
        pygame.display.set_caption(f"SEAL e-prop training  (seed={seed})")
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

    # ---- terminal progress bar (silent if --quiet or --gui) ----
    show_bar = (not quiet) and (not gui)
    pbar = tqdm(total=cfg.total_frames, initial=t, unit="fr",
                desc="SEAL e-prop", disable=not show_bar,
                dynamic_ncols=True, mininterval=0.5, miniters=50,
                smoothing=0.1)

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
                td_err = agent.learn(pending, float(r), next_state=None, done=True)
                ep_returns.append(ep_ret)
                raw_ep_returns.append(raw_ep_ret)
                v_at_ep.append(agent.last_v)
                recent_ret.append(ep_ret); recent_v.append(agent.last_v)
                # terminal-V convergence probe: V at the penultimate state
                # should predict the terminal reward r.
                recent_term_v.append(agent.last_v)
                recent_term_r.append(float(r))
                if recent_term_v:
                    term_v_mean = float(np.mean(recent_term_v))
                    term_r_mean = float(np.mean(recent_term_r))
                if len(ep_returns) >= 5 and len(v_at_ep) == len(ep_returns):
                    arr_r = np.array(ep_returns[-50:])
                    arr_v = np.array(v_at_ep[-50:])
                    # Only trust corrVr when returns actually SPREAD — with
                    # near-constant returns (std ~0.2 in the all-losing regime)
                    # corrVr is just noise on a 1-bit signal. Guard at 1.0.
                    if arr_r.std() > 1.0 and arr_v.std() > 1e-9:
                        corrVr = float(np.corrcoef(arr_v, arr_r)[0, 1])
                _log_episode(t, len(ep_returns), raw_ep_ret, corrVr, agent,
                             quiet or gui, log_every_ep, pbar,
                             term_v_mean, term_r_mean)
                _maybe_log(logger, t, len(ep_returns), raw_ep_ret, td_err,
                           agent, cfg, corrVr)
                if len(ep_returns) % ckpt_every_ep == 0:
                    save_ring_checkpoint()
                    cur_ret20 = float(np.mean(recent_ret)) if recent_ret else -1e9
                    if cur_ret20 > best_ret20:
                        best_ret20 = cur_ret20
                        save_best_checkpoint()
                    print(f"[ckpt] saved ep{len(ep_returns)} "
                          f"(ring={len(ckpt_ring)}/{ckpt_ring.maxlen}) "
                          f"best_ret20={best_ret20:.2f}", flush=True)
                agent.reset_episode()
                obs, _ = env.reset()
                a, pending = agent.act(obs)
                ep_ret = 0.0; raw_ep_ret = 0.0
            else:
                a_next, next_state = agent.act(next_obs)
                td_err = agent.learn(pending, float(r),
                                     next_state=next_state, done=False)
                pending = next_state; a = a_next
                if t - last_log >= cfg.log_every:
                    _maybe_log(logger, t, len(ep_returns), raw_ep_ret, td_err,
                               agent, cfg, corrVr, running=True)
                    last_log = t

            t += 1
            pbar.update(1)
            # refresh the bar's postfix with live "is it learning?" signals
            if show_bar and (t - last_log >= cfg.log_every or done):
                mean20 = float(np.mean(recent_ret)) if recent_ret else 0.0
                pbar.set_postfix({
                    "ep": len(ep_returns),
                    "ret20": f"{mean20:+.1f}",
                    "corrVr": f"{corrVr:+.2f}",
                    "Hz": f"{agent.last_spike_rate_hz:.0f}",
                }, refresh=True)
                last_log = t

            if gui and screen is not None:
                _render_gui(screen, font, font_sm, env, agent, t,
                            cfg.total_frames, len(ep_returns), corrVr,
                            recent_ret, start, game_w, game_h, panel_w)
                if frame_period:
                    dt = time.time() - last_frame_time
                    if dt < frame_period:
                        time.sleep(frame_period - dt)
                    last_frame_time = time.time()

    except KeyboardInterrupt:
        pbar.close()
        print(f"\nInterrupted at frame {t}.", flush=True)
        save_ring_checkpoint()
        print(f"[ckpt] final save on interrupt.", flush=True)
    finally:
        env.close()
        if gui and screen is not None:
            import pygame; pygame.quit()

    pbar.close()
    if t >= cfg.total_frames:
        save_ring_checkpoint()
        print(f"[ckpt] final save on completion.", flush=True)
    print(f"\nDone. episodes={len(ep_returns)} "
          f"mean_return(last20)={(float(np.mean(recent_ret)) if recent_ret else 0.0):.2f} "
          f"corrVr={corrVr:.3f}")


# ----------------------------------------------------------------- helpers
def _maybe_log(logger, t, ep_idx, ep_return, td_err, agent, cfg, corrVr,
               running=False):
    tags = agent.tag_norms()
    logger.log({
        "step": t, "episode": ep_idx,
        "return": round(ep_return, 3) if not running else "",
        "td_err": round(td_err, 5),
        "v": round(float(agent.last_v), 4),
        "spike_rate_hz": round(agent.last_spike_rate_hz, 2),
        "policy_entropy": round(agent.last_entropy, 4),
        "b_drift": round(agent.b_drift(), 5),
        "tag_norm_win": round(tags[0], 4) if tags else 0.0,
        "tag_norm_wrec": round(tags[1], 4) if len(tags) > 1 else 0.0,
        "dormant_frac": round(agent.dormant_frac(), 4),
        "max_episode_len": agent._current_max_len(),
    })


def _log_episode(t, n_eps, raw_ep_ret, corrVr, agent, silent, log_every_ep,
                 pbar, term_v_mean=0.0, term_r_mean=0.0):
    """Print one line per completed episode (throttled to every log_every_ep).

    Diagnostics:
      corrVr     — Pearson corr of V vs return over last 50 eps. Only
                   meaningful once returns SPREAD (std > 1.0); otherwise it
                   stays 0.0 and the early-stage probe below carries the load.
      termV/termR — rolling (100-ep) mean of end-of-episode V and terminal
                    reward. A correct critic drives termV → termR. A wide,
                    persistent gap signals a broken value channel (the bug
                    that stalled the previous run for ~1.5M frames).
    """
    if silent or log_every_ep <= 0:
        return
    if n_eps % log_every_ep != 0:
        return
    v_gap = term_v_mean - term_r_mean
    flag = ""
    # ---- early-stage probe (works from episode 1) ----
    # End-of-episode V should approach the terminal reward. After a short
    # warmup of the rolling window, a persistent wide gap is the fingerprint
    # of a broken critic channel — flag it loudly so it's caught in minutes.
    if n_eps > 100 and abs(v_gap) > 1.0:
        flag = (f"  ⚠ termV off-target (gap={v_gap:+.2f}; "
                f"critic not converging — check value channel)")
    # ---- late-stage probe (only meaningful once returns spread) ----
    elif corrVr > 0.15:
        flag = "  V-tracking (promising)"
    elif corrVr < -0.30:
        flag = "  ⚠ corrVr anti-correlated (sign/value-channel bug?)"
    elif corrVr < 0.05 and n_eps > 50 and corrVr != 0.0:
        flag = "  corrVr flat (not learning; stop if persists)"
    line = (f"[EP {n_eps:4d}] f={t:7d} pong={raw_ep_ret:+3.0f} "
            f"corrVr={corrVr:+.3f} δ={agent.last_td_err:+.3f} "
            f"V={agent.last_v:+.2f} Hz={agent.last_spike_rate_hz:.1f} "
            f"ent={agent.last_entropy:.2f} Bdrift={agent.b_drift():.4f} "
            f"termV={term_v_mean:+.2f} termR={term_r_mean:+.2f}{flag}")
    # tqdm.write prints above the bar without corrupting it
    try:
        pbar.write(line)
    except Exception:
        print(line, flush=True)


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
    corr_color = (80, 255, 120) if corrVr > 0.15 else \
                 (255, 120, 120) if corrVr < -0.30 else (255, 220, 80)
    lines = [
        (f"SEAL  (e-prop LSNN, adaptive)", (220, 220, 220), font),
        (f"", (0, 0, 0), font_sm),
        (f"frame   {t}/{total_frames}  ({100*t/total_frames:.1f}%)", (200, 200, 200), font),
        (f"episode {n_eps}", (200, 200, 200), font),
        (f"fps     {t/(time.time()-start):.0f}", (160, 160, 160), font_sm),
        (f"", (0, 0, 0), font_sm),
        (f"--- IS IT LEARNING? ---", (255, 220, 80), font),
        (f"corrVr  {corrVr:+.3f}   <- want rising", corr_color, font),
        (f"return  {run_ret:+.2f}  (last20 ep)", (200, 200, 200), font),
        (f"", (0, 0, 0), font_sm),
        (f"--- health ---", (180, 180, 200), font_sm),
        (f"delta   {agent.last_td_err:+.3f}", (170, 170, 170), font_sm),
        (f"V       {agent.last_v:+.2f}", (170, 170, 170), font_sm),
        (f"spike   {agent.last_spike_rate_hz:.1f} Hz", (170, 170, 170), font_sm),
        (f"entropy {agent.last_entropy:.3f}", (170, 170, 170), font_sm),
        (f"B=symmetric (Wout\u1d40)", (170, 170, 170), font_sm),
        (f"dormant {agent.dormant_frac():.2f}", (170, 170, 170), font_sm),
    ]
    y = 8
    for txt, col, fnt in lines:
        if txt:
            screen.blit(fnt.render(txt, True, col), (game_w + 12, y))
        y += fnt.get_height() + 2
    pygame.display.flip()


def main():
    p = argparse.ArgumentParser(description="SEAL training (e-prop LSNN)")
    p.add_argument("--frames", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", default="ALE/Pong-v5")
    p.add_argument("--gui", action="store_true", help="live Pygame window (disables terminal bar)")
    p.add_argument("--fps", type=int, default=0, help="cap display fps (0 = uncapped)")
    p.add_argument("--quiet", action="store_true", help="suppress terminal logging (CSV still written)")
    p.add_argument("--log-every-ep", type=int, default=1,
                   help="print an episode line every N episodes (default 1; use 5-10 for long runs)")
    p.add_argument("--resume", type=str, default="", help="checkpoint path to resume from")
    p.add_argument("--ckpt-every-ep", type=int, default=50,
                   help="checkpoint every N episodes (rotating ring)")
    p.add_argument("--ckpt-keep", type=int, default=5,
                   help="rotating checkpoints to keep (ring buffer)")
    args = p.parse_args()
    cfg = config_from_preset(args.env, total_frames=args.frames,
                             run_name=f"seal_eprop_s{args.seed}")
    run(cfg, seed=args.seed, gui=args.gui, fps_cap=args.fps,
        resume_path=args.resume, ckpt_every_ep=args.ckpt_every_ep,
        ckpt_keep=args.ckpt_keep, quiet=args.quiet, log_every_ep=args.log_every_ep)


if __name__ == "__main__":
    main()
