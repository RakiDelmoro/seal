"""SEAL inference entrypoint — run a trained agent greedily, no learning.

Loads a checkpoint produced by train.py, restores the model weights + the
streaming observation normalizer, sets ε=0 (pure greedy), and plays the game
with a live Pygame GUI. No learning, no CSV, no checkpointing — just watch it
play.

Usage:
  python play.py --checkpoint results/seal-pong_best.pt
  python play.py --checkpoint results/seal-pong_best.pt --episodes 20
  python play.py --checkpoint results/seal-pong_best.pt --fps 30
"""
from __future__ import annotations
import os, time, argparse, collections
import numpy as np
import torch

from config import config_from_preset
from env.envs import make_env, warmup, find_norm_stats, restore_norm_stats
from model.agent import SEALAgent


def play(checkpoint_path: str, env_id: str, episodes: int, fps_cap: int,
         seed: int):
    import pygame

    ck = torch.load(checkpoint_path, map_location="cpu")
    env_id = ck.get("env_id", env_id)
    cfg = config_from_preset(env_id, total_frames=1, run_name="play")

    env, spec = make_env(env_id, seed=seed, frame_stack=cfg.frame_stack,
                         render=True)
    agent = SEALAgent(cfg, n_actions=spec.n_actions, device="cpu")

    # restore model weights (drop keys that don't exist or whose lazy cache
    # shapes don't match yet — x_prev/out_prev are episode caches that get
    # populated on the first forward pass, not needed from the checkpoint)
    sd = ck["model_state"]
    model_sd = agent.state_dict()
    filtered = {}
    for k, v in sd.items():
        if k not in model_sd:
            continue
        if v.shape != model_sd[k].shape:
            continue
        filtered[k] = v
    agent.load_state_dict(filtered, strict=False)
    # restore the observation normalizer so inputs match what the agent trained on
    restore_norm_stats(env, ck.get("norm_mean"), ck.get("norm_var"),
                       ck.get("norm_count", 0))

    # pure greedy — no exploration, no learning
    agent.epsilon = 0.0
    agent.eval()
    agent.last_v = 0.0

    # ---- GUI ----
    pygame.init()
    scale = 3
    game_w, game_h = 160 * scale, 210 * scale
    panel_w = 340
    screen = pygame.display.set_mode((game_w + panel_w, game_h))
    pygame.display.set_caption(f"SEAL playing {env_id}  (greedy)")
    font = pygame.font.SysFont("monospace", 16)
    font_sm = pygame.font.SysFont("monospace", 13)
    frame_period = 1.0 / fps_cap if fps_cap else 0.0
    last_frame_time = time.time()

    scores = collections.deque(maxlen=20)
    ep = 0

    try:
        while ep < episodes:
            agent.reset_episode()
            obs, _ = env.reset(seed=seed + ep)
            a, tr = agent.act(obs)
            ep_ret = 0.0
            done = False
            agent.last_v = float(tr.logits[0, a].item())

            while not done:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt

                next_obs, r, term, trunc, info = env.step(a)
                done = bool(term or trunc)
                ep_ret += float(info.get("raw_reward", r))
                if not done:
                    a, tr = agent.act(next_obs)
                    agent.last_v = float(tr.logits[0, a].item())

                # render
                img = env.render()
                if img is not None:
                    surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
                    surf = pygame.transform.scale(surf, (game_w, game_h))
                    screen.blit(surf, (0, 0))
                    screen.fill((20, 20, 25), (game_w, 0, panel_w, game_h))
                    mean_score = float(np.mean(scores)) if scores else 0.0
                    lines = [
                        (f"SEAL playing  (greedy, no learning)", (220, 220, 220), font),
                        (f"", (0, 0, 0), font_sm),
                        (f"episode  {ep + 1}/{episodes}", (200, 200, 200), font),
                        (f"score    {ep_ret:+.0f}", (200, 200, 200), font),
                        (f"mean20   {mean_score:+.2f}", (200, 200, 200), font),
                        (f"played   {len(scores)} eps", (160, 160, 160), font_sm),
                        (f"", (0, 0, 0), font_sm),
                        (f"Q(s,a)   {agent.last_v:+.2f}", (170, 170, 170), font_sm),
                    ]
                    y = 8
                    for txt, col, fnt in lines:
                        if txt:
                            screen.blit(fnt.render(txt, True, col), (game_w + 12, y))
                        y += fnt.get_height() + 2
                    pygame.display.flip()

                if frame_period:
                    dt = time.time() - last_frame_time
                    if dt < frame_period:
                        time.sleep(frame_period - dt)
                    last_frame_time = time.time()

            ep += 1
            scores.append(ep_ret)
            print(f"[EP {ep:3d}] score={ep_ret:+.0f}  mean20={float(np.mean(scores)):+.2f}",
                  flush=True)

    except KeyboardInterrupt:
        print(f"\nStopped after {ep} episodes.")
    finally:
        env.close()
        pygame.quit()

    print(f"\nDone. {ep} episodes. mean score (last {len(scores)}) = "
          f"{(float(np.mean(scores)) if scores else 0.0):+.2f}")


def main():
    p = argparse.ArgumentParser(description="SEAL inference — play from a checkpoint")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="path to a seal-pong_*.pt checkpoint")
    p.add_argument("--env", type=str, default="ALE/Pong-v5",
                   help="env id (overridden by checkpoint if present)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--fps", type=int, default=30, help="display fps cap")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    play(args.checkpoint, args.env, args.episodes, args.fps, args.seed)


if __name__ == "__main__":
    main()
