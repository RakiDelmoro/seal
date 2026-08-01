"""SEAL — main training entry point.

SEAL learns and acts from frame 1. Action selection is two gates:
  - Gate 1: ε-random (adaptive from the game's ±1 reward — losing → explore more)
  - Gate 2: geometric goal exists? → imagination (40 rollouts toward s*)
            else → random

Only the transition model (A, B, b) and the inverse model (D) learn, both
from self-supervised prediction error every frame. No learned value function,
no learned policy, no eligibility traces, no TD.

Features for long runs:
  --frame-budget N         : run until N total frames are processed
  --checkpoint-interval N  : save checkpoint every N frames
  --log-path PATH          : write per-episode metrics to CSV
  --resume PATH            : resume from a checkpoint file

Usage:
  python train.py --frame-budget 100000 --checkpoint-interval 20000
  python train.py --episodes 100 --seed 0
  python train.py --resume results/seal_100k.npz --frame-budget 200000
"""
from __future__ import annotations
import argparse
import time
import numpy as np

from perception.pipeline import PerceptionPipeline
from env.pong_wrapper import PongEnv
from core.seal_core import SEALCore
from imagination.engine import ImaginationEngine
from training.success_tracker import SuccessTracker
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import MetricsLogger


def train(n_episodes: int = 100, seed: int = 0,
          frame_budget: int | None = None,
          checkpoint_interval: int | None = None,
          checkpoint_dir: str = "results",
          log_path: str | None = None,
          resume_path: str | None = None,
          verbose: bool = True):
    """Unified SEAL training: learn and act from frame 1.

    Either episode-count mode (n_episodes) or frame-budget mode
    (runs until frame_budget total frames are processed).
    """
    logger = MetricsLogger(log_path) if log_path else None

    # ── Load or initialize ──────────────────────────────────────────
    if resume_path:
        core, meta = load_checkpoint(resume_path)
        pipe = PerceptionPipeline()
        start_frames = int(meta.get("total_frames", 0))
        start_episodes = int(meta.get("episodes", 0))
        if verbose:
            print(f"Resumed from {resume_path}: {start_frames} frames, "
                  f"{start_episodes} episodes, steps={core.step_count}")
    else:
        core = SEALCore()
        pipe = PerceptionPipeline()
        start_frames = 0
        start_episodes = 0

    tracker = SuccessTracker()
    engine = ImaginationEngine()
    total_frames = start_frames

    use_frame_budget = frame_budget is not None
    if not use_frame_budget:
        frame_budget = float('inf')

    if verbose:
        print("=" * 70)
        print("SEAL — Imagination + online world-model learning")
        print("=" * 70)

    # ── Single unified loop ─────────────────────────────────────────
    ep = start_episodes
    while total_frames < frame_budget:
        if not use_frame_budget and ep >= start_episodes + n_episodes:
            break

        env = PongEnv(seed=seed + ep)
        frame, _ = env.reset()
        pipe.reset()
        s = pipe.forward(frame)[0]

        ep_reward = 0.0
        ep_len = 0
        scored = lost = 0
        done = False
        pred_err_sum = 0.0

        while not done and total_frames < frame_budget:
            # Unified action selection (gates 1-3 inside the engine)
            action, diag = engine.select_action(s, core, tracker)
            from_imagination = diag["source"] in ("greedy", "top5")

            nf, r, term, trunc, _ = env.step(action)
            done = term or trunc
            s_next = pipe.forward(nf)[0]

            # Online learning (all components, every frame)
            m = core.step_learn(s, action, s_next, r, done,
                                learned_from_imagination=from_imagination)
            pred_err_sum += m["pred_err_norm"]
            ep_reward += r
            ep_len += 1
            total_frames += 1
            if r > 0: scored += 1
            elif r < 0: lost += 1
            s = s_next

        tracker.on_episode_end(scored, lost)
        diag = core.diagnostics()

        if logger:
            logger.log_episode("train", ep, core, ep_reward, ep_len,
                               scored, lost, tracker.epsilon(),
                               pred_err_sum / max(ep_len, 1))

        if verbose and (ep % 5 == 0 or ep == start_episodes):
            print(f"  [ep {ep:3d}] frames={total_frames:6d} "
                  f"len={ep_len:4d} R={ep_reward:+5.1f} ({scored}-{lost}) "
                  f"ε={tracker.epsilon():.3f} "
                  f"‖A‖op={diag['a_op_norm']:.3f} D={diag['d_norm']:.2f} "
                  f"src={engine.source_counts()}",
                  flush=True)

        # Checkpoint
        if checkpoint_interval and total_frames % checkpoint_interval < ep_len:
            path = f"{checkpoint_dir}/seal_{total_frames//1000}k.npz"
            save_checkpoint(core, path, {
                "episodes": ep + 1, "total_frames": total_frames,
            })
            if verbose:
                print(f"  >> Checkpoint saved: {path}", flush=True)

        env.close()
        ep += 1

    # ── Final checkpoint ────────────────────────────────────────────
    final_path = f"{checkpoint_dir}/seal_final_{total_frames//1000}k.npz"
    save_checkpoint(core, final_path, {
        "episodes": ep, "total_frames": total_frames,
    })
    if verbose:
        print(f"\n  >> Final checkpoint: {final_path}", flush=True)
        print(f"  Total frames: {total_frames}")
        print(f"  Episodes: {ep - start_episodes}")
        print(f"  (Run test.py to evaluate)")

    if logger:
        logger.close()

    return {"core": core, "pipe": pipe, "total_frames": total_frames}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEAL training (single-phase)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="number of episodes (if no frame-budget)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-budget", type=int, default=None,
                        help="total frame budget (overrides episode count)")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="save checkpoint every N frames")
    parser.add_argument("--checkpoint-dir", type=str, default="results")
    parser.add_argument("--log-path", type=str, default="results/seal_metrics.csv",
                        help="CSV path for metrics (empty string to disable)")
    parser.add_argument("--resume", type=str, default=None,
                        help="resume from checkpoint .npz file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        seed=args.seed,
        frame_budget=args.frame_budget,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_path=args.log_path if args.log_path else None,
        resume_path=args.resume,
        verbose=not args.quiet,
    )
