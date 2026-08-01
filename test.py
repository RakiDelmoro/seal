"""SEAL — evaluation script.

Plays N episodes with imagination (greedy, no exploration) and reports
performance. Used to measure how well the learned cognitive map + imagination
support planning, separate from the exploration noise of training.
"""
from __future__ import annotations
import argparse
import numpy as np

from perception.pipeline import PerceptionPipeline
from env.pong_wrapper import PongEnv
from core.seal_core import SEALCore
from imagination.engine import ImaginationEngine
from training.success_tracker import SuccessTracker
from utils.checkpoint import load_checkpoint


def evaluate(core: SEALCore, pipe: PerceptionPipeline,
             n_episodes: int = 20, seed: int = 9999,
             greedy: bool = True, verbose: bool = True) -> dict:
    """Evaluate SEAL by playing N episodes (no learning).

    Args:
        core: trained SEALCore.
        pipe: trained PerceptionPipeline.
        n_episodes: episodes to play.
        seed: env seed.
        greedy: if True, disable exploration (ε=0, always best action).
        verbose: print per-episode results.

    Returns:
        dict with per-episode rewards, scores, and summary stats.
    """
    env = PongEnv(seed=seed)
    engine = ImaginationEngine()

    # For greedy eval, use a tracker pinned to max success (ε → floor)
    tracker = SuccessTracker()
    if greedy:
        for _ in range(20):
            tracker.on_episode_end(21, 0)

    rewards = []
    scores = []

    for ep in range(n_episodes):
        frame, info = env.reset(seed=seed + ep)
        pipe.reset()
        s = pipe.forward(frame)[0]

        ep_reward = 0.0
        scored = lost = 0
        done = False

        while not done:
            action, diag = engine.select_action(s, core, tracker)
            next_frame, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            s_next = pipe.forward(next_frame)[0]

            # No learning during evaluation
            if reward > 0:
                scored += 1
            elif reward < 0:
                lost += 1
            ep_reward += reward
            s = s_next

        rewards.append(ep_reward)
        scores.append((scored, lost))

        if verbose:
            print(f"  [eval ep {ep:3d}] R={ep_reward:+5.1f} "
                  f"({scored}-{lost}) src={diag['source']}", flush=True)

    env.close()

    rewards = np.array(rewards)
    summary = {
        "rewards": rewards.tolist(),
        "scores": scores,
        "mean_reward": float(np.mean(rewards)),
        "best_reward": float(np.max(rewards)),
        "worst_reward": float(np.min(rewards)),
        "mean_scored": float(np.mean([s for s, _ in scores])),
        "mean_lost": float(np.mean([l for _, l in scores])),
    }

    if verbose:
        print(f"\n  === Evaluation Summary ({n_episodes} episodes) ===")
        print(f"  Mean reward:  {summary['mean_reward']:+.2f}")
        print(f"  Best reward:  {summary['best_reward']:+.0f}")
        print(f"  Worst reward: {summary['worst_reward']:+.0f}")
        print(f"  Mean scored:  {summary['mean_scored']:.1f}")
        print(f"  Mean lost:    {summary['mean_lost']:.1f}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEAL evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="load trained core from checkpoint .npz")
    parser.add_argument("--train-episodes", type=int, default=50,
                        help="if no checkpoint, pre-train this many episodes")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.checkpoint:
        core, meta = load_checkpoint(args.checkpoint)
        pipe = PerceptionPipeline()
        print(f"Loaded checkpoint: {args.checkpoint} "
              f"({meta.get('total_frames', '?')} frames)")
    else:
        # Quick train-then-eval
        from train import train as run_train
        print("Training...")
        result = run_train(n_episodes=args.train_episodes, seed=args.seed,
                           verbose=False)
        core = result["core"]
        pipe = result["pipe"]

    print("Evaluating...")
    evaluate(core, pipe, n_episodes=args.eval_episodes, seed=args.seed + 5000)
