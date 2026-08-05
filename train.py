"""SEAL — main training entry point.

SEAL learns and acts from frame 1. Action selection is three gates:
  - Gate 1: ε-random (adaptive from the game's ±1 reward — losing → explore more)
  - Gate 2: π confident? → policy action (System 1)
            (20% force-imagination override keeps teaching π)
  - Gate 3: geometric goal exists? → imagination (40 rollouts toward s*)
            else → random

The transition model (A, B) and inverse model (D) learn from self-supervised
prediction error every frame. The value function V and policy π learn from the
sparse game reward via per-step streaming TD(λ) — one transition in, one
update out, no episode buffering, no Monte Carlo fallback.

Features for long runs:
  python train.py                    50M frames, checkpoint every 100k
  python train.py --resume PATH      resume from a checkpoint file
  Ctrl+C                             graceful stop — checkpoint saved on exit

Other options:
  --frame-budget N         : run until N total frames are processed
  --episodes N             : episode-count mode instead of a frame budget
  --checkpoint-interval N  : save checkpoint every N frames
  --log-path PATH          : write per-episode metrics to CSV
"""
from __future__ import annotations
import argparse
import os
import signal

# Pin math libraries to ONE thread each, BEFORE numpy/torch are imported.
# Multi-threaded BLAS spawns a thread pool per process and the processes
# thrash each other on a shared box (measured: 10-25x slowdown). One core
# per training process is faster overall. Override via env if wanted.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from perception.pipeline import PerceptionPipeline
from env.pong_wrapper import PongEnv
import core.seal_core as _seal_core_module
import core.value as _value_module
import imagination.engine as _engine_module
from core.seal_core import SEALCore
from imagination.engine import ImaginationEngine
from training.success_tracker import SuccessTracker
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import MetricsLogger

DEFAULT_FRAME_BUDGET = 50_000_000

# ── Graceful stop: Ctrl+C sets a flag; the loops exit at the next frame
#    boundary and the final checkpoint block saves all progress. ──
_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    if not _stop_requested:
        print("\n  !! Stop requested — finishing the current frame, "
              "then saving a checkpoint...", flush=True)
    _stop_requested = True


def train(n_episodes: int = 100, seed: int = 0,
          frame_budget: int | None = None,
          checkpoint_interval: int | None = None,
          checkpoint_dir: str = "results",
          log_path: str | None = None,
          resume_path: str | None = None,
          imagined_td: bool | None = None,
          rvi: bool | None = None,
          sf: bool | None = None,
          bootstrap: bool | None = None,
          verbose: bool = True):
    """Unified SEAL training: learn and act from frame 1.

    Either episode-count mode (n_episodes) or frame-budget mode
    (runs until frame_budget total frames are processed).
    imagined_td=None uses the config default; True/False overrides it
    (for A/B experiments).
    """
    if imagined_td is not None:
        # A/B override: toggles only the current-state imagined TD. The
        # from-memory variant (IMAGINED_TD_FROM_MEMORY_ENABLE) keeps its
        # config default so experiments stay single-variable.
        _seal_core_module.IMAGINED_TD_ENABLE = bool(imagined_td)
    if rvi is not None:
        # A/B override for the average-reward (RVI) critic.
        _value_module.RVI_ENABLE = bool(rvi)
    if sf is not None:
        # A/B override for the successor-feature value (V_sf).
        _seal_core_module.SF_ENABLE = bool(sf)
    logger = MetricsLogger(log_path) if log_path else None
    if bootstrap is not None:
        # A/B override for terminal-value bootstrap scoring.
        _engine_module.BOOTSTRAP_ENABLE = bool(bootstrap)

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
        budget_str = (f"{int(frame_budget):,} frames" if use_frame_budget
                      else f"{n_episodes} episodes")
        print("=" * 70)
        print("SEAL — Imagination + online world-model learning")
        print("=" * 70)
        print(f"  budget  : {budget_str}")
        print(f"  metrics : {log_path or '(disabled)'}")
        print(f"  stop    : Ctrl+C (or `python train.py --stop`) — "
              f"progress is saved on exit")
        print("=" * 70, flush=True)

    # ── Single unified loop ─────────────────────────────────────────
    ep = start_episodes
    while total_frames < frame_budget and not _stop_requested:
        if not use_frame_budget and ep >= start_episodes + n_episodes:
            break

        env = PongEnv(seed=seed + ep)
        frame, _ = env.reset()
        pipe.reset()
        s = pipe.forward(frame)

        ep_reward = 0.0
        ep_len = 0
        scored = lost = 0
        done = False
        pred_err_sum = 0.0
        td_delta_sum = 0.0
        r_err_sum = 0.0

        while not done and total_frames < frame_budget and not _stop_requested:
            # Unified action selection (gates 1-3 inside the engine)
            action, diag = engine.select_action(s, core, tracker)
            source = diag["source"]

            nf, r, term, trunc, _ = env.step(action)
            done = term or trunc
            s_next = pipe.forward(nf)

            # Online learning (all components, every frame)
            m = core.step_learn(s, action, s_next, r, done, source=source)
            pred_err_sum += m["pred_err_norm"]
            td_delta_sum += m["td_delta"]
            r_err_sum += abs(m["r_err"])
            ep_reward += r
            ep_len += 1
            total_frames += 1
            if r > 0: scored += 1
            elif r < 0: lost += 1
            s = s_next

        tracker.on_episode_end(scored, lost)
        diag = core.diagnostics()
        score_stats = engine.last_score_stats()

        if logger:
            logger.log_episode("train", ep, core, ep_reward, ep_len,
                               scored, lost, tracker.epsilon(),
                               pred_err_sum / max(ep_len, 1),
                               score_stats["score_std"],
                               td_delta_sum / max(ep_len, 1),
                               r_err_sum / max(ep_len, 1),
                               engine)

        if verbose and (ep % 5 == 0 or ep == start_episodes):
            print(f"  [ep {ep:3d}] frames={total_frames:6d} "
                  f"len={ep_len:4d} R={ep_reward:+5.1f} ({scored}-{lost}) "
                  f"ε={tracker.epsilon():.3f} "
                  f"‖A‖op={diag['a_op_norm']:.3f} D={diag['d_norm']:.2f} "
                  f"V={diag['v_norm']:.2f} π={diag['pi_norm']:.2f} "
                  f"td={m.get('td_delta', 0.0):+.3f} "
                  f"score_std={score_stats['score_std']:.2f} "
                  f"pre={diag['n_pre_score_states']} "
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
    parser = argparse.ArgumentParser(
        description="SEAL training — learn and act from frame 1. "
                    "Ctrl+C stops gracefully and saves a checkpoint.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="episode-count mode instead of a frame budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-budget", type=int, default=None,
                        help=f"total frame budget "
                             f"(default: {DEFAULT_FRAME_BUDGET:,})")
    parser.add_argument("--checkpoint-interval", type=int, default=100_000,
                        help="save checkpoint every N frames")
    parser.add_argument("--checkpoint-dir", type=str, default="results")
    parser.add_argument("--log-path", type=str, default="results/seal.csv",
                        help="CSV path for metrics (empty string to disable)")
    parser.add_argument("--resume", type=str, default=None,
                        help="resume from checkpoint .npz file")
    parser.add_argument("--imagined-td", type=str, default=None,
                        choices=["on", "off"],
                        help="override IMAGINED_TD_ENABLE for A/B runs")
    parser.add_argument("--rvi", type=str, default=None,
                        choices=["on", "off"],
                        help="override RVI_ENABLE (average-reward critic) "
                             "for A/B runs")
    parser.add_argument("--sf", type=str, default=None,
                        choices=["on", "off"],
                        help="override SF_ENABLE (successor-feature value) "
                             "for A/B runs")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--bootstrap", type=str, default=None,
                        choices=["on", "off"],
                        help="override BOOTSTRAP_ENABLE (terminal-value "
                             "scoring) for A/B runs")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # Default: the full 50M-frame budget. --episodes switches to
    # episode-count mode; --frame-budget overrides the budget.
    frame_budget = (DEFAULT_FRAME_BUDGET if args.frame_budget is None
                    else args.frame_budget)
    if args.episodes is not None:
        frame_budget = None

    train(
        n_episodes=args.episodes if args.episodes is not None else 100,
        seed=args.seed,
        frame_budget=frame_budget,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_path=args.log_path if args.log_path else None,
        resume_path=args.resume,
        imagined_td={"on": True, "off": False}.get(args.imagined_td),
        rvi={"on": True, "off": False}.get(args.rvi),
        sf={"on": True, "off": False}.get(args.sf),
        verbose=not args.quiet,
        bootstrap={"on": True, "off": False}.get(args.bootstrap),
    )
