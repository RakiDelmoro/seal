"""Metrics logger — writes training metrics to a CSV file for long runs.

Logs one row per episode with:
  phase, episode, step_count, episode_reward, episode_length,
  scored, lost, epsilon, a_op_norm, pred_err_avg,
  d_norm, rollout_norm_ratio, score_std, v_norm, pi_norm, td_delta_avg,
  source_epsilon, source_policy, source_imagination, source_no_goal, source_random

The CSV is flushed after every write so it's safe to monitor with `tail -f`.
"""
from __future__ import annotations
import os
import csv
import time
import numpy as np

from core.seal_core import SEALCore

CSV_FIELDS = [
    "timestamp", "phase", "episode", "step_count",
    "episode_reward", "episode_length", "scored", "lost",
    "epsilon", "a_op_norm", "pred_err_avg",
    "d_norm", "rollout_norm_ratio", "score_std",
    "v_norm", "pi_norm", "td_delta_avg",
    "src_epsilon", "src_policy", "src_imagination", "src_no_goal", "src_random",
]


class MetricsLogger:
    """CSV logger for SEAL training metrics."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._first_write = not os.path.exists(path)
        self._file = None
        self._writer = None

    def _ensure_open(self):
        if self._file is None:
            self._file = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
            if self._first_write:
                self._writer.writeheader()
                self._first_write = False

    def log_episode(self, phase: str, episode: int, core: SEALCore,
                    episode_reward: float, episode_length: int,
                    scored: int, lost: int, epsilon: float,
                    pred_err_avg: float, score_std: float = 0.0,
                    td_delta_avg: float = 0.0, engine=None):
        """Log one row of metrics."""
        self._ensure_open()
        diag = core.diagnostics()
        # 5-step rollout norm ratio (cheap diagnostic: is A collapsing rollouts?)
        rollout_ratio = 0.0
        if len(core.recent_states) > 0:
            s0 = list(core.recent_states)[-1]
            rollout_ratio = core._rollout_norm_ratio(s0, horizon=5)

        src = engine.source_counts() if engine else {}

        row = {
            "timestamp": f"{time.time():.0f}",
            "phase": phase,
            "episode": episode,
            "step_count": core.step_count,
            "episode_reward": f"{episode_reward:+.0f}",
            "episode_length": episode_length,
            "scored": scored,
            "lost": lost,
            "epsilon": f"{epsilon:.4f}",
            "a_op_norm": f"{diag['a_op_norm']:.4f}",
            "pred_err_avg": f"{pred_err_avg:.4f}",
            "d_norm": f"{diag['d_norm']:.4f}",
            "rollout_norm_ratio": f"{rollout_ratio:.4f}",
            "score_std": f"{score_std:.4f}",
            "v_norm": f"{diag['v_norm']:.4f}",
            "pi_norm": f"{diag['pi_norm']:.4f}",
            "td_delta_avg": f"{td_delta_avg:.4f}",
            "src_epsilon": src.get("epsilon", 0),
            "src_policy": src.get("policy", 0),
            "src_imagination": src.get("imagination", 0),
            "src_no_goal": src.get("no_goal", 0),
            "src_random": src.get("random", 0),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
