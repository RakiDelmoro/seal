"""Metrics logger — writes training metrics to a CSV file for long runs.

Logs one row per episode with:
  phase, episode, step_count, episode_reward, episode_length,
  scored, lost, epsilon, a_op_norm, pred_err_avg,
  d_norm, rollout_norm_ratio

There is no v_norm, pi_norm, last_td_error, or b_norm column — there is no
learned value function, policy, or bias in this architecture. The CSV is flushed after every
write so it's safe to monitor with `tail -f` or read mid-training.
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
    "d_norm", "rollout_norm_ratio",
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
                    pred_err_avg: float):
        """Log one row of metrics."""
        self._ensure_open()
        diag = core.diagnostics()
        # 5-step rollout norm ratio (cheap diagnostic: is A collapsing rollouts?)
        rollout_ratio = 0.0
        if len(core.recent_states) > 0:
            s0 = list(core.recent_states)[-1]
            rollout_ratio = core._rollout_norm_ratio(s0, horizon=5)

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
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
