"""SEAL metrics (spec §5): FLOP accounting, plasticity counters, CSV logger.

Analytics only -- never stores samples. Everything is computed from the
current single sample and a few running counters. (Auxiliary prediction
targets now live in model/gvf.py as the game-agnostic GVF bank.)
"""
from __future__ import annotations
import numpy as np
import torch


def flops_event_layers(layers) -> int:
    """Sum of analytic FLOPs across event layers (spec §0, §5)."""
    return sum(int(l.flops()) for l in layers)


def dense_flops_conv(in_ch, out_ch, k, stride, H, W) -> int:
    """Full dense conv FLOPs (for FLOP comparison)."""
    out_h = (H - k) // stride + 1
    out_w = (W - k) // stride + 1
    return out_h * out_w * k * k * in_ch * out_ch * 2


def dense_flops_linear(in_f, out_f) -> int:
    return in_f * out_f * 2


# ---------------------------------------------------------------------------
# Streaming plasticity metrics (spec §5): dormant fraction, feature rank.
# These keep tiny per-unit counters (not samples).
# ---------------------------------------------------------------------------
class DormantTracker:
    """Tracks per-unit silence time. A unit is dormant if silent >
    dormant_silence_steps (spec §5: >10k steps). 'Silent' for a trunk unit =
    |activation| below a small eps. Maintained online, batch-1."""
    def __init__(self, n_units: int, silence_steps: int = 10_000, eps: float = 1e-3):
        self.since_active = np.zeros(n_units, dtype=np.int64)
        self.silence_steps = int(silence_steps)
        self.eps = float(eps)

    def update(self, acts: np.ndarray):  # acts: [n_units]
        active = np.abs(acts) > self.eps
        self.since_active[active] = 0
        self.since_active[~active] += 1

    @property
    def dormant_fraction(self) -> float:
        return float((self.since_active > self.silence_steps).mean())


def feature_rank(acts: np.ndarray, eps: float = 1e-6) -> int:
    """Effective rank of trunk activations via singular values (spec §5).

    acts: [n_units] for a single sample -> rank is just count of non-zero
    dims. For the periodic 100k snapshot we accumulate a small window of
    recent activations; to stay buffer-free we instead compute rank on the
    Trunk features |acts|>eps count at this step. A richer snapshot can
    be computed by the caller from a rolling outer-product estimate (v2).
    """
    return int((np.abs(acts) > eps).sum())


# ---------------------------------------------------------------------------
# CSV logger -- one row per `log_every` steps (spec §5). No samples stored.
# ---------------------------------------------------------------------------
class CSVLogger:
    def __init__(self, path: str, columns):
        self.path = path
        self.columns = columns
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            f.write(",".join(columns) + "\n")

    def log(self, row: dict):
        with open(self.path, "a") as f:
            f.write(",".join(str(row.get(c, "")) for c in self.columns) + "\n")