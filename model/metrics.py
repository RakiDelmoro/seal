"""SEAL metrics (spec §5) + aux-target extraction (spec §2.4).

Analytics only -- never stores samples. Everything is computed from the
current single sample and a few running counters.

extract_aux_targets (spec §2.4, D5):
  Aux targets come FREE from the event mask of the FIRST conv layer. The
  centroid of event pixels approximates ball/paddle positions; the fraction
  of event mass in the paddle's x-column approximates paddle-contact. This is
  a deliberate design feature (spec §2.4): the aux task is supervised by the
  event mask itself, no extra env coupling.

  Calibration note (D5): with EventConv2d(8,s5) on 84x84, layer-0's mask is
  the INPUT-space mask [1,1,84,84] (we expose last_mask in input space). The
  centroid of lit pixels maps to a normalized (ball_x, ball_y) in [0,1].
  Pong's own paddle is on the left (~x in [0,8]) and the opponent on the right
  (~x in [75,83]); 'contact' = event mass in the left paddle column.
"""
from __future__ import annotations
import numpy as np
import torch


def extract_aux_targets(event_mask: torch.Tensor, obs_shape=None) -> torch.Tensor:
    """[1,3] = (ball_x, ball_y, paddle_contact), all normalized to [0,1].

    event_mask: [1, C, H, W] bool/float from EventConv2d.last_mask (input space).
    Returns a [1,3] float tensor. If no events, returns zeros (the agent will
    not learn from a degenerate frame; this is rare and self-corrects).
    """
    if event_mask.dim() == 4:
        m = event_mask.float().sum(dim=1)  # [1, H, W] event mass per pixel
    elif event_mask.dim() == 3:
        m = event_mask.float()
    else:
        raise ValueError(f"unexpected mask shape {event_mask.shape}")
    m = m[0]  # [H, W]
    total = m.sum().item()
    H, W = m.shape
    if total <= 1e-8:
        return torch.zeros(1, 3, device=event_mask.device)
    ys = torch.arange(H, device=m.device, dtype=torch.float32)
    xs = torch.arange(W, device=m.device, dtype=torch.float32)
    cy = (m.sum(dim=1) * ys).sum() / total / max(1, H - 1)   # [0,1]
    cx = (m.sum(dim=0) * xs).sum() / total / max(1, W - 1)   # [0,1]
    # paddle contact: event mass fraction in left paddle column (x < ~10% W)
    left_col = m[:, : max(1, W // 10)].sum() / total
    return torch.stack([cx, cy, left_col]).unsqueeze(0)  # [1,3]


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
