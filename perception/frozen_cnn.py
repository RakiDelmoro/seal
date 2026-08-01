"""Frozen CNN front-end: pixels → locality-ordered state (replaces Gabor + E).

A 2-layer stride-conv stack with FROZEN random weights turns the 84×84 frame
into a hierarchical feature map, flattened locality-ordered into the state
vector. This replaces the flat Gabor + random-projection pipeline with richer
features (edges → combinations → walls/paddles) — the Koopman lifting that
lets the linear banded A represent bounces and keep pred_err falling.

  Input:  2-channel image (normalized frame, |Δframe|)  →  (2, 84, 84)
  Conv1:  2→16 channels, 8×8 kernel, stride 4, leaky_relu → (16, 20, 20)
  Conv2:  16→32 channels, 4×4 kernel, stride 2, leaky_relu → (32, 9, 9)
  Output: 9×9 grid × 32 channels = 2592 features, flattened (py, px, channel)

The flatten order (py, px, channel) makes each spatial position (py, px) map
to a contiguous 32-dim block of state — the locality structure the banded A
needs. A ball shifting one grid cell shifts activation by one 32-dim block,
within the ±32 band.

FROZEN: weights are random (Kaiming init), never trained. This is the
grid-cell principle (fixed nonlinear high-D lifting) applied to images —
the CNN's hierarchical architecture gives "walls from edges" for free, without
backprop. The task-specific learning happens in the linear readouts (A, B, D)
on top, exactly as in the GCML/CML papers.

Implementation note: the forward pass uses torch.nn.functional.conv2d (a fast
C-optimized conv) with FROZEN weights — no training, no backprop, no GPU
required. The NumPy BLAS in this container is pathologically slow for small
matmuls (~20ms for a 16×128 @ 128×400), making a pure-NumPy im2col ~40ms per
frame; torch's conv2d is ~1ms. Weights are stored as torch tensors; the
interface is NumPy in / NumPy out.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


class FrozenCNN:
    """Frozen 2-layer stride-conv encoder: 2-channel image → locality-ordered state.

    The output is flattened as (py, px, channel) so each spatial position maps
    to a contiguous block of state dims — the locality structure the banded A
    needs.
    """

    def __init__(self, seed: int = 42,
                 conv1_channels: int = 16, conv2_channels: int = 16,
                 image_size: int = 84):
        rng = np.random.default_rng(seed)
        self.image_size = image_size
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels

        # Layer 1: 2→16, kernel 8, stride 4 → 20×20
        k1 = 8
        s1 = 4
        h1 = (image_size - k1) // s1 + 1   # 20
        fan_in1 = 2 * k1 * k1
        w1 = rng.normal(0, np.sqrt(2.0 / fan_in1),
                        (conv1_channels, 2, k1, k1)).astype(np.float32)

        # Layer 2: 16→32, kernel 4, stride 2 → 9×9
        k2 = 4
        s2 = 2
        h2 = (h1 - k2) // s2 + 2            # 9 (h1 - k2 = 16; (16)//2 + 1 = 9)
        fan_in2 = conv1_channels * k2 * k2
        w2 = rng.normal(0, np.sqrt(2.0 / fan_in2),
                        (conv2_channels, conv1_channels, k2, k2)
                        ).astype(np.float32)

        # Store as torch tensors (frozen — no gradients)
        self.w1 = torch.from_numpy(w1)
        self.b1 = torch.zeros(conv1_channels)
        self.w2 = torch.from_numpy(w2)
        self.b2 = torch.zeros(conv2_channels)

        # Output geometry
        self.grid_size = h2                 # 9
        self.n_positions = h2 * h2          # 81
        self.channels = conv2_channels      # 32
        self.n_features = self.n_positions * self.channels  # 2592

        # Precompute the position→state-dim ranges (locality-ordered)
        self.position_ranges = []
        for p in range(self.n_positions):
            start = p * self.channels
            self.position_ranges.append((start, start + self.channels))

    def forward(self, frame: np.ndarray, frame_diff: np.ndarray) -> np.ndarray:
        """Encode one frame → locality-ordered state vector.

        Args:
            frame: (H, W) normalized grayscale frame.
            frame_diff: (H, W) |frame_t - frame_{t-1}| (zeros on first frame).

        Returns:
            (n_features,) state vector, flattened (py, px, channel).
        """
        # Stack into 2-channel input: (1, 2, H, W) for torch conv2d
        x = np.stack([frame, frame_diff], axis=0).astype(np.float32)
        x_t = torch.from_numpy(x).unsqueeze(0)  # (1, 2, H, W)

        # Conv1 + leaky_relu
        h = F.conv2d(x_t, self.w1, self.b1, stride=4)
        h = F.leaky_relu(h)

        # Conv2 + leaky_relu
        h = F.conv2d(h, self.w2, self.b2, stride=2)
        h = F.leaky_relu(h)

        # Flatten (1, C, H, W) → (H, W, C) → locality-ordered (py, px, channel)
        # Transpose to (H, W, C) then flatten → position p = py*W + px,
        # each position's C channels are contiguous.
        out = h.squeeze(0).permute(1, 2, 0).flatten()  # (H*W*C,) = (2592,)
        return out.numpy().astype(np.float32)
