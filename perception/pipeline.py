"""Full perception pipeline: raw frame → locality-ordered state vector.

  Frame_t  ─┐
            ├─→ 2-channel input → FrozenCNN → 9×9×16 feature map
  |Δframe| ─┘                              → flatten (py, px, channel) → state s

The frozen CNN is a fixed nonlinear lifting (edges → combinations →
walls/paddles); all task-specific learning happens in the linear readouts
(A, B, D, V, π) on top. The flatten order (py, px, channel) gives the banded
A its locality structure.

The pipeline is stateful (stores the previous frame for temporal diff).
Call reset() at the start of each episode.
"""
from __future__ import annotations
import numpy as np

from perception.frozen_cnn import FrozenCNN

from config import N_STATE


class PerceptionPipeline:
    """Fixed perception: frame → state. No learning anywhere."""

    def __init__(self):
        self.cnn = FrozenCNN()
        self._prev_frame = None
        self.feature_dim = self.cnn.n_features
        self.state_dim = N_STATE

    def reset(self):
        """Clear temporal state (call at episode start)."""
        self._prev_frame = None

    def forward(self, frame: np.ndarray) -> np.ndarray:
        """Process one frame → (N_STATE,) float32 locality-ordered state."""
        frame = np.asarray(frame, dtype=np.float32)

        # Motion channel: |frame - prev_frame| (zeros on first frame)
        if self._prev_frame is not None:
            frame_diff = np.abs(frame - self._prev_frame)
        else:
            frame_diff = np.zeros_like(frame)

        self._prev_frame = frame.copy()

        # Frozen CNN: 2-channel input → locality-ordered state (1296,)
        state = self.cnn.forward(frame, frame_diff)
        assert len(state) == N_STATE, (
            f"CNN output {len(state)} features, expected N_STATE={N_STATE}")

        return state
