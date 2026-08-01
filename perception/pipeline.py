"""Full perception pipeline: raw frame → locality-ordered state vector.

  Frame_t  ─┐
            ├─→ 2-channel input → FrozenCNN → 9×9×32 feature map
  |Δframe| ─┘                              → flatten (py, px, channel) → state s

The frozen CNN replaces the old DoG + Gabor + random-projection pipeline with
hierarchical features (edges → combinations → walls/paddles). This is the
Koopman lifting that lets the linear banded A represent bounces and keep
pred_err falling past the Gabor ceiling.

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

    def forward(self, frame: np.ndarray):
        """Process one frame.

        Args:
            frame: (H, W) float32 — single-channel, normalized.

        Returns:
            state: (N_STATE,) float32 — locality-ordered.
            features: (n_features,) float32 — same as state (CNN output, no
                separate feature/state split; kept for API compatibility).
        """
        frame = np.asarray(frame, dtype=np.float32)

        # Motion channel: |frame - prev_frame| (zeros on first frame)
        if self._prev_frame is not None:
            frame_diff = np.abs(frame - self._prev_frame)
        else:
            frame_diff = np.zeros_like(frame)

        self._prev_frame = frame.copy()

        # Frozen CNN: 2-channel input → locality-ordered state
        state = self.cnn.forward(frame, frame_diff)

        # Pad or truncate to N_STATE if needed
        if len(state) < N_STATE:
            state = np.pad(state, (0, N_STATE - len(state)))
        elif len(state) > N_STATE:
            state = state[:N_STATE]

        return state, state.copy()
