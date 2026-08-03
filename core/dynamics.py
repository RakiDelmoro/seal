"""Banded dynamics matrix A + bias b.

  s_{t+1} = A · s_t + B · a_t + b   (B handled in action_effect.py)

A is N×N but only a band of ±A_HALF_BAND entries per row is learnable
(33 entries per row for half_band=16 → 33K learnable params, not 1M).
All out-of-band entries are fixed at 0.

Initialization:
  Diagonal: 1.0 + N(0, 0.01)   — near-identity (state persists)
  Off-diagonal band: N(0, 0.001) — tiny noise, sculpted by learning

Spectral safeguard (after each update):
  1. Clip diagonal to [0.95, 1.0]  — state persists, doesn't grow
  2. Clip per-row off-diagonal L2 norm to ≤ 0.1 — shift structure stays small

This keeps ‖A‖_op ≈ 1, preventing multi-step rollout divergence.
"""
from __future__ import annotations
import numpy as np

from config import (
    N_STATE, A_HALF_BAND, A_SEED,
    A_DIAG_INIT, A_DIAG_NOISE_STD, A_OFFDIAG_NOISE_STD,
    A_DIAG_CLIP, A_OFFDIAG_ROW_L2_CLIP,
    A_SPECTRAL_RADIUS_MAX, A_SPECTRAL_CHECK_INTERVAL,
)


class BandedDynamics:
    """Banded near-identity dynamics matrix A with bias b.

    Stored compactly as A_band: (N, K) where K = 2*half_band + 1.
    Column j of A_band corresponds to offset band_offsets[j] = j - half_band.
    Entry A_band[i, j] = A[i, i + offset_j].
    """

    def __init__(self, n_state: int = N_STATE,
                 half_band: int = A_HALF_BAND,
                 seed: int = A_SEED):
        self.N = n_state
        self.half_band = half_band
        self.K = 2 * half_band + 1
        self.band_offsets = np.arange(-half_band, half_band + 1)  # -16 .. +16
        self.diag_idx = half_band  # index of offset 0 in A_band

        rng = np.random.default_rng(seed)

        # A_band: (N, K) — compact banded storage
        self.A_band = np.zeros((n_state, self.K), dtype=np.float32)
        # Diagonal: 1.0 + small noise
        self.A_band[:, self.diag_idx] = (
            A_DIAG_INIT + rng.normal(0, A_DIAG_NOISE_STD, n_state)
        ).astype(np.float32)
        # Off-diagonal: tiny noise
        for j in range(self.K):
            if j != self.diag_idx:
                self.A_band[:, j] = (
                    rng.normal(0, A_OFFDIAG_NOISE_STD, n_state)
                ).astype(np.float32)

        # Bias b is intentionally absent: the ball's autonomous motion is
        # state-dependent ("ball at X → X+1"), captured by A's banded shift
        # structure — NOT by a constant drift. A constant b grows into an
        # attractor that collapses multi-step rollouts (all 40 imagined
        # futures drift to the same default state → indistinguishable →
        # scoring flatlines). The paper's forward model ŝ = s + V·a has no
        # bias for the same reason.
        self._update_count = 0
        self._dense_A_dirty = True
        self._dense_A = None

    # ── Forward: A @ s ─────────────────────────────────────────────
    def forward(self, s: np.ndarray) -> np.ndarray:
        """Compute A @ s for a single state vector using banded structure."""
        result = np.zeros(self.N, dtype=np.float32)
        for j, offset in enumerate(self.band_offsets):
            if offset == 0:
                result += self.A_band[:, j] * s
            elif offset > 0:
                result[:-offset] += self.A_band[:-offset, j] * s[offset:]
            else:  # offset < 0
                result[-offset:] += self.A_band[-offset:, j] * s[:self.N + offset]
        return result

    def _build_dense_A(self) -> np.ndarray:
        """Build a dense (N, N) version of A from the banded storage.

        Used for batched matmul during imagination — BLAS dense matmul is
        much faster than the 33-offset Python loop for large batches.
        Called once after each update, then reused for all 40×5=200 rollouts.
        """
        A = np.zeros((self.N, self.N), dtype=np.float32)
        for j, offset in enumerate(self.band_offsets):
            if offset == 0:
                np.fill_diagonal(A, self.A_band[:, j])
            elif offset > 0:
                idx = np.arange(self.N - offset)
                A[idx, idx + offset] = self.A_band[:self.N - offset, j]
            else:
                idx = np.arange(-offset, self.N)
                A[idx, idx + offset] = self.A_band[-offset:, j]
        self._dense_A = A
        self._dense_A_dirty = False
        return A

    def forward_batch(self, S: np.ndarray) -> np.ndarray:
        """Compute A @ S for a batch of states using the banded storage directly.

        Args:
            S: (B, N) batch of state vectors.

        Returns:
            (B, N) batch of A @ s for each row.

        Uses the banded A_band (33 nonzero offsets) with broadcast shifted
        multiplies — 33×(B×N) element ops instead of a full (B×N×N) dense
        matmul. Numerically identical to the dense form (A is zero outside the
        band) but ~30× faster when BLAS is single-threaded, and it avoids
        rebuilding a 1000×1000 dense matrix every frame.
        """
        S = np.asarray(S, dtype=np.float32)
        res = np.zeros_like(S)
        for j, offset in enumerate(self.band_offsets):
            col = self.A_band[:, j]              # (N,)
            if offset == 0:
                res += col * S                   # (B,N) * (N,) broadcast
            elif offset > 0:
                res[:, :-offset] += col[:-offset] * S[:, offset:]
            else:  # offset < 0
                res[:, -offset:] += col[-offset:] * S[:, :self.N + offset]
        return res

    def predict_batch(self, S: np.ndarray, A_actions: np.ndarray | None = None,
                      B=None) -> np.ndarray:
        """Batched full transition: A @ S + B @ actions + b.

        Args:
            S: (B, N) states.
            A_actions: (B, n_actions) one-hot action vectors, or None.
            B: action-effect matrix (n_state, n_actions), or None.

        Returns:
            (B, N) predicted next states.
        """
        out = self.forward_batch(S)
        if B is not None and A_actions is not None:
            out += A_actions @ B.T  # (B, n_actions) @ (n_actions, N) → (B, N)
        return out

    def predict(self, s: np.ndarray, action_vec: np.ndarray | None = None,
                B=None) -> np.ndarray:
        """Full transition: A @ s + B @ a (no bias — see __init__)."""
        out = self.forward(s)
        if B is not None and action_vec is not None:
            out = out + B @ action_vec
        return out

    # ── Update: delta rule on banded entries ───────────────────────
    def update(self, err: np.ndarray, s_t: np.ndarray,
               eta_a: float):
        """Normalized delta-rule update for A.

        ΔA[i, i+δ] = η_a × err[i] × s_t[i+δ] / (‖s_t‖² + ε)

        Normalization by ‖s_t‖² stabilizes the update when states have large
        norm (Pong ‖s‖²≈324; without it η_a=5e-4 gives effective rate 0.16,
        which is aggressive and fights the spectral clip).
        """
        norm_sq = max(float(s_t @ s_t), 1.0)  # prevent zero-state explosion
        scale_a = eta_a / norm_sq

        # Banded A update
        for j, offset in enumerate(self.band_offsets):
            if offset == 0:
                self.A_band[:, j] += scale_a * err * s_t
            elif offset > 0:
                self.A_band[:-offset, j] += scale_a * err[:-offset] * s_t[offset:]
            else:  # offset < 0
                self.A_band[-offset:, j] += scale_a * err[-offset:] * s_t[:self.N + offset]

        self._dense_A_dirty = True

    # ── Spectral safeguard ─────────────────────────────────────────
    def clip(self):
        """Apply structural clips + periodic spectral radius safeguard.

        1. Clip diagonal to A_DIAG_CLIP — allows decay (old activation fades)
        2. Clip per-row off-diagonal L2 to A_OFFDIAG_ROW_L2_CLIP — allows
           meaningful shift transfer while keeping rows sparse
        3. Every A_SPECTRAL_CHECK_INTERVAL updates, estimate ρ(A) via power
           iteration and rescale if > A_SPECTRAL_RADIUS_MAX
        """
        lo, hi = A_DIAG_CLIP
        np.clip(self.A_band[:, self.diag_idx], lo, hi,
                out=self.A_band[:, self.diag_idx])

        # Off-diagonal: all columns except diag_idx
        off = np.delete(self.A_band, self.diag_idx, axis=1)  # (N, K-1) copy
        norms = np.linalg.norm(off, axis=1)                   # (N,)
        scale = np.minimum(1.0, A_OFFDIAG_ROW_L2_CLIP / (norms + 1e-12))
        off *= scale[:, None]

        # Write back
        self.A_band[:, :self.diag_idx] = off[:, :self.diag_idx]
        self.A_band[:, self.diag_idx + 1:] = off[:, self.diag_idx:]

        self._dense_A_dirty = True

        # Periodic spectral radius check
        self._update_count += 1
        if self._update_count % A_SPECTRAL_CHECK_INTERVAL == 0:
            self._spectral_clip()

    def _spectral_clip(self):
        """Estimate ρ(A) via power iteration; rescale A if > max."""
        rho = self.operator_norm_estimate(n_iter=20)
        if rho > A_SPECTRAL_RADIUS_MAX:
            scale = A_SPECTRAL_RADIUS_MAX / rho
            self.A_band *= scale
            self._dense_A_dirty = True

    # ── Diagnostics ────────────────────────────────────────────────
    def operator_norm_estimate(self, n_iter: int = 10) -> float:
        """Estimate ‖A‖_op via power iteration (for monitoring)."""
        v = np.random.default_rng(0).standard_normal(self.N).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        for _ in range(n_iter):
            w = self.forward(v)
            nw = np.linalg.norm(w)
            if nw < 1e-12:
                return 0.0
            v = w / nw
        return float(nw)
