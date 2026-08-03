"""Checkpointing — save/load the learned SEAL components.

Learned components: A_band (dynamics), B (action effect), D (direction),
V (value function), π (policy). The perception pipeline and env are
deterministic given a seed.

Checkpoint format: a single .npz file with all weight arrays + metadata.
"""
from __future__ import annotations
import os
import time
import numpy as np

from core.seal_core import SEALCore


def save_checkpoint(core: SEALCore, path: str, metadata: dict | None = None):
    """Save all learned weights to a .npz file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "A_band": core.dynamics.A_band,
        "B": core.action_effect.B,
        "D": core.direction.D,
        "V_w": core.value.w,
        "V_e": core.value.e,
        "pi_theta": core.policy.theta,
        "step_count": core.step_count,
    }
    if metadata:
        for k, v in metadata.items():
            data[f"meta_{k}"] = v
    data["meta_saved_at"] = time.time()
    np.savez(path, **data)


def load_checkpoint(path: str) -> tuple[SEALCore, dict]:
    """Load weights from a .npz file into a new SEALCore.

    Returns:
        (core, metadata_dict)
    """
    data = np.load(path, allow_pickle=False)
    core = SEALCore()
    core.dynamics.A_band = data["A_band"].astype(np.float32)
    core.dynamics._dense_A_dirty = True
    core.action_effect.B = data["B"].astype(np.float32)
    core.direction.D = data["D"].astype(np.float32)

    # Load value and policy if present (backward-compatible with old checkpoints)
    if "V_w" in data.files:
        core.value.w = data["V_w"].astype(np.float32)
    if "V_e" in data.files:
        core.value.e = data["V_e"].astype(np.float32)
    if "pi_theta" in data.files:
        core.policy.theta = data["pi_theta"].astype(np.float32)

    core.step_count = int(data["step_count"])

    metadata = {}
    for key in data.files:
        if key.startswith("meta_"):
            metadata[key[5:]] = data[key]
    return core, metadata
