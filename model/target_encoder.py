"""Momentum target encoder for SPR (arXiv:2602.09396).

A slow EMA copy of the online encoder, used to produce stable stop-gradient
targets for the self-prediction auxiliary loss. The target's latents don't
chase a moving goal (the online net's own output), which prevents the
self-referential drift that destabilized the GVF bank.

Update rule (each step):  θ' ← (1 - τ) · θ' + τ · θ
The target never receives gradients — it only follows the online encoder
slowly. τ small (default 0.01) → target moves ~1% per step → stable.
"""
from __future__ import annotations
import copy
import torch


class TargetEncoder:
    """EMA copy of an nn.Module's parameters. No gradients flow into it."""

    def __init__(self, online_encoder: torch.nn.Module, tau: float = 0.01):
        self.tau = float(tau)
        self.target = copy.deepcopy(online_encoder)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: torch.nn.Module):
        """EMA update: target ← (1-τ)·target + τ·online."""
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(),
                              online_encoder.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the target encoder forward (no grad). Returns trunk features [1, 256]."""
        return self.target(x)

    def state_dict(self) -> dict:
        return {"target": self.target.state_dict(), "tau": self.tau}

    def load_state_dict(self, sd: dict):
        self.target.load_state_dict(sd["target"])
        self.tau = float(sd.get("tau", self.tau))
