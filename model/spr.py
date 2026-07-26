"""SPR — Self-Predictive Representations for streaming RL (arXiv:2602.09396).

The online encoder maps observation o_t → latent z_t. A transition model
predicts future latents from the current latent + a sequence of actions:
  ẑ_{t+1} = D(z_t, a_t)
  ẑ_{t+2} = D(ẑ_{t+1}, a_{t+1})
  ẑ_{t+K} = D(ẑ_{t+K-1}, a_{t+K-1})
These predicted latents are projected through a projection head P and
compared (negative cosine similarity) against STOP-GRADIENT target
projections from the momentum (EMA) target encoder:
  target_{t+k} = P'(z'_{t+k})     where z'_{t+k} = target_encoder(o_{t+k})

The loss is the sum of negative cosine similarities over k=1..K. The network
learns to make z_t carry enough information to predict its own future —
discovering what's predictive without any hand-designed cumulants.

This replaces the GVF bank. The GVF bank used hand-designed cumulants
(motion density, reward) and its encoder-shaping gradient conflicted with the
Q gradient (gradient conflict → encoder destabilization → z-explosion). SPR's
gradient is orthogonalized against Q (in agent.learn) so it only shapes the
encoder in non-conflicting directions, and the EMA target prevents
self-referential drift.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionModel(nn.Module):
    """Predicts the next latent from the current latent + one-hot action.

    z is [1, trunk_dim]; action is an int. The action is embedded and
    concatenated with z, then passed through a small MLP. This is unrolled K
    times to predict K steps ahead (no BPTT over the stream — each unroll
    step is a single forward through this module).
    """

    def __init__(self, trunk_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, trunk_dim)
        self.mlp = nn.Sequential(
            nn.Linear(trunk_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, trunk_dim),
        )

    def forward(self, z: torch.Tensor, action: int) -> torch.Tensor:
        """z: [1, trunk_dim], action: int → predicted next z [1, trunk_dim]."""
        a_emb = self.action_embed(torch.tensor([action], device=z.device))
        return self.mlp(torch.cat([z, a_emb], dim=-1))


class ProjectionHead(nn.Module):
    """Projects a latent to a lower-dimensional space for the cosine-sim loss.

    The SPR paper uses a projection head on both the predicted and target
    latents. We share one head for the online predictions; the target uses
    a stop-gradient copy of the same projection (the paper allows sharing).
    """

    def __init__(self, trunk_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk_dim, proj_dim),
            nn.LeakyReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def spr_loss(pred_latents: list, target_projs: list) -> torch.Tensor:
    """Sum of negative cosine similarities over K prediction steps.

    pred_latents: list of K projected predicted latents [1, proj_dim]
    target_projs: list of K stop-gradient target projections [1, proj_dim]
    Returns a scalar loss (minimize → predictions align with targets).
    """
    total = torch.tensor(0.0, device=pred_latents[0].device)
    for pred, tgt in zip(pred_latents, target_projs):
        total = total + (1.0 - F.cosine_similarity(pred, tgt.detach(), dim=-1)).sum()
    return total
