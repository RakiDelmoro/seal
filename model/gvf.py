"""Game-agnostic General Value Function (GVF) auxiliary bank.

Replaces the old Pong-specific aux head (ball_x/ball_y/paddle_contact from
the event-mask centroid, with a hardcoded left-paddle column). That head did
not transfer: its `contact` target was degenerate (nonzero 1.3% of frames on
Pong) and its centroid semantics were Pong-specific.

This module defines a bank of temporal-prediction GVFs whose cumulants are
derived ONLY from signals that exist in every frame-based env:

  * the event mask of the first conv layer (which pixels changed) — already
    computed for free by the event-driven encoder;
  * the scalar reward;
  * nothing else.

Each GVF is a TD(λ) prediction of the discounted future return of its
cumulant:  G_t = Σ_{j>=0} γ^j c_{t+j}. It is learned by a SwiftTD linear
learner (256→1) with its own λ. This is the Horde (Sutton et al. 2011,
"Horde: A scalable real-time architecture for learning knowledge from
unsupervised sensorimotor interaction") — many off-policy value predictions
sharing a representation — and the reward-prediction heads are the
UNREAL (Jaderberg et al. 2016) aux task, here delivered streaming (no replay)
via SwiftTD.

Why these four (grounded in the empirical probe of Pong under random play,
which showed positive reward fires on only 0.17% of frames):

  motion_density   λ=0.9  c = event_mask.mean()
     "how much of the screen will be moving in the near future" — a dense
     dynamics signal available every frame, teaching the trunk to encode
     motion. (λ long: credit spreads over the ~seconds-long motion envelope.)
  pos_reward       λ=0.5  c = max(r, 0)
     THE sparse-reward forecaster. Directly attacks the 0.17% positive-reward
     sparsity: discounting gives a dense TD signal on every frame LEADING UP
     to a score, not just on the rare scoring frame.
  neg_reward       λ=0.5  c = max(-r, 0)
     "am I about to lose a point" — denser than positive reward in Pong
     (9.8% of frames), useful everywhere there is a losing side.
  motion_spread    λ=0.9  c = event_mask.float().std()
     "is motion localized (a ball) or spread out (a swarm)" — captures
     spatial structure of motion, game-agnostic. Teaches the trunk a richer
     representation than the scalar density alone.

All cumulants are computable on any Atari game (and any frame-based env with
a reward signal) with zero game-specific knowledge. Switching Pong →
Breakout → SpaceInvaders leaves this bank identical and meaningful.

The encoder is shaped by these heads through the joint loss (see agent.learn);
the head weights themselves are updated by SwiftTD with per-GVF δ and λ.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch


@dataclass(frozen=True)
class GVFSpec:
    """Specification of one general value function (one linear TD(λ) head)."""
    name: str
    lam: float            # eligibility trace λ for this GVF
    weight: float         # coefficient in the encoder-shaping loss
    cumulant: Callable[[torch.Tensor, float], float]
    # cumulant(event_mask: [1,C,H,W] bool/float, reward: float) -> float


def _motion_density(mask: torch.Tensor, r: float) -> float:
    return float(mask.float().mean().item())


def _pos_reward(mask: torch.Tensor, r: float) -> float:
    return float(max(r, 0.0))


def _neg_reward(mask: torch.Tensor, r: float) -> float:
    return float(max(-r, 0.0))


def _motion_spread(mask: torch.Tensor, r: float) -> float:
    m = mask.float()
    return float(m.std().item()) if m.numel() > 1 else 0.0


DEFAULT_GVFS: tuple[GVFSpec, ...] = (
    GVFSpec(name="motion_density", lam=0.9, weight=0.1, cumulant=_motion_density),
    GVFSpec(name="pos_reward",    lam=0.5, weight=0.1, cumulant=_pos_reward),
    GVFSpec(name="neg_reward",    lam=0.5, weight=0.1, cumulant=_neg_reward),
    GVFSpec(name="motion_spread", lam=0.9, weight=0.1, cumulant=_motion_spread),
)


def n_gvfs(gvfs: tuple[GVFSpec, ...] = DEFAULT_GVFS) -> int:
    return len(gvfs)


def gvf_lams(gvfs: tuple[GVFSpec, ...] = DEFAULT_GVFS) -> tuple[float, ...]:
    return tuple(g.lam for g in gvfs)


def gvf_weights(gvfs: tuple[GVFSpec, ...] = DEFAULT_GVFS) -> tuple[float, ...]:
    return tuple(g.weight for g in gvfs)


def compute_cumulants(event_mask: torch.Tensor, reward: float,
                      gvfs: tuple[GVFSpec, ...] = DEFAULT_GVFS) -> torch.Tensor:
    """Per-GVF cumulant c_t from the event mask and the scalar reward.

    Returns a [n_gvfs] float tensor (detached). Game-agnostic by construction.
    """
    mask = event_mask.detach()
    return torch.tensor([g.cumulant(mask, reward) for g in gvfs],
                        dtype=torch.float32, device=mask.device)
