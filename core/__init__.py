"""SEAL core package — learned readouts on the frozen state, all local rules.

  A, B   transition model (self-supervised; banded dynamics + action effect)
  D      inverse model — the paper's W (self-supervised)
  V      value function (streaming TD(λ), reward-driven)
  π      reactive policy (imitation + actor-critic)
  r̂      reward predictor (drives imagined TD)
  V_sf   successor-feature value — TD(λ) on the r̂ stream ("where does s lead?")
"""
from core.dynamics import BandedDynamics
from core.action_effect import ActionEffect
from core.direction import Direction
from core.gate import FeasibilityGate
from core.seal_core import SEALCore
from core.successor import SuccessorValue

__all__ = [
    "BandedDynamics", "ActionEffect", "Direction", "FeasibilityGate",
    "SEALCore", "SuccessorValue",
]
