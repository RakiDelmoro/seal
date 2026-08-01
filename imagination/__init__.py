"""SEAL imagination package — geometric goal-directed trajectory sampling.

Imagination plans by sampling noisy rollouts on the learned cognitive map,
aiming at a geometric goal s* and scoring each rollout by geometric distance
to that goal (the paper's eq-28 form). No learned value function is used on
the planning path.
"""
from imagination.geometric_goal import GeometricGoal
from imagination.sampler import sample_trajectories
from imagination.evaluator import evaluate_trajectories, evaluate_trajectory
from imagination.engine import ImaginationEngine

__all__ = [
    "GeometricGoal", "sample_trajectories", "evaluate_trajectories",
    "evaluate_trajectory", "ImaginationEngine",
]
