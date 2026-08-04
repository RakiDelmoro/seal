"""SEAL imagination package — goal-directed trajectory sampling.

Imagination plans by sampling noisy rollouts on the learned cognitive map,
aiming at a geometric goal s* and scoring each rollout by a blend of the
learned value function V and geometric distance to the goal (the paper's
eq-28 form). The blend weight α starts at 0 (pure geometry while V is
random) and grows as V gains signal.
"""
from imagination.geometric_goal import GeometricGoal
from imagination.sampler import sample_trajectories
from imagination.evaluator import evaluate_trajectories, evaluate_trajectory
from imagination.engine import ImaginationEngine

__all__ = [
    "GeometricGoal", "sample_trajectories", "evaluate_trajectories",
    "evaluate_trajectory", "ImaginationEngine",
]
