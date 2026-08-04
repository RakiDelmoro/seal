"""Imagined TD — streaming Dyna without a replay buffer.

After a real transition, the agent rehearses: it rolls out K short futures
from the new state using its own learned dynamics (A, B), picks imagined
actions with its current policy π (mixed with a little uniform noise for
diversity), asks the reward model r̂ how much reward each imagined step would
bring, and gives the critic V a small one-step TD update at each imagined
step.

Nothing is stored — rollouts are generated fresh from the CURRENT weights
and consumed immediately. That is what makes it streaming: O(1) memory, no
replay buffer, no off-policy correction.

Two safeguards:
  • Confidence decay κ: model error compounds with depth, so the learning
    rate of each imagined update is scaled by κ = kappa_decay ** depth.
  • Norm renormalization: A slightly shrinks vectors, and V/r̂ were trained
    on real-magnitude states, so each imagined state is rescaled to the
    starting state's norm (the same trick the planning sampler uses).

Imagined transitions use Value.update_imagined (one-step TD, λ=0): the real
eligibility trace is never contaminated with synthetic states.
"""
from __future__ import annotations
import numpy as np

from config import (
    N_ACTIONS, GAMMA,
    IMAGINED_TD_K, IMAGINED_TD_HORIZON, IMAGINED_TD_ETA,
    IMAGINED_TD_KAPPA_DECAY, IMAGINED_TD_EXPLORE,
    IMAGINED_TD_FROM_MEMORY_K,
)


def imagined_td(core, s_start: np.ndarray,
                K: int = IMAGINED_TD_K,
                horizon: int = IMAGINED_TD_HORIZON,
                eta: float = IMAGINED_TD_ETA,
                kappa_decay: float = IMAGINED_TD_KAPPA_DECAY,
                explore: float = IMAGINED_TD_EXPLORE,
                rng: np.random.Generator | None = None) -> dict:
    """Run imagined-TD updates from s_start. Returns logging metrics.

    Args:
        core: SEALCore (provides dynamics, policy, reward model, critic).
        s_start: state to imagine from (normally the state just reached).
        K: number of imagined rollouts.
        horizon: imagined steps per rollout.
        eta: base critic learning rate for imagined updates.
        kappa_decay: per-step confidence decay (scales eta).
        explore: probability mass mixed in from a uniform action distribution.
        rng: numpy Generator (defaults to core's own).
    """
    if rng is None:
        rng = core._rng

    target_norm = float(np.linalg.norm(s_start)) + 1e-8
    deltas: list[float] = []
    r_hats: list[float] = []

    for _ in range(K):
        s_hat = s_start
        kappa = 1.0
        for _ in range(horizon):
            # Imagined action: current policy, softened with uniform noise so
            # the rehearsal doesn't just follow one deterministic groove.
            probs = core.policy.forward(s_hat)
            probs = (1.0 - explore) * probs + explore / N_ACTIONS
            probs = probs / probs.sum()  # float32 safety for rng.choice
            action = int(rng.choice(N_ACTIONS, p=probs))

            # World model predicts the next state.
            s_next = core.predict_next_state(s_hat, action)
            n = float(np.linalg.norm(s_next))
            if n > 1e-8:
                s_next = s_next * (target_norm / n)

            # Reward model predicts reward on arrival; critic learns from it.
            r_hat = core.reward_model.forward(s_next)
            delta = core.value.update_imagined(
                s_hat, r_hat, s_next, eta=eta * kappa, gamma=GAMMA
            )
            deltas.append(delta)
            r_hats.append(r_hat)

            s_hat = s_next
            kappa *= kappa_decay

    return {
        "n_updates": len(deltas),
        "imagined_delta_avg": float(np.mean(deltas)) if deltas else 0.0,
        "r_hat_avg": float(np.mean(r_hats)) if r_hats else 0.0,
    }


def imagined_td_from_memory(core,
                            K: int = IMAGINED_TD_FROM_MEMORY_K,
                            horizon: int = IMAGINED_TD_HORIZON,
                            eta: float = IMAGINED_TD_ETA,
                            kappa_decay: float = IMAGINED_TD_KAPPA_DECAY,
                            explore: float = IMAGINED_TD_EXPLORE,
                            rng: np.random.Generator | None = None) -> dict:
    """Rehearse from PROVEN-GOOD states (the pre-score memory).

    core.pre_score_states holds the frames that preceded an actual +1.
    Rolling out from a random one of them repeatedly re-enters the rewarded
    region, so the critic stays sharp around success patterns across
    episodes — focused rehearsal, still streaming (the deque already exists
    for goal selection; nothing new is stored). Returns zeroed metrics when
    the memory is empty (cold start).
    """
    if rng is None:
        rng = core._rng
    if not core.pre_score_states:
        return {"n_updates": 0, "imagined_delta_avg": 0.0, "r_hat_avg": 0.0}
    idx = int(rng.integers(len(core.pre_score_states)))
    s_mem = core.pre_score_states[idx]
    return imagined_td(core, s_mem, K=K, horizon=horizon, eta=eta,
                       kappa_decay=kappa_decay, explore=explore, rng=rng)
