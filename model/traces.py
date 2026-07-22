"""Eligibility traces (spec §2.5).

    trace[p] = lam * gamma * trace[p] + grads[p]       # standard accumulating traces

In SEAL the traces live INSIDE the ObGD optimizer (the streaming-RL convention;
see optimizers.py). Traces carry temporal credit backward through time so the
agent never needs BPTT or a replay buffer: each step's gradient is folded into
the trace, and when a reward arrives the trace determines which past parameters
get credit. Traces reset on episode boundaries (spec §2.8).

λ (lam) is the trace decay -- the ONLY true dial in ObGD's bound-active regime
(see optimizers.py docstring for the α-cancels derivation). Paper value: 0.8.
"""
from __future__ import annotations
# No class here: ObGD holds the traces directly. This module exists to document
# the mechanism and keep the spec's file layout (seal/traces.py).
