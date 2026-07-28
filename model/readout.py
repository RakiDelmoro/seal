"""Leaky readout neurons: actor (policy) + critic (value) heads (Eq. 11).

    y_k^t = κ·y_{t-1} + Σ_j Wout_kj·z_j^t + b_k

where κ = exp(-dt/τ_out) is the readout leak, z_j the LSNN spike rates, and
b_k the bias. The actor head produces logits -> softmax policy π(a|y); the
critic head produces a scalar value V.

**Separate weights for actor and critic** (paper Eq. 37 structure): the actor
readout `Wout_actor [n_actor, n_in]` and the critic readout `Wout_critic
[n_critic, n_in]` are DISTINCT parameters, trained by their own loss terms
(actor: −logπ·δ, critic: c_V·½δ²). This matches the paper's notation where
B^π_jk = (Wout^π_kj)^T and B^V_j = (Wout^V_j)^T are separate feedback channels
in the learning signal L_j (Eq. 37). Sharing one weight matrix lets the 6:1
actor:critic gradient ratio drown out the critic, so V never learns — the
separation fixes that.

These are FEEDFORWARD (not recurrent) — the readout has no recurrent
connections. Per the paper's Methods, the readout weights do NOT require
e-prop theory; they are trained by ordinary gradient descent (autograd + SGD)
on the actor-critic losses. The symmetric e-prop feedback weights B_jk are the
transposes of these readout weights (see broadcast.py).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyReadout(nn.Module):
    """Leaky output neurons (Eq. 11) for actor + critic, with SEPARATE weights.

    State (per output): y (leaky accumulator), carried across ms-substeps and
    across env steps (reset on episode boundary).

    Args:
        n_in:  number of input neurons (LSNN population size)
        n_actor: number of action logits (= n_actions)
        n_critic: 1 (scalar value)
        kappa: readout leak exp(-dt/τ_out)
    """
    def __init__(self, n_in: int, n_actor: int, n_critic: int = 1,
                 kappa: float = 0.95, logit_cap: float = 2.0,
                 kappa_critic: float = -1.0):
        super().__init__()
        self.n_in = n_in
        self.n_actor = n_actor
        self.n_critic = n_critic
        self.n_out = n_actor + n_critic
        self.kappa = float(kappa)
        # Per-channel leak for the critic head. The actor stays leaky
        # (e-prop Eq. 11, smoothing logits over ~tau_out frames). But a
        # leaky critic is a ~1/(1-kappa)=20x-gain integrator that
        # amplifies any DC bias in Wout*z+b into a huge wrong V, AND it
        # holds that bias in state for ~20 frames so terminal-reward
        # kicks can't flush it. That combination drove termV from -50 to
        # -178 in ~600 episodes (a monotonic runaway). Default -1.0 means
        # "inherit kappa" (legacy behavior); set 0.0 for a memoryless
        # critic V = Wout*z + b, matching stream-x's value head, which is
        # what ObGD's normalization was designed for.
        self.kappa_critic = float(kappa_critic) if kappa_critic >= 0.0 else self.kappa
        # Structural entropy floor: actor logits = cap·tanh(y/cap), so the
        # max logit gap is 2·cap = 4 and π_max ≈ 0.7–0.9 — softmax can never
        # saturate to one-hot, so (π − 1_a) — the policy channel of L_j —
        # can never die (observed failure: entropy = 0.00 for 50+ episodes).
        # Critic value is NOT capped (must represent ±21). 0 disables.
        self.logit_cap = float(logit_cap)
        # SEPARATE readout weights for actor and critic (paper Eq. 37).
        # This prevents the 6:1 actor:critic gradient ratio on a shared matrix
        # from drowning out the critic's value prediction.
        self.Wout_actor = nn.Parameter(torch.empty(n_actor, n_in))
        self.b_actor = nn.Parameter(torch.zeros(n_actor))
        self.Wout_critic = nn.Parameter(torch.empty(n_critic, n_in))
        self.b_critic = nn.Parameter(torch.zeros(n_critic))
        nn.init.kaiming_uniform_(self.Wout_actor, a=0.5)
        # Critic weights start SMALL: V = Wout_critic@LN(z) + b_critic, and with
        # kaiming L2~1.3 the Wout@LN(z) noise (std~1.3) swamps b_critic and
        # injects a negative bias into non-terminal delta (gamma*V'-V) that
        # sinks b_critic faster than terminal rewards lift it. Scaling to
        # L2~0.1 makes V≈b_critic initially, so delta is dominated by the
        # real (gamma-1)*V and terminal r-V signals, which both push b_critic
        # toward the true mean return. Wout_critic grows back as the TD
        # signal (clean once b_critic tracks) teaches it state-dependence.
        nn.init.kaiming_uniform_(self.Wout_critic, a=0.5)
        with torch.no_grad():
            self.Wout_critic.mul_(0.08)
        # leaky state (combined actor+critic, carried across steps)
        self.y = torch.zeros(self.n_out)

    def reset(self):
        self.y.zero_()

    def _split_y(self, y: torch.Tensor):
        return y[:self.n_actor], y[self.n_actor:]

    def step_ms(self, z: torch.Tensor) -> torch.Tensor:
        """Advance one ms. z = LSNN spike vector [n_in]. Returns y [n_out].

        The state self.y is updated DETACHED — the autograd graph for the
        readout is rebuilt in learn() via forward_from(), so we must not let
        y carry grad history across env steps (it would accumulate/freed-graph).
        """
        i_a = F.linear(z.unsqueeze(0), self.Wout_actor, self.b_actor).squeeze(0)
        i_c = F.linear(z.unsqueeze(0), self.Wout_critic, self.b_critic).squeeze(0)
        # Per-channel leak: actor decays by kappa, critic by kappa_critic.
        # Splitting here (not in forward_from) keeps both paths in sync.
        y_prev = self.y
        y_a = self.kappa * y_prev[:self.n_actor] + i_a
        y_c = self.kappa_critic * y_prev[self.n_actor:] + i_c
        y_new = torch.cat([y_a, y_c])
        self.y = y_new.detach()
        return y_new

    @staticmethod
    def _norm(z: torch.Tensor) -> torch.Tensor:
        """Parameter-free LayerNorm on the readout input (stream-x ingredient).

        ObGD's overshooting bound M = δ̄·‖e‖₁·lr·κ is only operative when
        features (and hence gradients) are O(1). Raw spike rates are ~1e-3,
        so without normalization M < 1 and the bound never engages — full
        lr·δ steps destabilize actor and critic. LN pins the feature scale.
        Applied IDENTICALLY in forward() and forward_from() so act() and
        learn() see the same function.
        """
        return F.layer_norm(z, z.shape)

    def forward(self, spike_rate: torch.Tensor) -> tuple:
        """Read out actor logits + critic value for one env step.

        Args:
            spike_rate: per-neuron spike rate over the env-step's sub-steps [n_in]
                        (from LSNNCore.forward).
        Returns:
            logits: [n_actor]  (pre-softmax)
            value:  scalar tensor (critic)
        """
        spike_rate = self._norm(spike_rate)
        y_prev = self.y.clone()  # capture BEFORE step_ms mutates
        y = self.step_ms(spike_rate)
        # snapshot so learn() can recompute a fresh graph without mutating state.
        self._y_prev = y_prev
        logits, value = self._split_y(y)
        logits = self._cap(logits)
        return logits, value[self.n_critic - 1] if self.n_critic == 1 else value

    def _cap(self, logits: torch.Tensor) -> torch.Tensor:
        """Tanh cap on actor logits (structural entropy floor). 0 = off."""
        if self.logit_cap > 0:
            return self.logit_cap * torch.tanh(logits / self.logit_cap)
        return logits

    def forward_from(self, y_prev: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Recompute y = κ·y_prev + Wout·z + b WITHOUT mutating state.

        Used by learn() to build a fresh autograd graph for the readout loss.
        Gradients route automatically to Wout_actor / Wout_critic since they
        are separate Parameters.
        """
        z = self._norm(z)   # same normalization as forward() — keep in sync
        i_a = F.linear(z.unsqueeze(0), self.Wout_actor, self.b_actor).squeeze(0)
        i_c = F.linear(z.unsqueeze(0), self.Wout_critic, self.b_critic).squeeze(0)
        # Per-channel leak — MUST match step_ms() so act() and learn()
        # see the same function (else the ObGD gradient is wrong).
        y_a = self.kappa * y_prev[:self.n_actor] + i_a
        y_c = self.kappa_critic * y_prev[self.n_actor:] + i_c
        y = torch.cat([y_a, y_c])
        # cap AFTER the κ-mixing, exactly as forward() does — act() and
        # learn() must see ONE function
        return torch.cat([self._cap(y[:self.n_actor]), y[self.n_actor:]])
