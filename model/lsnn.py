"""LSNN recurrent core: LIF + ALIF populations with recurrent + input weights.

The core runs `sim_ms_per_step` ms of simulation per env step, consuming input
spikes from the spiking CNN and producing recurrent spikes z. Eligibility
traces for Win and Wrec are maintained per-synapse and advanced each ms.

Weights:
    Win  [n_total, n_input]   — input -> core
    Wrec [n_total, n_total]   — recurrent (no self-connections)

The population is split: n_lif LIF neurons followed by n_alif ALIF neurons.
Eligibility objects track ε for Win and Wrec separately (Win uses input spikes
as presynaptic; Wrec uses recurrent spikes).

State carried across the sim_ms_per_step sub-steps within one env step;
reset on episode boundary.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.neurons import LIFNeurons, ALIFNeurons
from model.eligibility import LIFEligibility, ALIFEligibility


class LSNNCore(nn.Module):
    """Recurrent spiking core: LIF + ALIF mix, eligibility traces included."""

    def __init__(self, cfg, n_input_neurons: int):
        super().__init__()
        self.cfg = cfg
        self.n_input = int(n_input_neurons)
        self.n_lif = int(cfg.n_lif)
        self.n_alif = int(cfg.n_alif)
        self.n_total = self.n_lif + self.n_alif
        self.sim_ms = int(cfg.sim_ms_per_step)

        # ---- weights (learnable by e-prop) ----
        Win = torch.empty(self.n_total, self.n_input)
        nn.init.uniform_(Win, -cfg.win_scale, cfg.win_scale)
        Wrec = torch.empty(self.n_total, self.n_total)
        nn.init.uniform_(Wrec, -cfg.wrec_scale, cfg.wrec_scale)
        # no self-connections
        Wrec.fill_diagonal_(0.0)
        self.Win = nn.Parameter(Win)
        self.Wrec = nn.Parameter(Wrec)

        # ---- neuron populations ----
        self.lif = LIFNeurons(self.n_lif, cfg.alpha, cfg.v_threshold,
                              cfg.gamma_pd, cfg.refractory_steps)
        self.alif = ALIFNeurons(self.n_alif, cfg.alpha, cfg.rho, cfg.v_threshold,
                                cfg.beta, cfg.gamma_pd, cfg.refractory_steps)

        # ---- eligibility traces (per-synapse, forward-computed) ----
        # Win eligibility: presynaptic = input spikes (n_input)
        self.elig_win_lif = LIFEligibility(self.n_lif, self.n_input, cfg.alpha)
        self.elig_win_alif = ALIFEligibility(self.n_alif, self.n_input, cfg.alpha,
                                             cfg.rho, cfg.beta, cfg.alif_elig_approx)
        # Wrec eligibility: presynaptic = recurrent spikes (n_total)
        self.elig_wrec_lif = LIFEligibility(self.n_lif, self.n_total, cfg.alpha)
        self.elig_wrec_alif = ALIFEligibility(self.n_alif, self.n_total, cfg.alpha,
                                              cfg.rho, cfg.beta, cfg.alif_elig_approx)

        # last full-population spike vector (for readout + learning signal)
        self.last_z = torch.zeros(self.n_total)
        # accumulated spikes over the env-step's sub-steps (for readout input)
        self.spike_count = torch.zeros(self.n_total)

    def reset(self):
        """Reset neuron state + eligibility traces (episode boundary)."""
        self.lif.reset()
        self.alif.reset()
        self.elig_win_lif.reset()
        self.elig_win_alif.reset()
        self.elig_wrec_lif.reset()
        self.elig_wrec_alif.reset()
        self.last_z.zero_()
        self.spike_count.zero_()

    def _split(self, vec: torch.Tensor):
        return vec[:self.n_lif], vec[self.n_lif:]

    def step_ms(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Advance ONE ms. Returns the full-population spike vector [n_total].

        Drives neuron dynamics + eligibility trace recursion. Runs under
        torch.no_grad() — e-prop computes gradients manually via eligibility
        traces, so the core must NOT build an autograd graph through Win/Wrec.
        """
        with torch.no_grad():
            # previous spikes (presynaptic side for this ms)
            z_prev = self.last_z
            z_prev_lif, z_prev_alif = self._split(z_prev)

            # synaptic input currents
            i_rec = F_linear(self.Wrec, z_prev)        # [n_total]
            i_in = F_linear(self.Win, input_spikes)    # [n_total]
            i_total = i_rec + i_in
            i_lif, i_alif = self._split(i_total)

            # neuron steps (membrane + spike + psi)
            z_lif, psi_lif = self.lif.step(i_lif)
            z_alif, psi_alif = self.alif.step(i_alif)

            # eligibility trace recursion (Eq. 14/22/24) using z_prev (presynaptic)
            # Win eligibility: presynaptic = input_spikes
            self.elig_win_lif.step(input_spikes, psi_lif)
            self.elig_win_alif.step(input_spikes, psi_alif)
            # Wrec eligibility: presynaptic = z_prev (full population)
            self.elig_wrec_lif.step(z_prev, psi_lif)
            self.elig_wrec_alif.step(z_prev, psi_alif)

            z = torch.cat([z_lif, z_alif])
            self.last_z = z
            self.spike_count += z
        return z

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Run `sim_ms_per_step` ms of simulation for one env step.

        Runs under torch.no_grad() — the readout (which needs autograd) takes
        the returned spike_rate as a detached input and builds its own graph.
        """
        with torch.no_grad():
            self.spike_count.zero_()
            for _ in range(self.sim_ms):
                self.step_ms(input_spikes)
            return self.spike_count / float(self.sim_ms)

    # ----------------------------------------------------------- eligibility
    def eligibility_win(self) -> torch.Tensor:
        """Current eligibility trace for Win [n_total, n_input] (Eq. 23/25)."""
        return torch.cat([self.elig_win_lif.trace, self.elig_win_alif.trace], 0)

    def eligibility_wrec(self) -> torch.Tensor:
        """Current eligibility trace for Wrec [n_total, n_total]."""
        return torch.cat([self.elig_wrec_lif.trace, self.elig_wrec_alif.trace], 0)

    def spike_rate(self) -> float:
        """Mean per-neuron spike rate over the last env step (Hz-ish)."""
        return float(self.spike_count.mean().item()) / float(self.sim_ms) * 1000.0


def F_linear(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """weight @ x  (F.linear expects [*, in] -> [*, out]; we want [out] = W·x)."""
    return F.linear(x.unsqueeze(0), weight).squeeze(0)
