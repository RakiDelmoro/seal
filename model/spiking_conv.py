"""Spiking CNN front-end: pixels -> input spike trains (paper Fig. 4b).

A lightweight stride-conv pipeline downsamples the 84x84 normalized grayscale
frame into a population of input neurons, each of which emits a Poisson-like
spike train with rate proportional to (a function of) its feature intensity.
The LSNN recurrent core then consumes these input spikes.

Encoding: rate coding with a soft threshold. Each input neuron i has an
input current x_i (the conv feature at its location). It spikes with
probability p_i = clip(gain · relu(x_i)) per ms (Bernoulli), giving a spike
rate that saturates at high intensities. This keeps firing sparse (<~100 Hz)
and energy-efficient, matching the paper's spike-coding regime.

TRAINABLE (paper-faithful): in Bellec et al. 2020 the spiking CNN is trained —
"the current error in prediction is fed back both to the LSNN and the spiking
CNN" (Fig. 4b caption; official code trains the torso with its own optimizer).
SEAL trains it the e-prop way: the input-layer learning signal

    L_in = Win^T · L_j

(one more hop of symmetric feedback, the same locality approximation the paper
makes for L_j) is injected at the rates p, and ordinary autograd computes the
local gradient through the feedforward conv stack — no BPTT needed, since the
CNN is memoryless within a frame. The δ-gating and γλ trace filtering happen
in the ObGD optimizer (model/optim.py), exactly as for the readout.

A frozen-random encoder is an information bottleneck: task-relevant structure
(ball/paddle) may not survive the random projection, capping sample efficiency
regardless of the learning rule downstream.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikingCNN(nn.Module):
    """Trainable stride-conv front-end -> input spike population.

    Input:  [1, 1, 84, 84]  (normalized grayscale frame)
    Output: input spikes [n_input_neurons] (Bernoulli, per ms)

    Args:
        conv_layers: tuple of (in_ch, out_ch, kernel, stride)
        target_rate: kept for interface compat (homeostasis is via normalization)
        gain: encoding gain on the normalized features
        max_p: cap on per-ms spike probability (soft when trainable)
        seed: init seed
        trainable: if False, freeze weights (ablation: frozen random encoder)
    """
    def __init__(self, conv_layers: tuple, target_rate: float = 0.10,
                 gain: float = 0.15, max_p: float = 0.3, seed: int = 0,
                 trainable: bool = True):
        super().__init__()
        self.target_rate = float(target_rate)
        self.gain = float(gain)        # FIXED global encoding gain
        self.max_p = float(max_p)      # cap per-ms spike probability
        self.trainable = bool(trainable)
        torch.manual_seed(seed)
        self.convs = nn.ModuleList()
        in_ch = conv_layers[0][0]
        H = W = 84
        for (ic, oc, k, st) in conv_layers:
            conv = nn.Conv2d(ic, oc, kernel_size=k, stride=st, bias=True)
            nn.init.kaiming_uniform_(conv.weight, a=0.5)
            nn.init.zeros_(conv.bias)
            conv.weight.requires_grad_(self.trainable)
            conv.bias.requires_grad_(self.trainable)
            self.convs.append(conv)
            in_ch = oc
            H = (H - k) // st + 1
            W = (W - k) // st + 1
        self.flat_dim = in_ch * H * W
        self._H, self._W = H, W

    @property
    def n_input_neurons(self) -> int:
        return self.flat_dim

    def rates(self, frame: torch.Tensor) -> torch.Tensor:
        """Differentiable spike-rate vector p [flat_dim] for one frame.

        Features are normalized to zero mean / unit std over the population,
        then ONLY above-average features fire, with probability proportional
        to (gain · relu(feats_n)). Sparse by construction (dark background ->
        no spikes; bright objects -> spikes).

        When trainable, the max_p cap is SOFT (tanh) so gradients do not die
        in the saturated region (hard clamp has zero derivative there).
        """
        h = frame
        for conv in self.convs:
            h = F.leaky_relu(conv(h))
        feats = h.flatten(1).squeeze(0)          # [flat_dim]
        mu = feats.mean(); std = feats.std() + 1e-6
        feats_n = (feats - mu) / std
        x = self.gain * F.relu(feats_n)
        if self.trainable:
            p = self.max_p * torch.tanh(x / self.max_p)   # soft cap
        else:
            p = torch.clamp(x, max=self.max_p)            # hard cap (frozen)
        return p

    def forward(self, frame: torch.Tensor, dt_ms: float = 1.0) -> torch.Tensor:
        """One ms of encoding. Returns input spikes [n_input_neurons].

        Samples Bernoulli(rates) under no_grad — the gradient path for the
        conv weights is built separately in agent.learn() via rates() and the
        input-layer learning signal L_in (no autograd through sampling).

        Args:
            frame: [1, 1, 84, 84] normalized grayscale
            dt_ms: timestep (unused for Bernoulli; rate is per-ms)
        """
        with torch.no_grad():
            p = self.rates(frame)
            spikes = torch.bernoulli(p)
        return spikes

    def reset(self):
        """Reset homeostatic state (call on episode boundary, optional)."""
        # keep gain + rate_ewma across episodes (they track the input statistics)
        pass
