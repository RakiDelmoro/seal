"""StreamingActorCritic — SEAL architecture (event-driven + frame stacking).

Architecture:
  4 stacked frames [1,4,84,84] (velocity is in the input, no RNN)
    -> EventConv2d(4->16, 8, s5) -> LeakyReLU + LayerNorm
    -> EventConv2d(16->32, 4, s3) -> LeakyReLU + LayerNorm
    -> EventConv2d(32->32, 3, s2) -> LeakyReLU + LayerNorm
    -> flatten -> EventLinear(256) -> LeakyReLU + LayerNorm
    -> heads:
       value:  Linear(256, 1)
       policy: Linear(256, 6)    # softmax, adaptive entropy
       aux:    Linear(256, 3)    # (ball_x, ball_y, paddle_contact)

No GRU. No BPTT. No hidden state. Pure feedforward with frame stacking.
The event deltas are 4 channels of frame-to-frame motion — velocity is baked
into the event mechanism itself.

Paper-faithful recipe: ObGD + eligibility traces (λ=0.8) + LayerNorm + sparse
init 90% + adaptive entropy (sign(δ)·τ·∇H). See optimizers.py for the ObGD
α-cancels derivation and agent.py learn() for the paper-exact loss.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import Config
from model.event_layers import EventConv2d, EventLinear
from model.thresholds import HomeostaticThreshold
from model.optimizers import ObGD
from model.utility import UtilityTracker, regenerate_dead_units
from model.metrics import extract_aux_targets, flops_event_layers, dense_flops_conv
from model.sparse_init import apply_sparse_init


def _leaky(x):
    return F.leaky_relu(x)


class EventEncoder(nn.Module):
    """Event-driven conv trunk + EventLinear -> 256-dim features.

    Takes 4 stacked frames [1,4,84,84]. The event deltas are 4 channels of
    frame-to-frame motion — each channel captures a different timestep's
    change, so velocity is directly in the event input. No GRU needed.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.event_layers = nn.ModuleList()
        in_ch = cfg.conv_layers[0][0]  # 4 (frame stack)
        H = W = 84
        prev_ch = in_ch
        for (ic, oc, k, st) in cfg.conv_layers:
            th = HomeostaticThreshold(
                target_lo=cfg.threshold_target_lo,
                target_hi=cfg.threshold_target_hi,
                adapt_rate=cfg.threshold_adapt_rate,
                theta0=cfg.threshold_theta0,
            )
            self.event_layers.append(EventConv2d(ic, oc, k, st, th))
            prev_ch = oc
            H = (H - k) // st + 1
            W = (W - k) // st + 1
        self.flat_dim = prev_ch * H * W
        self.th_lin = HomeostaticThreshold(
            target_lo=cfg.threshold_target_lo,
            target_hi=cfg.threshold_target_hi,
            adapt_rate=cfg.threshold_adapt_rate,
            theta0=cfg.threshold_theta0,
        )
        self.fc = EventLinear(self.flat_dim, cfg.trunk_dim, self.th_lin)
        self.event_layers.append(self.fc)
        self._flat_dim = self.flat_dim
        self.record_acts = False
        self.last_acts = []

    def _ln(self, x):
        """Paper §3.3 LayerNorm: no learnable scale/bias. Applied to pre-act."""
        return F.layer_norm(x, x.shape[-1:] if x.dim() == 2 else x.shape[1:])

    def forward(self, x):
        h = x
        acts = []
        for layer in list(self.event_layers)[:-1]:
            z = layer(h)
            h = _leaky(self._ln(z))
            if self.record_acts:
                acts.append(float(h.detach().abs().mean().item()))
        h = h.flatten(1)
        z = self.fc(h)
        feats = _leaky(self._ln(z))
        if self.record_acts:
            acts.append(float(feats.detach().abs().mean().item()))
        self.last_acts = acts
        return feats

    def reset_cache(self):
        for layer in self.event_layers:
            layer.reset_cache()

    def event_rates(self):
        return [l.last_event_rate for l in self.event_layers]

    def first_conv_mask(self):
        return self.event_layers[0].last_mask

    def flops(self):
        return flops_event_layers(self.event_layers)

    def dense_flops(self):
        H = W = 84
        total = 0
        for (ic, oc, k, st) in self.cfg.conv_layers:
            total += dense_flops_conv(ic, oc, k, st, H, W)
            H = (H - k) // st + 1
            W = (W - k) // st + 1
        total += self._flat_dim * self.cfg.trunk_dim * 2
        return total


class Heads(nn.Module):
    """Value, policy, aux heads with LayerNorm."""
    def __init__(self, cfg: Config, n_actions: int):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.trunk_dim)
        self.value = nn.Linear(cfg.trunk_dim, 1)
        self.policy = nn.Linear(cfg.trunk_dim, n_actions)
        self.aux = nn.Linear(cfg.trunk_dim, cfg.aux_dim)

    def forward(self, h):
        z = self.ln(h)
        v = self.value(z).squeeze(-1)
        logits = self.policy(z)
        aux = self.aux(z)
        return v, logits, aux


@dataclass
class Transition:
    """Outputs from one forward pass, held for the next-step TD update."""
    v: torch.Tensor
    logits: torch.Tensor
    aux: torch.Tensor
    action: int
    aux_targets: torch.Tensor
    feats: torch.Tensor


class StreamingActorCritic(nn.Module):
    """SEAL: event-driven encoder + frame stacking + ObGD + traces.

    No GRU. No hidden state. Pure feedforward: 4 stacked frames -> event convs
    -> event linear -> heads. Temporal info is in the input (frame stack), not
    in recurrent memory.
    """
    def __init__(self, cfg: Config, n_actions: int, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = device
        self.encoder = EventEncoder(cfg)
        self.heads = Heads(cfg, n_actions)
        apply_sparse_init(self, sparsity=0.9)

        self.params = [p for p in self.parameters() if p.requires_grad]
        self.opt = ObGD(self.params, alpha=cfg.alpha, kappa=cfg.kappa,
                        lam=cfg.lam, gamma=cfg.gamma)
        self.utility = UtilityTracker(self.params, decay=cfg.utility_decay,
                                      tau_low=cfg.utility_tau_low,
                                      n_trunk_units=cfg.trunk_dim)
        self.since_active = np.zeros(cfg.trunk_dim, dtype=np.int64)
        self.global_step = 0

    # ------------------------------------------------------------------ state
    def reset_episode(self):
        """Reset encoder caches + traces on episode boundary."""
        self.encoder.reset_cache()
        self.opt.reset()

    def warmup_forward(self, obs):
        """Forward one obs through encoder (advancing caches + homeostat), no
        learning. Used during normalizer warmup."""
        x = self._to_obs(obs)
        _ = self.encoder(x)
        for layer in self.encoder.event_layers:
            layer.threshold.update(layer.last_event_rate)

    def reset_after_warmup(self):
        self.reset_episode()
        self.since_active[:] = 0
        self.global_step = 0

    def _to_obs(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        if obs.dim() == 3 and obs.shape[-1] in (1, 3, 4):
            obs = obs.permute(2, 0, 1)  # HWC -> CHW
        if obs.dim() == 2:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 3:
            obs = obs.unsqueeze(0)  # [1,C,H,W]
        return obs.contiguous().float().to(self.device)

    # ----------------------------------------------------------- forward/act
    def act(self, obs) -> tuple:
        """Forward one observation; sample an action; return (action, Transition)."""
        x = self._to_obs(obs)
        feats = self.encoder(x)
        # No GRU — heads read directly from encoder features.
        v, logits, aux = self.heads(feats)
        dist = Categorical(logits=logits)
        a = int(dist.sample().item())
        aux_targets = extract_aux_targets(self.encoder.first_conv_mask(), x.shape)
        with torch.no_grad():
            f_np = feats.detach().squeeze(0).cpu().numpy()
            self.utility.update_unit_utility(f_np)
            active = (np.abs(f_np) > 1e-3)
            self.since_active[active] = 0
            self.since_active[~active] += 1
        tr = Transition(v=v.squeeze().detach() if v.dim() else v,
                        logits=logits, aux=aux, action=a,
                        aux_targets=aux_targets.detach(), feats=feats.detach())
        tr.v = v if v.dim() == 0 else v.squeeze()
        return a, tr

    # --------------------------------------------------------------- learn
    def learn(self, pending: Transition, r: float, v_next, done: bool,
              reset_on_done: bool = True):
        """TD(λ) update for the transition `pending` with bootstrap v_next."""
        cfg = self.cfg
        v = pending.v
        v_next_t = torch.as_tensor(v_next, dtype=torch.float, device=v.device) \
            if not isinstance(v_next, torch.Tensor) else v_next.detach()
        td_err = float(r + cfg.gamma * float(v_next_t) * (0.0 if done else 1.0) - float(v.detach()))

        # ---- loss (paper-exact, matches official stream_ac code) ----
        # Do NOT multiply value/policy/entropy by td_err here. ObGD multiplies
        # by delta INTERNALLY. Putting delta in the loss double-counts it.
        # Entropy term: sign(delta) in loss -> ObGD×delta gives |delta|·ascent.
        dist = Categorical(logits=pending.logits)
        logp_a = dist.log_prob(torch.tensor(pending.action, device=pending.logits.device))
        entropy = dist.entropy().mean()
        sign_delta = 1.0 if td_err > 0 else (-1.0 if td_err < 0 else 1e-3)
        aux_loss_t = F.mse_loss(pending.aux, pending.aux_targets.to(pending.aux.device))
        loss = (-v - logp_a - cfg.entropy_coeff * sign_delta * entropy
                + cfg.aux_weight * aux_loss_t)

        self.last_td_err = td_err
        self.last_entropy = float(entropy.detach().item())
        self.last_v = float(v.detach().item())

        # ---- backward (one-step; no BPTT, no recurrent graph) ----
        grads = torch.autograd.grad(loss, self.params, allow_unused=True,
                                    retain_graph=False)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, self.params)]

        # ---- utility gate + ObGD step ----
        gates = self.utility.update_param_utility(td_err, self.opt.traces)
        self.opt.step(td_err, grads, reset_traces=bool(done and reset_on_done),
                      update_mask=gates)

        # ---- homeostasis ----
        for layer in self.encoder.event_layers:
            layer.threshold.update(layer.last_event_rate)

        # ---- plasticity: regenerate dead units ----
        self.global_step += 1
        if self.global_step % cfg.regen_every == 0:
            self._regenerate()

        return td_err

    def _regenerate(self):
        """Regenerate bottom-1% dead trunk units."""
        cfg = self.cfg
        dead = self.utility.dormant_units(self.since_active, cfg.dormant_silence_steps)
        if len(dead) == 0:
            return
        with torch.no_grad():
            for lin in (self.heads.value, self.heads.policy, self.heads.aux):
                lin.weight[:, dead] = 0.0
            # reinit incoming weights to dead units (EventLinear output rows)
            fc_w = self.encoder.fc.weight
            for j in dead:
                if j < fc_w.shape[0]:
                    b = (1.0 / fc_w.shape[1]) ** 0.5
                    fc_w[j] = torch.empty_like(fc_w[j]).uniform_(-b, b)
            self.since_active[dead] = 0
        self._last_regen = len(dead)

    # ----------------------------------------------------------------- stats
    def event_flops(self) -> int:
        return self.encoder.flops()

    def dense_flops(self) -> int:
        d = self.encoder.dense_flops()
        d += self.cfg.trunk_dim * 1 * 2 + self.cfg.trunk_dim * self.n_actions * 2
        d += self.cfg.trunk_dim * self.cfg.aux_dim * 2
        return d
