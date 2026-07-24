"""SEAL agent — Stream Q with event-driven encoder.

Architecture:
  1 EMA channel [1,1,84,84]  (single EMA, alpha=0.2, ~4-frame trail)
    -> EventConv2d(1->16, 8, s5)  -> LeakyReLU + LayerNorm
    -> EventConv2d(16->32, 4, s3) -> LeakyReLU + LayerNorm
    -> EventConv2d(32->32, 3, s2) -> LeakyReLU + LayerNorm
    -> flatten -> EventLinear(256) -> LeakyReLU + LayerNorm
    -> heads:
       Q:     Linear(256, 6)   # Q-values per action (argmax = greedy action)
       aux:   Linear(256, 3)   # (ball_x, ball_y, paddle_contact)

Stream Q (off-policy): δ = r + γ·max_a'Q(s',a') - Q(s,a). Bootstraps from the
greedy next Q regardless of the action taken, so the agent learns greedy
Q-values even during epsilon-greedy exploration. Traces reset on exploration
actions (off-policy correction). No policy gradient, no entropy — exploration
is handled by epsilon-greedy.

ObGD + eligibility traces (λ=0.8) + LayerNorm + 90% sparse init + per-element
thresholds + utility gate + dead-unit regeneration.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model.event_layers import EventConv2d, EventLinear
from model.thresholds import PerPixelThreshold
from model.optimizers import AdaptiveObGD
from model.utility import UtilityTracker
from model.metrics import extract_aux_targets, flops_event_layers, dense_flops_conv
from model.sparse_init import apply_sparse_init


def _leaky(x):
    return F.leaky_relu(x)


class EventEncoder(nn.Module):
    """Event-driven conv trunk + EventLinear -> 256-dim features.

    Takes 1 EMA channel [1,1,84,84]. The event deltas are frame-to-frame
    changes of the EMA trail — velocity is directly in the event input.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.event_layers = nn.ModuleList()
        in_ch = cfg.conv_layers[0][0]   # 1 (single EMA)
        H = W = 84
        prev_ch = in_ch
        for (ic, oc, k, st) in cfg.conv_layers:
            th = PerPixelThreshold(k=cfg.perpixel_k,
                                   warmup_steps=cfg.perpixel_warmup,
                                   floor=cfg.perpixel_floor)
            self.event_layers.append(EventConv2d(ic, oc, k, st, th))
            prev_ch = oc
            H = (H - k) // st + 1
            W = (W - k) // st + 1
        self.flat_dim = prev_ch * H * W
        self.th_lin = PerPixelThreshold(k=cfg.perpixel_k,
                                        warmup_steps=cfg.perpixel_warmup,
                                        floor=cfg.perpixel_floor)
        self.fc = EventLinear(self.flat_dim, cfg.trunk_dim, self.th_lin)
        self.event_layers.append(self.fc)
        self._flat_dim = self.flat_dim
        self._conv_cfg = cfg.conv_layers
        self.record_acts = False
        self.last_acts = []

    def _ln(self, x):
        """LayerNorm: no learnable scale/bias. Applied to pre-act."""
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
        for (ic, oc, k, st) in self._conv_cfg:
            total += dense_flops_conv(ic, oc, k, st, H, W)
            H = (H - k) // st + 1
            W = (W - k) // st + 1
        total += self._flat_dim * self.cfg.trunk_dim * 2
        return total


class Heads(nn.Module):
    """Q + aux heads with LayerNorm. (No separate value head — Q IS the value.)"""
    def __init__(self, cfg: Config, n_actions: int):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.trunk_dim)
        self.q = nn.Linear(cfg.trunk_dim, n_actions)
        self.aux = nn.Linear(cfg.trunk_dim, cfg.aux_dim)

    def forward(self, h):
        z = self.ln(h)
        logits = self.q(z)            # Q-values per action
        aux = self.aux(z)
        return logits, aux


@dataclass
class Transition:
    """Outputs from one forward pass, held for the next-step TD update."""
    logits: torch.Tensor
    aux: torch.Tensor
    action: int
    aux_targets: torch.Tensor
    feats: torch.Tensor
    is_exploration: bool = False   # epsilon-greedy random action (skip policy grad)


class SEALAgent(nn.Module):
    """SEAL: event-driven encoder + Stream Q + ObGD + eligibility traces.

    No GRU. No BPTT. No hidden state. Pure feedforward: EMA -> event convs
    -> event linear -> Q/aux heads. Temporal info is in the EMA input.
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
        self.opt = AdaptiveObGD(self.params, alpha=cfg.alpha, kappa=cfg.kappa,
                                lam=cfg.lam, gamma=cfg.gamma,
                                beta2=cfg.beta2, eps=cfg.eps)
        self.utility = UtilityTracker(self.params, decay=cfg.utility_decay,
                                      tau_low=cfg.utility_tau_low,
                                      n_trunk_units=cfg.trunk_dim)
        self.since_active = np.zeros(cfg.trunk_dim, dtype=np.int64)
        self.global_step = 0
        self.epsilon = float(cfg.epsilon_start)

    def _update_epsilon(self):
        """Linear decay from epsilon_start to epsilon_end over exploration_fraction."""
        cfg = self.cfg
        duration = max(1, int(cfg.exploration_fraction * cfg.total_frames))
        slope = (cfg.epsilon_end - cfg.epsilon_start) / duration
        self.epsilon = max(slope * self.global_step + cfg.epsilon_start, cfg.epsilon_end)

    # ------------------------------------------------------------------ state
    def reset_episode(self):
        """Reset encoder caches + traces on episode boundary."""
        self.encoder.reset_cache()
        self.opt.reset()

    def warmup_forward(self, obs):
        """Forward one obs through encoder (advancing caches + thresholds), no
        learning. Used during normalizer warmup."""
        x = self._to_obs(obs)
        _ = self.encoder(x)

    def reset_after_warmup(self):
        self.reset_episode()
        self.since_active[:] = 0
        self.global_step = 0

    def _to_obs(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        # HWC -> CHW: gym image obs are (H, W, C) with H==W==84 and C < 84.
        # CHW (C, 84, 84) has C < 84 in dim 0, not dim 2, so is left alone.
        if obs.dim() == 3 and obs.shape[0] == obs.shape[1] \
                and obs.shape[-1] <= obs.shape[0] and obs.shape[-1] != obs.shape[0]:
            obs = obs.permute(2, 0, 1)  # HWC -> CHW
        if obs.dim() == 2:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 3:
            obs = obs.unsqueeze(0)  # [1,C,H,W]
        return obs.contiguous().float().to(self.device)

    # ----------------------------------------------------------- forward/act
    def act(self, obs) -> tuple:
        """Forward one observation; select an action; return (action, Transition).

        Epsilon-greedy: with probability ε, take a uniform random action
        (exploration); otherwise take argmax Q(s,a) (greedy). The greedy
        action is what we LEARN (Q-learning), the random action is what we
        EXECUTE during exploration. Traces reset on random actions (learn()).
        """
        x = self._to_obs(obs)
        feats = self.encoder(x)
        logits, aux = self.heads(feats)
        is_expl = False
        if np.random.rand() < self.epsilon:
            a = int(np.random.randint(0, self.n_actions))
            is_expl = True
        else:
            a = int(logits.detach().argmax().item())
        aux_targets = extract_aux_targets(self.encoder.first_conv_mask(), x.shape)
        with torch.no_grad():
            f_np = feats.detach().squeeze(0).cpu().numpy()
            self.utility.update_unit_utility(f_np)
            active = (np.abs(f_np) > 1e-3)
            self.since_active[active] = 0
            self.since_active[~active] += 1
        tr = Transition(logits=logits, aux=aux, action=a,
                        aux_targets=aux_targets.detach(), feats=feats.detach(),
                        is_exploration=is_expl)
        return a, tr

    def bootstrap(self, next_tr: Transition, done: bool) -> float:
        """Bootstrap value for the TD target: max_a' Q(s',a') (greedy next Q)."""
        if done:
            return 0.0
        return float(next_tr.logits.detach()[0].max().item())

    # --------------------------------------------------------------- learn
    def learn(self, pending: Transition, r: float, v_next, done: bool,
              reset_on_done: bool = True):
        """TD(λ) Stream Q update for `pending` with bootstrap v_next.

        δ = r + γ·max_a'Q(s',a')·(1-done) - Q(s,a)
        loss = -Q(s,a) + aux_weight·MSE(aux, aux_targets)
        (ObGD multiplies the gradient by δ internally; no delta in the loss.)
        Traces reset on exploration actions (off-policy correction).
        """
        cfg = self.cfg
        v_next_t = torch.as_tensor(v_next, dtype=torch.float, device=pending.logits.device) \
            if not isinstance(v_next, torch.Tensor) else v_next.detach()
        q_sa = pending.logits[0, pending.action]
        td_err = float(r + cfg.gamma * float(v_next_t) * (0.0 if done else 1.0)
                       - float(q_sa.detach()))

        aux_loss_t = F.mse_loss(pending.aux, pending.aux_targets.to(pending.aux.device))
        loss = (-q_sa + cfg.aux_weight * aux_loss_t)

        self.last_td_err = td_err
        self.last_entropy = 0.0          # n/a for Stream Q (epsilon-greedy explores)
        self.last_v = float(q_sa.detach().item())

        # ---- backward (one-step; no BPTT, no recurrent graph) ----
        grads = torch.autograd.grad(loss, self.params, allow_unused=True,
                                    retain_graph=False)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, self.params)]

        # ---- utility gate + ObGD step ----
        gates = self.utility.update_param_utility(td_err, self.opt.traces)
        # Reset traces on exploration actions (off-policy correction, matching
        # paper stream_q `reset=(done or is_nongreedy)`): the random action's
        # gradient should not accumulate credit for past greedy choices.
        reset = bool(done and reset_on_done) or pending.is_exploration
        self.opt.step(td_err, grads, reset_traces=reset,
                      update_mask=gates)

        # ---- epsilon decay + plasticity: regenerate dead units ----
        self.global_step += 1
        self._update_epsilon()
        if self.global_step % cfg.regen_every == 0:
            self._regenerate()

        return td_err

    def _regenerate(self):
        """Regenerate bottom-1% dead trunk units (ReDo-style)."""
        cfg = self.cfg
        dead = self.utility.dormant_units(self.since_active, cfg.dormant_silence_steps)
        if len(dead) == 0:
            return
        with torch.no_grad():
            for lin in (self.heads.q, self.heads.aux):
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
        d += self.cfg.trunk_dim * self.n_actions * 2
        d += self.cfg.trunk_dim * self.cfg.aux_dim * 2
        return d
