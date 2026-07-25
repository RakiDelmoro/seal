"""SEAL agent — Stream Q with event-driven encoder.

Architecture:
  4 stacked frames [1, 4, 84, 84]  (velocity is in the input, no RNN)
    -> EventConv2d(4->32,  8, s5) -> LeakyReLU + LayerNorm
    -> EventConv2d(32->64, 4, s3) -> LeakyReLU + LayerNorm
    -> EventConv2d(64->64, 3, s2) -> LeakyReLU + LayerNorm
    -> flatten -> EventLinear(256) -> LeakyReLU + LayerNorm
    -> heads (all pure linear via affine-free LayerNorm):
       Q:       Linear(256, 6)   # Q-values per action (argmax = greedy action)
       GVF bank: 4× Linear(256,1) # game-agnostic TD(λ) value predictions
                                  # (motion_density, pos/neg_reward, motion_spread)

Stream Q (off-policy): δ = r + γ·max_a'Q(s',a') - Q(s,a). Bootstraps from the
greedy next Q regardless of the action taken, so the agent learns greedy
Q-values even during epsilon-greedy exploration. Traces reset on exploration
actions (off-policy correction). No policy gradient, no entropy — exploration
is handled by epsilon-greedy.

Two optimizers:
  * Encoder  — AdaptiveObGD (κ-bound + Adam-style 2nd-moment normalization).
  * Heads    — SwiftTD (True Online TD(λ) + IDBD per-feature step sizes +
               overshoot bound + step-size decay). Exact for linear heads.

Plus: eligibility traces (λ=0.8) + affine-free LayerNorm + 90% sparse init +
per-element event thresholds + utility gate + dead-unit regeneration.
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
from model.swift_td import SwiftTD
from model.gvf import DEFAULT_GVFS, gvf_lams, gvf_weights, n_gvfs, compute_cumulants
from model.utility import UtilityTracker
from model.metrics import flops_event_layers, dense_flops_conv
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
        in_ch = cfg.conv_layers[0][0]   # 4 (frame stack)
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
    """Q head + game-agnostic GVF bank. (No separate value head — Q IS the value.)

    LayerNorm here is affine-free: it has no learnable params, so the head
    input `z` is a deterministic function of the encoder trunk. This makes
    every head a PURE LINEAR learner (`out = W·z + b`) — the exact setting
    SwiftTD is derived for. The heads are updated by SwiftTD (per-feature
    IDBD step sizes + True Online TD(λ) + overshoot bound); the encoder is
    updated by AdaptiveObGD via gradients flowing through these
    (frozen-per-step) heads.

    The GVF bank (model/gvf.py) replaces the old Pong-specific aux head:
    each GVF is a linear TD(λ) prediction of a discounted future cumulant
    derived only from the event mask + reward, so it transfers across games.
    """
    def __init__(self, cfg: Config, n_actions: int, n_gvfs: int):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.trunk_dim, elementwise_affine=False)
        self.q = nn.Linear(cfg.trunk_dim, n_actions)
        self.gvf = nn.ModuleList([nn.Linear(cfg.trunk_dim, 1)
                                  for _ in range(n_gvfs)])

    def forward(self, h):
        z = self.ln(h)
        logits = self.q(z)                       # Q-values per action
        gvf_preds = torch.cat([g(z) for g in self.gvf], dim=1)  # [1, n_gvfs]
        return logits, gvf_preds, z


@dataclass
class Transition:
    """Outputs from one forward pass, held for the next-step TD update."""
    logits: torch.Tensor         # Q-values [1, n_actions]
    gvf_preds: torch.Tensor      # GVF predictions [1, n_gvfs] (for bootstrap)
    action: int
    feats: torch.Tensor          # encoder trunk features (256) — diagnostics / utility
    head_features: torch.Tensor  # LayerNormed trunk (input to the linear heads) — SwiftTD φ
    event_mask: torch.Tensor     # first-conv event mask [1,C,H,W] — GVF cumulant source
    is_exploration: bool = False   # epsilon-greedy random action (skip policy grad)


class SEALAgent(nn.Module):
    """SEAL: event-driven encoder + Stream Q + ObGD + eligibility traces.

    No GRU. No BPTT. No hidden state. Pure feedforward: EMA -> event convs
    -> event linear -> Q/GVF heads. Temporal info is in the 4-frame-stack input.
    """
    def __init__(self, cfg: Config, n_actions: int, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = device
        self.gvfs = DEFAULT_GVFS
        self.n_gvfs = n_gvfs(self.gvfs)
        self.encoder = EventEncoder(cfg)
        self.heads = Heads(cfg, n_actions, self.n_gvfs)
        apply_sparse_init(self, sparsity=0.9)

        # ---- parameter split: encoder + Q head on AdaptiveObGD, GVFs on SwiftTD ----
        # AdaptiveObGD owns the nonlinear encoder AND the Q head. The Q head
        # bootstraps from max_a' Q(s',a') — nonlinear in the weights, the
        # deadly-triad offender — so it needs AdaptiveObGD's κ-bound (fixed α
        # that cancels in the bound-active regime), NOT SwiftTD's IDBD which
        # would AMPLIFY the max-overestimation cascade by growing step sizes.
        # SwiftTD owns only the GVF heads: pure linear PREDICTION (each
        # bootstraps from its own next prediction, no max) — the exact setting
        # True Online TD(λ) is valid for.
        gvf_param_ids = set(id(p) for p in self.heads.gvf.parameters())
        self.obgd_params = [p for p in self.parameters()
                            if p.requires_grad and id(p) not in gvf_param_ids]
        self.head_params = [p for p in self.parameters()
                            if p.requires_grad and id(p) in gvf_param_ids]
        # `self.params` = AdaptiveObGD params (autograd.grad target + utility
        # gate). GVF heads are owned by SwiftTD and never see an autograd update.
        self.params = self.obgd_params

        self.opt = AdaptiveObGD(self.obgd_params, alpha=cfg.alpha,
                                kappa=cfg.kappa, lam=cfg.lam, gamma=cfg.gamma,
                                beta2=cfg.beta2, eps=cfg.eps)
        self.utility = UtilityTracker(self.obgd_params, decay=cfg.utility_decay,
                                      tau_low=cfg.utility_tau_low,
                                      n_trunk_units=cfg.trunk_dim)
        # SwiftTD on the GVF bank only (True Online TD(λ) + IDBD + bound +
        # decay). Seeded from the sparse-initialized GVF weights.
        self.swift = SwiftTD(self.heads.gvf, cfg, gvf_lams=gvf_lams(self.gvfs))
        self.swift.load_from_params()
        self.gvf_weights = gvf_weights(self.gvfs)

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
        self.swift.reset_all()

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
        logits, gvf_preds, z = self.heads(feats)
        is_expl = False
        if np.random.rand() < self.epsilon:
            a = int(np.random.randint(0, self.n_actions))
            is_expl = True
        else:
            a = int(logits.detach().argmax().item())
        event_mask = self.encoder.first_conv_mask()
        with torch.no_grad():
            f_np = feats.detach().squeeze(0).cpu().numpy()
            self.utility.update_unit_utility(f_np)
            active = (np.abs(f_np) > 1e-3)
            self.since_active[active] = 0
            self.since_active[~active] += 1
        tr = Transition(logits=logits, gvf_preds=gvf_preds, action=a,
                        feats=feats.detach(), head_features=z.detach(),
                        event_mask=event_mask.detach(),
                        is_exploration=is_expl)
        return a, tr

    # --------------------------------------------------------------- learn
    def learn(self, pending: Transition, r: float,
              next_pending: Transition, done: bool,
              reset_on_done: bool = True):
        """Streaming TD(λ) update for `pending` using the transition (s,a,r,s').

        `next_pending` is the Transition from the next observation (s'); pass
        None on a terminal step. Bootstrap values are read from it: Q's
        bootstrap = max_a' Q(s',a') (clipped to ±q_clip), each GVF's bootstrap
        = that GVF's own prediction at s'. All are 0 if done.

        Two optimizers, split at the control/prediction boundary:
          * ENCODER + Q HEAD (AdaptiveObGD): traced TD(λ) on the gradient of
            `loss = -Q(s,a) + Σ_k gvf_weight_k·½(GVF_k - c_k)²`. The Q head
            lives here (not SwiftTD) because its max_a' Q(s',a') bootstrap is
            nonlinear in the weights — the deadly-triad offender — and
            AdaptiveObGD's κ-bound (fixed α that cancels) damps the
            overestimation cascade instead of amplifying it. The GVF terms
            shape the encoder representation (game-agnostic).
          * GVF BANK (SwiftTD): exact True Online TD(λ) + IDBD per-feature
            step sizes + overshoot bound + step-size decay. Pure linear
            PREDICTION (each GVF bootstraps from its own next prediction, no
            max) — the exact setting True Online TD(λ) is valid for.
            δ_k = c_k + γ·GVF_k(s') − GVF_k(s), each with its own λ.

        δ (Q's) is NOT in the encoder/Q loss (AdaptiveObGD multiplies the
        gradient by δ internally); SwiftTD carries each GVF's own δ into its
        linear update.
        """
        cfg = self.cfg
        q_sa = pending.logits[0, pending.action]
        v_next_q = 0.0 if done else float(next_pending.logits.detach()[0].max().item())
        # overestimation guard: clip the bootstrap to the physically possible
        # return range (Pong returns are ±21).
        v_next_q = max(-cfg.q_clip, min(cfg.q_clip, v_next_q))
        td_err = float(r + cfg.gamma * v_next_q * (0.0 if done else 1.0)
                       - float(q_sa.detach()))

        self.last_td_err = td_err
        self.last_entropy = 0.0          # n/a for Stream Q (epsilon-greedy explores)
        self.last_v = float(q_sa.detach().item())

        # ---- off-policy / terminal trace reset (Stream Q correction) ----
        reset = bool(done and reset_on_done) or pending.is_exploration

        # ---- GVF cumulants (game-agnostic: event mask + reward) ----
        cumulants = compute_cumulants(pending.event_mask, float(r), self.gvfs)
        if done:
            v_next_gvfs = torch.zeros(self.n_gvfs)
        else:
            v_next_gvfs = next_pending.gvf_preds.detach().reshape(-1)

        # ---- encoder + Q head backward + AdaptiveObGD step ----
        # GVF shaping: MSE between each GVF's prediction and its cumulant,
        # summed with per-GVF weights. (Semi-gradient: cumulants detached.)
        gvf_pred = pending.gvf_preds.to(pending.head_features.device).reshape(-1)
        gvf_loss = 0.5 * ((gvf_pred - cumulants.to(gvf_pred.device)) ** 2)
        gvf_loss = float(cfg.gvf_weight) * (gvf_loss * torch.tensor(
            self.gvf_weights, device=gvf_pred.device, dtype=gvf_pred.dtype)).sum()
        loss = (-q_sa + gvf_loss)
        grads = torch.autograd.grad(loss, self.obgd_params, allow_unused=True,
                                    retain_graph=False)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, self.obgd_params)]
        gates = self.utility.update_param_utility(td_err, self.opt.traces)
        self.opt.step(td_err, grads, reset_traces=reset, update_mask=gates)

        # ---- GVF heads: SwiftTD (True Online TD(λ) + IDBD + bound + decay) ----
        self.swift.step_gvfs(pending.head_features, cumulants, v_next_gvfs,
                             done=done, reset=reset)

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
            for lin in [self.heads.q, *self.heads.gvf]:
                lin.weight[:, dead] = 0.0
            # reinit incoming weights to dead units (EventLinear output rows)
            fc_w = self.encoder.fc.weight
            for j in dead:
                if j < fc_w.shape[0]:
                    b = (1.0 / fc_w.shape[1]) ** 0.5
                    fc_w[j] = torch.empty_like(fc_w[j]).uniform_(-b, b)
            self.since_active[dead] = 0
        # GVF head weight columns were zeroed above — resync SwiftTD's GVF
        # weight buffers so they mirror the (frozen) GVF params. β/z/h state
        # for the affected feature indices persists, matching how AdaptiveObGD's
        # traces persist across the fc-row/Q-row reinit. (The Q head is owned
        # by AdaptiveObGD, so it is NOT resynced here — its weights are already
        # the source of truth.)
        self.swift.load_from_params()
        self._last_regen = len(dead)

    # ----------------------------------------------------------------- stats
    def event_flops(self) -> int:
        return self.encoder.flops()

    def dense_flops(self) -> int:
        d = self.encoder.dense_flops()
        d += self.cfg.trunk_dim * self.n_actions * 2
        d += self.cfg.trunk_dim * self.n_gvfs * 2
        return d
