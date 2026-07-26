"""SEAL agent — Stream Q with event-driven encoder + SPR auxiliary loss.

Architecture:
  4 stacked frames [1, 4, 84, 84]  (velocity is in the input, no RNN)
    -> EventConv2d(4->32,  8, s5) -> LeakyReLU + LayerNorm
    -> EventConv2d(32->64, 4, s3) -> LeakyReLU + LayerNorm
    -> EventConv2d(64->64, 3, s2) -> LeakyReLU + LayerNorm
    -> flatten -> EventLinear(256) -> LeakyReLU + LayerNorm
    -> LayerNorm(256, affine-free) -> z
    -> Q head: Linear(256, n_actions)   (argmax = greedy action)

Auxiliary (SPR, arXiv:2602.09396):
  A transition model predicts z_{t+1..t+K} from z_t + actions. A momentum
  (EMA) target encoder produces stop-gradient target latents. The SPR loss
  (negative cosine similarity) shapes the encoder to be predictive of its own
  future. The SPR gradient is orthogonalized against the Q gradient (so it
  only shapes the encoder in non-conflicting directions) and norm-bounded
  (so it can't destabilize the trunk).

Stream Q (off-policy): δ = r + γ·max_a'Q(s',a') - Q(s,a). ε-greedy.
Traces reset on exploration actions (off-policy correction).

Optimizer:
  * Encoder + Q head — AdaptiveObGD (κ-bound + 2nd-moment normalization).
  * SPR (transition + projection) — SGD with orthogonal projection vs Q.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model.event_layers import EventConv2d, EventLinear
from model.thresholds import PerPixelThreshold
from model.optimizers import AdaptiveObGD
from model.target_encoder import TargetEncoder
from model.spr import TransitionModel, ProjectionHead, spr_loss
from model.utility import UtilityTracker
from model.metrics import flops_event_layers, dense_flops_conv
from model.sparse_init import apply_sparse_init


def _leaky(x):
    return F.leaky_relu(x)


class EventEncoder(nn.Module):
    """Event-driven conv trunk + EventLinear -> 256-dim features."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.event_layers = nn.ModuleList()
        in_ch = cfg.conv_layers[0][0]
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
    """Q head only. LayerNorm is affine-free → Q is a pure linear learner."""

    def __init__(self, cfg: Config, n_actions: int):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.trunk_dim, elementwise_affine=False)
        self.q = nn.Linear(cfg.trunk_dim, n_actions)

    def forward(self, h):
        z = self.ln(h)
        logits = self.q(z)
        return logits, z


@dataclass
class Transition:
    """Outputs from one forward pass, held for the TD update + SPR delay queue."""
    logits: torch.Tensor
    action: int
    feats: torch.Tensor         # trunk features [1, 256] (in-graph for SPR loss)
    head_features: torch.Tensor # LayerNormed z [1, 256] (SPR's latent input)
    obs: torch.Tensor           # raw input [1, 4, 84, 84] (for target encoder)
    is_exploration: bool = False


class SEALAgent(nn.Module):
    """SEAL: event-driven encoder + Stream Q (AdaptiveObGD) + SPR (orthogonal SGD)."""

    def __init__(self, cfg: Config, n_actions: int, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = device
        self.encoder = EventEncoder(cfg)
        self.heads = Heads(cfg, n_actions)
        self.spr_transition = TransitionModel(cfg.trunk_dim, n_actions)
        self.spr_projection = ProjectionHead(cfg.trunk_dim, cfg.spr_proj_dim)
        apply_sparse_init(self, sparsity=0.9)

        # ---- parameter split: encoder+Q on ObGD, SPR on SGD ----
        spr_param_ids = set(id(p) for p in self.spr_transition.parameters()) | \
                        set(id(p) for p in self.spr_projection.parameters())
        self.obgd_params = [p for p in self.parameters()
                            if p.requires_grad and id(p) not in spr_param_ids]
        self.spr_params = [p for p in self.parameters()
                           if p.requires_grad and id(p) in spr_param_ids]
        self.params = self.obgd_params

        self.opt = AdaptiveObGD(self.obgd_params, alpha=cfg.alpha,
                                kappa=cfg.kappa, lam=cfg.lam, gamma=cfg.gamma,
                                beta2=cfg.beta2, eps=cfg.eps)
        self.utility = UtilityTracker(self.obgd_params, decay=cfg.utility_decay,
                                      tau_low=cfg.utility_tau_low,
                                      n_trunk_units=cfg.trunk_dim)

        # ---- SPR target encoder (EMA copy, no gradients) ----
        self.target_enc = TargetEncoder(self.encoder, tau=cfg.spr_tau)

        # ---- SPR delay queue: hold K transitions to compute the loss K steps later ----
        self.spr_queue = deque(maxlen=cfg.spr_horizon)

        self.since_active = np.zeros(cfg.trunk_dim, dtype=np.int64)
        self.global_step = 0
        self.epsilon = float(cfg.epsilon_start)

    def _update_epsilon(self):
        cfg = self.cfg
        duration = max(1, int(cfg.exploration_fraction * cfg.total_frames))
        slope = (cfg.epsilon_end - cfg.epsilon_start) / duration
        self.epsilon = max(slope * self.global_step + cfg.epsilon_start, cfg.epsilon_end)

    # ------------------------------------------------------------------ state
    def reset_episode(self):
        self.encoder.reset_cache()
        self.opt.reset()
        self.spr_queue.clear()

    def warmup_forward(self, obs):
        x = self._to_obs(obs)
        _ = self.encoder(x)

    def reset_after_warmup(self):
        self.reset_episode()
        self.since_active[:] = 0
        self.global_step = 0

    def _to_obs(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        if obs.dim() == 3 and obs.shape[0] == obs.shape[1] \
                and obs.shape[-1] <= obs.shape[0] and obs.shape[-1] != obs.shape[0]:
            obs = obs.permute(2, 0, 1)
        if obs.dim() == 2:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 3:
            obs = obs.unsqueeze(0)
        return obs.contiguous().float().to(self.device)

    # ----------------------------------------------------------- forward/act
    def act(self, obs) -> tuple:
        x = self._to_obs(obs)
        feats = self.encoder(x)
        logits, z = self.heads(feats)
        is_expl = False
        if np.random.rand() < self.epsilon:
            a = int(np.random.randint(0, self.n_actions))
            is_expl = True
        else:
            a = int(logits.detach().argmax().item())
        with torch.no_grad():
            f_np = feats.detach().squeeze(0).cpu().numpy()
            self.utility.update_unit_utility(f_np)
            active = (np.abs(f_np) > 1e-3)
            self.since_active[active] = 0
            self.since_active[~active] += 1
        tr = Transition(logits=logits, action=a, feats=feats,
                        head_features=z, obs=x, is_exploration=is_expl)
        return a, tr

    # --------------------------------------------------------------- learn
    def learn(self, pending: Transition, r: float,
              next_pending: Transition, done: bool,
              reset_on_done: bool = True):
        """Streaming TD(λ) Q update + SPR auxiliary update (delayed K steps).

        Two gradient paths, kept separate so they don't conflict:
          * Q gradient  → AdaptiveObGD (traced TD(λ), ×δ, κ-bound, utility gate)
          * SPR gradient → SGD, orthogonalized against Q, norm-bounded

        The SPR loss at time t needs target latents from o_{t+1..t+K}, so it
        is computed K steps late via a fixed-length delay queue (NOT a replay
        buffer — just K=3 held transitions).
        """
        cfg = self.cfg
        q_sa = pending.logits[0, pending.action]
        v_next_q = 0.0 if done else float(next_pending.logits.detach()[0].max().item())
        v_next_q = max(-cfg.q_clip, min(cfg.q_clip, v_next_q))
        td_err = float(r + cfg.gamma * v_next_q * (0.0 if done else 1.0)
                       - float(q_sa.detach()))

        self.last_td_err = td_err
        self.last_entropy = 0.0
        self.last_v = float(q_sa.detach().item())

        reset = bool(done and reset_on_done) or pending.is_exploration

        # ---- Q gradient (for ObGD) ----
        grads_q = torch.autograd.grad(-q_sa, self.obgd_params, allow_unused=True,
                                      retain_graph=True)
        grads_q = [g if g is not None else torch.zeros_like(p)
                   for g, p in zip(grads_q, self.obgd_params)]

        # ---- ObGD step (Q only) ----
        gates = self.utility.update_param_utility(td_err, self.opt.traces)
        self.opt.step(td_err, grads_q, reset_traces=reset, update_mask=gates)

        # ---- SPR: enqueue this transition, compute loss if queue is full ----
        spr_grads = None
        self.spr_queue.append(pending)
        if len(self.spr_queue) == cfg.spr_horizon and not done:
            spr_grads = self._compute_spr_grads()

        # ---- SPR gradient: orthogonalize against Q, norm-bound, SGD step ----
        if spr_grads is not None:
            self._spr_step(spr_grads, grads_q)

        # ---- update the target encoder (EMA) ----
        self.target_enc.update(self.encoder)

        # ---- epsilon decay + plasticity ----
        self.global_step += 1
        self._update_epsilon()
        if self.global_step % cfg.regen_every == 0:
            self._regenerate()

        return td_err

    def _compute_spr_grads(self):
        """Compute the SPR loss gradient w.r.t. ALL learnable params (encoder+SPR).

        The transition model unrolls from the OLDEST queued transition's latent,
        predicting K steps ahead. Targets come from the EMA target encoder run
        on the queued future observations (stop-gradient).
        """
        cfg = self.cfg
        queue = list(self.spr_queue)
        K = len(queue)
        z0 = queue[0].head_features  # [1, 256], in-graph from the online encoder

        # unroll the transition model K steps
        pred_latents = []
        z_cur = z0
        for k in range(K):
            z_cur = self.spr_transition(z_cur, queue[k].action)
            pred_latents.append(self.spr_projection(z_cur))

        # target latents from the EMA target encoder (stop-gradient)
        with torch.no_grad():
            target_projs = []
            for k in range(K):
                z_tgt = self.target_enc.encode(queue[k].obs)  # [1, 256]
                # apply the SAME projection head (stop-grad) — the paper allows sharing
                target_projs.append(self.spr_projection(z_tgt))

        loss = spr_loss(pred_latents, target_projs)
        all_params = self.obgd_params + self.spr_params
        grads = torch.autograd.grad(loss, all_params, allow_unused=True,
                                    retain_graph=False)
        return [g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, all_params)]

    def _spr_step(self, spr_grads_all, grads_q):
        """Orthogonalize the SPR gradient against Q, norm-bound, SGD step.

        spr_grads_all: gradients w.r.t. (obgd_params + spr_params).
        grads_q:       the Q gradient w.r.t. obgd_params (from the Q path).
        """
        cfg = self.cfg
        n_obgd = len(self.obgd_params)
        spr_grads_obgd = spr_grads_all[:n_obgd]
        spr_grads_spr = spr_grads_all[n_obgd:]

        # ---- norm-bound the full SPR gradient before orthogonalization ----
        total_norm = 0.0
        for g in spr_grads_obgd + spr_grads_spr:
            total_norm += float((g.flatten() ** 2).sum().item())
        total_norm = (total_norm + 1e-12) ** 0.5
        scale = min(1.0, cfg.spr_grad_clip / total_norm)
        spr_grads_obgd = [g * scale for g in spr_grads_obgd]
        spr_grads_spr = [g * scale for g in spr_grads_spr]

        # ---- orthogonalize the encoder part against Q ----
        spr_orth_obgd = []
        for g_q, g_spr in zip(grads_q, spr_grads_obgd):
            dot = float((g_q.flatten() * g_spr.flatten()).sum().item())
            nq = float((g_q.flatten() ** 2).sum().item()) + 1e-12
            spr_orth_obgd.append(g_spr - (dot / nq) * g_q)

        # ---- SGD step (encoder part orthogonalized, SPR params unmodified) ----
        with torch.no_grad():
            for p, g in zip(self.obgd_params, spr_orth_obgd):
                p.data.add_(g.reshape(p.shape), alpha=-float(cfg.spr_lr))
            for p, g in zip(self.spr_params, spr_grads_spr):
                p.data.add_(g.reshape(p.shape), alpha=-float(cfg.spr_lr))

    def _regenerate(self):
        cfg = self.cfg
        dead = self.utility.dormant_units(self.since_active, cfg.dormant_silence_steps)
        if len(dead) == 0:
            return
        with torch.no_grad():
            self.heads.q.weight[:, dead] = 0.0
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
        return d
