"""SEAL agent: spiking CNN -> LSNN -> actor/critic, trained by reward-based e-prop.

SEAL = Streaming Event-driven Adaptive Learner. The architecture is a recurrent
network of spiking neurons (LSNN = LIF + ALIF) trained online by adaptive
reward-based e-prop (Bellec et al., Nature Communications 2020, Eq. 5/36/37).
No BPTT, no replay, no frame stacking.

One env step:
  1. encode frame -> input spikes (spiking CNN)
  2. run LSNN `sim_ms_per_step` ms -> spike rate, eligibility traces updated
  3. readout -> actor logits + critic V
  4. sample action from softmax policy
  5. (next step) compute δ = r + γV' - V, learning signal L_j, accumulate tags
  6. e-prop update on Win/Wrec; autograd SGD update on readout
  7. discard the sample

Eligibility traces carry temporal credit forward; the neuron-specific learning
signal L_j (via symmetric feedback B_jk = Wout_kjᵀ) routes output errors to each
neuron; the reward prediction error δ gates the update.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model.spiking_conv import SpikingCNN
from model.lsnn import LSNNCore
from model.readout import LeakyReadout
from model.broadcast import FeedbackWeights
from model.eprop_optimizer import EpropOptimizer
from model.optim import ObGD
from model.utility import UtilityTracker


@dataclass
class StepState:
    """Outputs from one forward pass, held for the TD update."""
    logits: torch.Tensor       # [n_actor]
    value: torch.Tensor        # scalar
    action: int
    log_prob: torch.Tensor     # scalar (for entropy)
    z_rate: torch.Tensor       # LSNN spike rate [n_total] (readout input)
    frame: torch.Tensor = None  # [1,1,84,84] input frame (for the CNN grad path)


class SEALAgent(nn.Module):
    """SEAL: adaptive reward-based e-prop on an LSNN, actor-critic, ALE Pong."""

    def __init__(self, cfg: Config, n_actions: int, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = device

        # ---- front-end + recurrent core ----
        self.cnn = SpikingCNN(cfg.conv_layers, gain=0.15, max_p=0.3,
                              seed=cfg.seed, trainable=cfg.train_cnn)
        self.core = LSNNCore(cfg, n_input_neurons=self.cnn.n_input_neurons)

        # ---- readout (actor + critic) ----
        self.readout = LeakyReadout(cfg.n_lif + cfg.n_alif, n_actions,
                                    n_critic=1, kappa=cfg.kappa,
                                    kappa_critic=cfg.kappa_critic,
                                    logit_cap=cfg.logit_cap)

        # ---- symmetric e-prop feedback weights (B^π = Wout_actor^T, B^V = Wout_critic^T) ----
        # B is NOT a separate parameter; it is read live from the readout's
        # separate actor/critic weights at each learning_signal() call, so it
        # always tracks the current readout. Separate channels (paper Eq. 37)
        # give the critic its own undiluted feedback path into the LSNN.
        self.feedback = FeedbackWeights(
            n_total=cfg.n_lif + cfg.n_alif, n_actor=n_actions, n_critic=1,
            readout=self.readout)

        # ---- optimizers ----
        # Win + Wrec on e-prop with an ObGD-bounded step (Eq. 36 direction,
        # overshooting-bounded size — model/eprop_optimizer.py).
        self.eprop_opt = EpropOptimizer(
            [self.core.Win, self.core.Wrec],
            eta=cfg.eta_rec, gamma=cfg.gamma, lam=cfg.lam_rec,
            kappa=cfg.kappa_rec, grad_clip=cfg.grad_clip)
        # Readout + CNN on autograd ObGD (stream-x): ONE optimizer, per-group
        # lr/κ. Separate actor/critic groups (κ_policy=3, κ_value=2 per the
        # paper); the CNN gets its own group. ObGD maintains per-parameter
        # γλ eligibility traces and multiplies by δ itself, so the losses in
        # learn() must be δ-free (see model/optim.py docstring).
        # Critic group SPLIT: Wout_critic (400-dim, zero-mean grad via
        # LayerNorm(z)) is in ObGD; b_critic (the scalar value bias) is NOT
        # -- it is updated by a separate reward-centering EMA in learn()
        # (Naik et al. 2024). Rationale (verified on the ep2400 checkpoint):
        # with b_critic in the shared ObGD group, ||e||_1 was dominated by
        # Wout_critic's 400 noisy zero-mean traces (~1540), shrinking the
        # step to ~2.6e-6 for both; b_critic needed ~4M steps to track the
        # mean return, dominated V (V~b_critic=-3), starved Wout_critic
        # (L2 0.95->0.12), and the critic collapsed to a constant. Even
        # after splitting groups, the per-step delta driving b_critic via
        # ObGD is dominated by zero-mean noise from Wout_critic@LN(z)
        # (spike rates mean-revert -> negative delta bias) that sinks
        # b_critic faster than terminal rewards lift it (verified: no
        # upward trend across seeds). Reward centering decouples the bias
        # (mean return) from that noise: an EMA of delta averages it out,
        # leaving the true mean-return signal. Wout_critic keeps a moderate
        # kappa so it holds signal without bouncing, and starts small
        # (readout.py init) so V~b_critic until the TD signal teaches it
        # state-dependence.
        self.stream_opt = ObGD(
            [{"params": [self.readout.Wout_actor, self.readout.b_actor],
              "lr": cfg.eta_out, "kappa": cfg.kappa_policy,
              "weight_decay": cfg.wd_policy},
             {"params": [self.readout.Wout_critic],
              "lr": cfg.eta_out, "kappa": cfg.kappa_value,
              "weight_decay": cfg.wd_value,
              "delta_cap": cfg.critic_delta_cap},
             # b_critic is NOT in ObGD: see learn() for its reward-centering update.
             {"params": list(self.cnn.parameters()),
              "lr": cfg.eta_cnn, "kappa": cfg.kappa_cnn,
              "weight_decay": cfg.wd_cnn}],
            gamma=cfg.gamma, lamda=cfg.lam)

        # ---- plasticity ----
        self.utility = UtilityTracker(
            n_total=cfg.n_lif + cfg.n_alif, regen_every=cfg.regen_every,
            dormant_silence_ms=cfg.dormant_silence_ms, regen_frac=cfg.regen_frac,
            win_scale=cfg.win_scale, wrec_scale=cfg.wrec_scale)

        self.global_step = 0
        self.last_td_err = 0.0
        self.last_v = 0.0
        self.last_entropy = 0.0
        self.last_spike_rate_hz = 0.0
        # Reward centering for b_critic (Naik et al. 2024): the scalar value
        # bias tracks the running mean return via an EMA of the TD error.
        # E[delta] -> 0 when V equals the true value; a persistent E[delta]>0
        # means V is too low and pulls b_critic up, E[delta]<0 pushes it down.
        # This is DECOUPLED from the per-step delta that drives Wout_critic
        # via ObGD: that per-step delta is dominated by zero-mean noise from
        # Wout_critic@LayerNorm(z) (spike rates mean-revert), whose negative
        # bias sinks b_critic faster than terminal rewards lift it (verified:
        # under ObGD alone b_critic oscillated around -3 with no upward trend).
        # The EMA averages out that noise, leaving the true mean-return signal.
        self._delta_ema = 0.0
        self._bias_centering_lr = cfg.bias_centering_lr

    # ----------------------------------------------------------- episode ctl
    def reset_episode(self):
        self.core.reset()
        self.readout.reset()
        self.eprop_opt.reset()
        # NOTE: _delta_ema is NOT reset on episode boundary — it is a
        # cross-episode running average of the TD error (the mean-return
        # signal). Resetting it would discard the centering information.

    def warmup_forward(self, obs):
        """Run a forward pass without learning (normalize stats warmup)."""
        with torch.no_grad():
            x = self._to_frame(obs)
            in_spikes = self.cnn(x)
            _ = self.core(in_spikes)
            _ = self.readout(self.core.spike_count / float(self.cfg.sim_ms_per_step))

    def reset_after_warmup(self):
        self.reset_episode()
        self.global_step = 0

    def _to_frame(self, obs) -> torch.Tensor:
        from env.envs import obs_to_chw
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs_to_chw(obs))
        return obs.unsqueeze(0).contiguous().float().to(self.device)  # [1,1,84,84]

    # ----------------------------------------------------------- episode len
    def _current_max_len(self) -> int:
        """From the episode-length schedule, the current max episode length."""
        sched = self.cfg.episode_schedule
        cur = sched[0][1]
        for (start_step, max_len) in sched:
            if self.global_step >= start_step:
                cur = max_len
        return cur

    # ----------------------------------------------------------- act
    def act(self, obs) -> tuple:
        """Forward pass + action sampling. Returns (action, StepState)."""
        x = self._to_frame(obs)
        in_spikes = self.cnn(x)
        z_rate = self.core(in_spikes)          # [n_total], drives eligibility
        logits, value = self.readout(z_rate)
        probs = F.softmax(logits, dim=-1)
        # ε-greedy on top of the softmax policy (stream-x's Atari recipe):
        # exploration is DECOUPLED from policy sharpness, so a sharpening
        # softmax can never kill exploration. The PG update still treats the
        # taken action as on-policy (at ε=0.05 the bias is negligible).
        is_expl = bool(np.random.rand() < self.cfg.explore_eps)
        if is_expl:
            a = int(np.random.randint(self.n_actions))
        else:
            a = int(torch.multinomial(probs, 1).item())  # stochastic policy
        log_prob = torch.log(probs[a] + 1e-12)
        # track spiking activity for plasticity
        self.utility.observe(self.core.last_z, self.cfg.sim_ms_per_step)
        self.last_spike_rate_hz = self.core.spike_rate()
        self.last_entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())
        self.last_v = float(value.item())
        # NOTE: z_rate is detached; the readout graph is rebuilt in learn()
        # so backward never crosses a freed graph from a previous step.
        st = StepState(logits=logits.detach(), value=value.detach(), action=a,
                       log_prob=log_prob.detach(), z_rate=z_rate.detach(),
                       frame=x.detach())
        return a, st

    # ----------------------------------------------------------- learn
    def learn(self, pending: StepState, r: float,
              next_state: StepState = None, done: bool = False):
        """One e-prop update step (reward-based, Eq. 5/36).

        Computes δ = r + γV' − V, the neuron-specific learning signal L_j,
        accumulates the F_γ-filtered eligibility tag, and applies the update
        to Win/Wrec. The readout is updated by autograd on the actor-critic
        loss. (Symmetric e-prop: B_jk = Wout_kjᵀ is a live view, no separate
        update.)
        """
        cfg = self.cfg
        v = pending.value
        v_next = torch.tensor(0.0) if done else next_state.value.detach()
        # ---- optional clip of critic bootstrap + TD error (0 = off) ----
        # OFF by default: v_clip=10 < |true Pong return| = 21 makes the TD
        # targets inconsistent (critic can never fit, rams into the clip), and
        # delta_clip censors the terminal-scoring events — the highest-value
        # samples. ObGD's δ̄ = max(|δ|,1) normalization auto-shrinks the step
        # for large δ, making these clips unnecessary (stream-x runs unclipped).
        v_val = float(v.item())
        v_next_val = float(v_next.item())
        if cfg.v_clip > 0:
            v_val = max(-cfg.v_clip, min(cfg.v_clip, v_val))
            v_next_val = max(-cfg.v_clip, min(cfg.v_clip, v_next_val))
        delta = float(r + cfg.gamma * v_next_val * (0.0 if done else 1.0) - v_val)
        if cfg.delta_clip > 0:
            delta = max(-cfg.delta_clip, min(cfg.delta_clip, delta))
        self.last_td_err = delta

        # ---- learning signal L_j (Eq. 37) ----
        #   L_j = c_V * B^V_j  +  Σ_k B^π_jk (π_k − 1_{a=k})
        # The VALUE term is a CONSTANT (c_V · B^V_j): it only tells neuron j
        # how much it influences the value prediction. The value ERROR itself
        # is carried by the global δ_t in eprop_opt.step(delta) (Eq. 36), NOT
        # by an extra factor inside L_j. The previous code multiplied by
        # (V_{t+1} − V), which (a) is not in the paper and (b) collapses to ~0
        # whenever V is flat — switching off the critic channel entirely.
        probs = F.softmax(pending.logits, dim=-1)
        policy_err = probs.clone()
        policy_err[pending.action] -= 1.0          # (π_k − 1_{a=k})
        critic_err = 1.0   # constant value term; error is gated by δ_t (Eq. 36)
        L_j = self.feedback.learning_signal(policy_err, critic_err, cfg.c_v)

        # ---- accumulate e-prop tags: tag <- γ·tag + L_j · ε̄_ji ----
        elig_win = self.core.eligibility_win()     # [n_total, n_input]
        elig_wrec = self.core.eligibility_wrec()   # [n_total, n_total]
        self.eprop_opt.accumulate(L_j, elig_win, self.core.Win, 0)
        self.eprop_opt.accumulate(L_j, elig_wrec, self.core.Wrec, 1)

        # ---- e-prop update on Win/Wrec ----
        self.eprop_opt.step(delta)

        # ---- readout + CNN update (autograd ObGD, δ-free losses) ----
        # Rebuild the readout forward on the pending step's (z, y_prev) snapshot
        # so the autograd graph is fresh (the act() graph was detached/freed).
        # ObGD multiplies the traced gradient by δ itself, so the losses must
        # NOT contain δ (stream AC form; see model/optim.py):
        #   policy: −log π(a)           -> update += step·δ·∇log π  (PG ascent)
        #   value:  −c_V·V              -> update += step·c_V·δ·∇V (semi-grad TD)
        #   cnn:    −(L_in · p)         -> update += step·δ·∇(L_in·p) (Eq. 36)
        y_fresh = self.readout.forward_from(self.readout._y_prev,
                                            pending.z_rate)
        logits_fresh = y_fresh[:self.n_actions]
        value_fresh = y_fresh[self.n_actions]          # differentiable V (critic)
        probs_fresh = F.softmax(logits_fresh, dim=-1)
        log_prob_fresh = torch.log(probs_fresh[pending.action] + 1e-12)
        sign_d = 1.0 if delta >= 0.0 else -1.0
        policy_loss = -log_prob_fresh
        if cfg.entropy_coef > 0:
            # entropy bonus weighted by sign(δ) (stream-x): encourage
            # exploration when outcomes are worse than expected
            ent = -(probs_fresh * torch.log(probs_fresh + 1e-12)).sum()
            policy_loss = policy_loss - cfg.entropy_coef * sign_d * ent
        value_loss = -cfg.c_v * value_fresh

        if cfg.train_cnn:
            # ---- input-layer learning signal (symmetric e-prop, one more hop) ----
            # L_in = Winᵀ · L_j : input unit i's influence on the loss flows
            # through Win — the same locality approximation the paper makes for
            # L_j (its Fig. 4b: error fed back to the spiking CNN as well).
            # The CNN is feedforward within a frame, so its eligibility is the
            # ordinary local gradient through rates p (E[spikes] = p — the
            # Rao-Blackwellized straight-through at the Bernoulli sampling).
            L_in = self.core.Win.detach().t() @ L_j.detach()      # [n_input]
            p_in = self.cnn.rates(pending.frame)                  # differentiable
            cnn_loss = -(L_in * p_in).sum()
        else:
            cnn_loss = 0.0

        total_loss = policy_loss + value_loss + cnn_loss
        self.stream_opt.zero_grad()
        total_loss.backward()
        # b_critic must NOT receive an autograd gradient (it is updated by the
        # reward-centering EMA below, not ObGD). Zero its grad before step so
        # the value_loss = -c_v*V path (whose d/db_critic = -c_v) cannot reach it.
        if self.readout.b_critic.grad is not None:
            self.readout.b_critic.grad.zero_()
        self.stream_opt.step(delta, reset=done)

        # ---- reward centering: b_critic tracks the mean return ----
        # EMA of delta: _delta_ema -> 0 at the true value. Pull b_critic toward
        # higher V when E[delta]>0 (V too low) and lower when E[delta]<0.
        # Step ~ bias_centering_lr * delta_ema; the (1-gamma) factor matches the
        # relationship between the average reward and the discounted value bias.
        self._delta_ema = (cfg.bias_ema_decay * self._delta_ema
                           + (1.0 - cfg.bias_ema_decay) * delta)
        with torch.no_grad():
            self.readout.b_critic.add_(self._bias_centering_lr * self._delta_ema)

        # ---- plasticity regen ----
        self.global_step += 1
        self.utility.maybe_regen(self.global_step, self.core, self.readout)

        return delta

    # ----------------------------------------------------------- diagnostics
    def b_drift(self) -> float:
        # Symmetric e-prop: B = Woutᵀ (live view), so there is no separate B
        # to drift. Kept as a no-op so logging/CSV columns stay compatible.
        return self.feedback.drift_from_init()

    def tag_norms(self):
        return self.eprop_opt.tag_norms

    def dormant_frac(self) -> float:
        return self.utility.dormant_frac()
