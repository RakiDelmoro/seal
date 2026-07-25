"""SwiftTD optimizer for the linear heads (Tier 1 + Tier 2).

Source: Javed, Sharifnassab & Sutton, "SwiftTD: A Fast and Robust Algorithm for
Temporal Difference Learning", RLC 2024 (arXiv/RLJ_RLC_2024_111). Algorithm 1
implemented verbatim for the per-feature linear setting, generalized to a
multi-output head by running one independent SwiftTD linear learner per output
row over the shared trunk-feature vector.

SwiftTD = True Online TD(λ) + IDBD per-feature step-size optimization +
overshoot bound on the eligibility vector + step-size decay + β-clipping.

Applied ONLY to the GVF bank (linear PREDICTION heads). True Online TD(λ)'s
exact equivalence with the online λ-return holds for linear learners whose
bootstrap target does not depend on the learner's own weights via a max
operator. The GVF heads satisfy this — each bootstraps from its OWN next
prediction under a fixed cumulant. The Q head does NOT (max_a' Q(s',a') is
nonlinear in the weights → the deadly-triad overestimation cascade), so the
Q head is owned by AdaptiveObGD with its κ-bound, NOT by SwiftTD. This split
matches both source papers: Elsayed et al. used AdaptiveObGD for the
control/Q head; Javed et al. validated SwiftTD on prediction, never on
Q-learning control.

  * True Online TD(λ)  — the `δ'·z − z_δ·v_δ` correction that makes the
    online update exactly match the online λ-return at LARGE step sizes.
  * IDBD step-size optimization — per-feature β[i] learned by meta-gradient;
    grows steps of features that reduce error, shrinks irrelevant ones.
    Loss-aware, unlike Adam/RMSProp normalization.
  * Overshoot bound — correction ratio τ = Σ α[i]φ[i]² capped at η; the
    eligibility increment is scaled by min(1, η/τ) so a step never moves the
    prediction past its target.
  * Step-size decay — when the bound fires (τ > η), α[i] ← α[i]·ε^{φ[i]²},
    shrinking the step sizes that caused the overshoot.

SEAL wiring (see agent.py):
  * GVF bank: one SwiftTD learner per general value function (model/gvf.py),
    each a TD(λ) prediction of a discounted future cumulant with its OWN λ.
    All GVFs update every step (off-policy value predictions). Bootstrap for
    GVF k = that GVF's prediction at the next state (0 if terminal). Traces
    reset on done / exploration.
  * The event encoder + Q head are owned by AdaptiveObGD (κ-bound +
    Adam-style normalization), receiving the GVF-shaping gradient through
    the joint loss so the aux tasks keep shaping the trunk.

All SwiftTD math is on an augmented feature vector φ_aug = [trunk_feats (256),
1] (the trailing 1 is the bias feature), so each learner's weight vector is
exactly [W[k], b[k]] (257). Per-learner state (257-vectors): z, h, htemp,
hold, p, zbar, z_delta, β; plus scalars v_old, v_delta.
"""
from __future__ import annotations
import math
import torch
from config import Config


def _safe_exp(beta: torch.Tensor) -> torch.Tensor:
    return torch.exp(beta.clamp(max=20.0))  # guard against overflow


class _SwiftTDRow:
    """One SwiftTD linear learner: v = w·φ over a 257-length augmented vector.

    Owns its weight buffer `w` (257,) which mirrors a row of an nn.Linear
    weight + the corresponding bias. The container syncs w ↔ the parameter
    tensors after each step so the forward pass sees updated weights.
    """

    def __init__(self, in_aug: int, cfg: Config, lam: float):
        self.in_aug = int(in_aug)
        self.theta = float(cfg.swift_theta)
        self.eta = float(cfg.swift_eta)
        self.eps_decay = float(cfg.swift_eps_decay)
        self.alpha_init = float(cfg.swift_alpha_init)
        self.eta_min = float(cfg.swift_eta_min)
        self.lam = float(lam)
        self.gamma = float(cfg.swift_gamma)
        self.ln_eta_min = math.log(self.eta_min)
        self.ln_eta = math.log(self.eta)
        self.ln_eps = math.log(self.eps_decay)
        self._init_state()

    def _init_state(self):
        d = self.in_aug
        # weight buffer (mirrors [W row, b]); container copies params in.
        self.w = torch.zeros(d)
        # eligibility trace + IDBD meta-gradient machinery
        self.z = torch.zeros(d)
        self.h = torch.zeros(d)
        self.htemp = torch.zeros(d)
        self.hold = torch.zeros(d)
        self.p = torch.zeros(d)
        self.zbar = torch.zeros(d)
        self.zdelta = torch.zeros(d)
        self.beta = torch.full((d,), math.log(self.alpha_init))
        # carried scalars (True Online TD(λ))
        self.v_old = 0.0
        self.v_delta = 0.0
        # diagnostics
        self.last_tau = 0.0
        self.last_delta_prime = 0.0
        self.last_v = 0.0

    def load_weight(self, w_row: torch.Tensor, b_row: torch.Tensor):
        with torch.no_grad():
            self.w.copy_(torch.cat([w_row.detach().reshape(-1),
                                    b_row.detach().reshape(1)]))

    def store_weight(self, w_row: torch.Tensor, b_row: torch.Tensor):
        with torch.no_grad():
            w_row.data.copy_(self.w[: w_row.numel()])
            b_row.data.copy_(self.w[w_row.numel():].reshape(()))

    def decay(self):
        """Time-passes the trace without an update (non-taken Q action)."""
        with torch.no_grad():
            self.z.mul_(self.gamma * self.lam)
            self.p.mul_(self.gamma * self.lam)
            self.zbar.mul_(self.gamma * self.lam)

    def reset(self):
        """Off-policy / episode-boundary reset of all trace + meta state."""
        with torch.no_grad():
            self.z.zero_()
            self.h.zero_()
            self.htemp.zero_()
            self.hold.zero_()
            self.p.zero_()
            self.zbar.zero_()
            self.zdelta.zero_()
            self.v_old = 0.0
            self.v_delta = 0.0

    def step(self, phi: torch.Tensor, c: float, v_next: float,
             done: bool, reset: bool):
        """One SwiftTD update (Algorithm 1, verbatim; accumulating traces).

        phi:    [in_aug] detached augmented features for the state updated.
        c:      scalar cumulant for the transition out of that state
                (reward for the Q head; the GVF cumulant for a GVF head).
        v_next: bootstrap target (max_a' Q(s',a') for Q; the GVF's own next
                prediction for a GVF; 0 if terminal).
        done / reset: terminal / off-policy reset flags.
        """
        with torch.no_grad():
            phi = phi.detach()
            ebeta = _safe_exp(self.beta)
            # prediction with current weights (start of step)
            v = float((self.w * phi).sum().item())
            delta_prime = c + self.gamma * v_next * (0.0 if done else 1.0) - self.v_old

            mask_z = self.z.abs() > 0.0
            mask_phi = phi.abs() > 1e-9

            # ---- Algorithm 1, first loop (over trace-active z[i] != 0) ----
            delta_w = torch.where(mask_z,
                                  delta_prime * self.z - self.zdelta * self.v_delta,
                                  torch.zeros_like(self.w))
            self.w.add_(delta_w)
            self.beta.add_(torch.where(mask_z,
                                       (self.theta / ebeta) * (delta_prime - self.v_delta) * self.p,
                                       torch.zeros_like(self.w)))
            self.beta.clamp_(self.ln_eta_min, self.ln_eta)
            new_hold = torch.where(mask_z, self.h, self.hold)
            new_h = torch.where(mask_z, self.htemp, self.h)
            new_htemp = torch.where(mask_z,
                                    self.h + delta_prime * self.zbar - self.zdelta * self.v_delta,
                                    self.htemp)
            self.hold.copy_(new_hold)
            self.h.copy_(new_h)
            self.htemp.copy_(new_htemp)
            self.zdelta.zero_()
            self.z.mul_(self.gamma * self.lam)
            self.p.mul_(self.gamma * self.lam)
            self.zbar.mul_(self.gamma * self.lam)
            self.v_delta = 0.0

            # ---- τ (correction ratio) and T (trace-feature overlap) ----
            tau = float((ebeta * phi * phi * mask_phi).sum().item())
            T = float((self.z * phi * mask_phi).sum().item())

            if tau <= 0.0:
                self.last_tau = tau
                self.last_delta_prime = delta_prime
                self.last_v = v
                self.v_old = v
                if reset:
                    self.reset()
                return v, delta_prime

            # ---- Algorithm 1, second loop (over feature-active φ[i] != 0) ----
            bound = min(1.0, self.eta / tau)
            self.v_delta = float((delta_w * phi * mask_phi).sum().item())
            new_zdelta = torch.where(mask_phi, bound * ebeta * phi, torch.zeros_like(self.w))
            self.z.add_(new_zdelta * (1.0 - T))
            self.p.add_(torch.where(mask_phi, phi * self.h, torch.zeros_like(self.w)))
            self.zbar.add_(torch.where(mask_phi,
                                       new_zdelta * (1.0 - T - phi * self.zbar),
                                       torch.zeros_like(self.w)))
            self.htemp.add_(torch.where(mask_phi,
                                        -(self.hold * phi * (self.z - new_zdelta)
                                          - self.h * new_zdelta * phi),
                                        torch.zeros_like(self.w)))

            # ---- step-size decay (bound fired) ----
            if tau > self.eta:
                self.beta.add_(torch.where(mask_phi, phi * phi * self.ln_eps,
                                           torch.zeros_like(self.w)))
                self.htemp = torch.where(mask_phi, torch.zeros_like(self.htemp), self.htemp)
                self.h = torch.where(mask_phi, torch.zeros_like(self.h), self.h)
                self.zbar = torch.where(mask_phi, torch.zeros_like(self.zbar), self.zbar)

            self.v_old = v
            self.last_tau = tau
            self.last_delta_prime = delta_prime
            self.last_v = v
            if reset:
                self.reset()
            return v, delta_prime


class SwiftTD:
    """Container managing SwiftTD learners for the GVF bank only.

    SwiftTD = True Online TD(λ) + IDBD + overshoot bound + step-size decay. It
    is mathematically EXACT for linear PREDICTION (v = w·φ with a target that
    does not depend on the learner's own weights via a max). The GVF heads
    satisfy this — each GVF bootstraps from its OWN next prediction under a
    fixed cumulant. The Q head does NOT (it bootstraps from max_a' Q(s',a'),
    a nonlinear-in-the-weights operator → deadly-triad overestimation), so
    the Q head is owned by AdaptiveObGD with its κ-bound, NOT by SwiftTD.

    Constructed with the GVF heads (nn.ModuleList of nn.Linear, one per GVF)
    whose weights/biases SwiftTD owns. After sparse-init, call
    `load_from_params()` once to seed the weight buffers; thereafter each
    `step_gvfs` call updates the buffers and writes them back to the params.
    """

    def __init__(self, gvf_heads: torch.nn.ModuleList,
                 cfg: Config, gvf_lams: tuple[float, ...]):
        self.cfg = cfg
        self.gvf_heads = gvf_heads
        self.n_gvfs = int(len(gvf_heads))
        self.trunk_dim = int(gvf_heads[0].in_features)
        self.in_aug = self.trunk_dim + 1  # + bias feature

        self.gvf_rows = [_SwiftTDRow(self.in_aug, cfg, lam=gvf_lams[k])
                         for k in range(self.n_gvfs)]

        self._synced = False
        # diagnostics
        self.last_gvf_tau = 0.0

    # --------------------------------------------------------------- sync
    def load_from_params(self):
        """Seed weight buffers from the (sparse-initialized) GVF params."""
        with torch.no_grad():
            for k, row in enumerate(self.gvf_rows):
                row.load_weight(self.gvf_heads[k].weight[0],
                                self.gvf_heads[k].bias[0])
        self._synced = True

    def _store_gvf_row(self, k: int):
        self.gvf_rows[k].store_weight(self.gvf_heads[k].weight[0],
                                      self.gvf_heads[k].bias[0])

    def _augment(self, head_features: torch.Tensor) -> torch.Tensor:
        """trunk_features [trunk_dim] → augmented [trunk_dim+1] with bias=1."""
        f = head_features.detach().reshape(-1)
        return torch.cat([f, torch.ones(1, dtype=f.dtype, device=f.device)])

    # --------------------------------------------------------------- resets
    def reset_all(self):
        for row in self.gvf_rows:
            row.reset()

    # --------------------------------------------------------------- steps
    def step_gvfs(self, head_features: torch.Tensor, cumulants: torch.Tensor,
                  v_nexts: torch.Tensor, done: bool, reset: bool):
        """SwiftTD TD(λ) update for every GVF (off-policy value prediction).

        cumulants: [n_gvfs] detached cumulants c_t per GVF.
        v_nexts:   [n_gvfs] detached bootstrap (each GVF's own next prediction;
                   0 if terminal).
        """
        if not self._synced:
            self.load_from_params()
        phi = self._augment(head_features)
        c = cumulants.detach().reshape(-1)
        vn = v_nexts.detach().reshape(-1)
        max_tau = 0.0
        for k, row in enumerate(self.gvf_rows):
            row.step(phi, c=float(c[k].item()), v_next=float(vn[k].item()),
                     done=done, reset=reset)
            self._store_gvf_row(k)
            max_tau = max(max_tau, float(row.last_tau))
        self.last_gvf_tau = max_tau

    # --------------------------------------------------------------- state
    def state_dict(self) -> dict:
        def _row_sd(row):
            return dict(w=row.w.clone(), z=row.z.clone(), h=row.h.clone(),
                        htemp=row.htemp.clone(), hold=row.hold.clone(),
                        p=row.p.clone(), zbar=row.zbar.clone(),
                        zdelta=row.zdelta.clone(), beta=row.beta.clone(),
                        v_old=row.v_old, v_delta=row.v_delta)
        return dict(gvf=[_row_sd(r) for r in self.gvf_rows],
                    synced=self._synced)

    def load_state_dict(self, sd: dict):
        def _row_load(row, d):
            row.w.copy_(d["w"]); row.z.copy_(d["z"]); row.h.copy_(d["h"])
            row.htemp.copy_(d["htemp"]); row.hold.copy_(d["hold"])
            row.p.copy_(d["p"]); row.zbar.copy_(d["zbar"])
            row.zdelta.copy_(d["zdelta"]); row.beta.copy_(d["beta"])
            row.v_old = float(d["v_old"]); row.v_delta = float(d["v_delta"])
        for r, d in zip(self.gvf_rows, sd["gvf"]):
            _row_load(r, d)
        self._synced = bool(sd.get("synced", True))
