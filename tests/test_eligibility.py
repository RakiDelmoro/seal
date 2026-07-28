"""Stage 1 tests: eligibility traces (Eqs. 13-14, 22-25).

THE make-or-break test: verify that  Σ_t L_j·ε_ji^t  equals the surrogate
gradient  dE/dW_ji  for a feedforward spiking layer. For a feedforward net
the local eligibility trace IS the full gradient (no recurrent route (ii)),
so Eq. 1 is exact — this validates the eligibility-trace math.

We compare against autograd through a straight-through pseudo-derivative
(the same surrogate e-prop uses), not against finite differences of the
discontinuous true system (which would be ill-defined).

Run: pytest tests/test_eligibility.py -v
"""
import torch
import torch.nn.functional as F

from config import Config
from model.neurons import LIFNeurons, ALIFNeurons, pseudo_derivative
from model.eligibility import LIFEligibility, ALIFEligibility


def cfg():
    return Config()


# ------------------------------------------------------------ LIF ε_vec
def test_lif_eligibility_vector_is_lowpass_of_presynaptic_spikes():
    """Eq. 22: ε_vec^t = Σ_{t'<=t} α^{t-t'} z_i^{t'} = α·ε_vec^{t-1} + z_i^t."""
    c = cfg()
    n_post, n_pre, T = 4, 6, 12
    alpha = c.alpha
    elig = LIFEligibility(n_post, n_pre, alpha)
    # random presynaptic spike trains and per-step ψ
    torch.manual_seed(0)
    z_pre_seq = (torch.rand(T, n_pre) > 0.5).float()
    psi_seq = torch.rand(T, n_post) * 0.3

    eps_v_ref = torch.zeros(n_post, n_pre)
    for t in range(T):
        eps = elig.step(z_pre_seq[t], psi_seq[t])
        # reference: ε_vec^t = α·ε_vec^{t-1} + z_i^t
        eps_v_ref = alpha * eps_v_ref + z_pre_seq[t].unsqueeze(0)
        assert torch.allclose(elig.eps_v, eps_v_ref, atol=1e-6), \
            f"ε_vec mismatch at t={t}"
        # trace = ψ · ε_vec  (Eq. 23)
        assert torch.allclose(eps, psi_seq[t].unsqueeze(1) * eps_v_ref, atol=1e-6)


# ------------------------------------------------------------ LIF Eq. 1
def test_lif_eprop_gradient_matches_autograd():
    """Eq. 1 for a feedforward LIF layer: Σ_t c·ε_ji^t == surrogate dE/dW.

    Loss E = Σ_t c_j·z_j^t (linear, so ∂E/∂z_j^t = c_j, the learning signal).
    The e-prop gradient is Σ_t c_j·ε_ji^t. We compare to autograd through a
    straight-through pseudo-derivative (same ψ as e-prop).
    """
    c = cfg()
    torch.manual_seed(1)
    n_post, n_pre, T = 5, 7, 15
    alpha = c.alpha
    v_th = c.v_threshold
    gamma_pd = c.gamma_pd

    W = torch.zeros(n_post, n_pre, requires_grad=True)
    # fixed input spike train
    x = (torch.rand(T, n_pre) > 0.5).float()
    # fixed learning signal c_j = ∂E/∂z_j (one per postsynaptic neuron)
    cj = torch.randn(n_post)

    # ---- e-prop forward (eligibility accumulation) ----
    lif = LIFNeurons(n_post, alpha, v_th, gamma_pd, refractory_steps=1)
    elig = LIFEligibility(n_post, n_pre, alpha)
    g_eprop = torch.zeros(n_post, n_pre)
    z_pre = x[0]
    for t in range(1, T):
        i_syn = F.linear(z_pre.unsqueeze(0), W).squeeze(0)  # W @ z_pre
        z_post, psi = lif.step(i_syn)
        eps = elig.step(z_pre, psi)               # ε_ji^t [n_post, n_pre]
        g_eprop += cj.unsqueeze(1) * eps          # Σ_t c_j · ε_ji^t
        z_pre = x[t]

    # ---- autograd reference (straight-through ψ) ----
    W_ref = W.detach().clone().requires_grad_()
    v = torch.zeros(n_post)
    ref = torch.zeros(n_post)
    z_pre = x[0]
    for t in range(1, T):
        i_syn = F.linear(z_pre.unsqueeze(0), W_ref).squeeze(0)
        v_new = alpha * v + i_syn
        # spike (discontinuous forward)
        z = (v_new >= v_th).float()
        # straight-through: forward = z, backward dz/dv = ψ
        psi = (gamma_pd / v_th) * F.relu(1.0 - (v_new - v_th).abs() / v_th)
        z_st = z.detach() + psi * (v_new - v_new.detach())
        # subtractive reset on spike (for v dynamics; gradient flows through)
        v_next = torch.where(z > 0.5, v_new - v_th, v_new)
        ref = ref + cj * z_st                     # E = Σ c·z
        v = v_next.detach()                       # match e-prop's hard reset
        z_pre = x[t]
    g_auto = torch.autograd.grad(ref.sum(), W_ref, retain_graph=False)[0]

    assert torch.allclose(g_eprop, g_auto, atol=1e-5), \
        f"LIF e-prop gradient mismatch:\n{g_eprop}\nvs autograd:\n{g_auto}"


# ------------------------------------------------------------ ALIF ε_vec
def test_alif_eligibility_vector_matches_reference():
    """Eq. 24: ε_a^{t+1} = ψ·z_i + (ρ - ψ·β)·ε_a^t; ε_v as LIF."""
    c = cfg()
    n_post, n_pre, T = 3, 5, 10
    alpha, rho, beta = c.alpha, c.rho, c.beta
    elig = ALIFEligibility(n_post, n_pre, alpha, rho, beta, approx=False)
    torch.manual_seed(2)
    z_seq = (torch.rand(T, n_pre) > 0.5).float()
    psi_seq = torch.rand(T, n_post) * 0.3

    eps_v_ref = torch.zeros(n_post, n_pre)
    eps_a_ref = torch.zeros(n_post, n_pre)
    for t in range(T):
        eps = elig.step(z_seq[t], psi_seq[t])
        zpre = z_seq[t].unsqueeze(0)
        psi = psi_seq[t].unsqueeze(1)
        eps_v_ref = alpha * eps_v_ref + zpre
        eps_a_ref = psi * zpre + (rho - psi * beta) * eps_a_ref
        assert torch.allclose(elig.eps_v, eps_v_ref, atol=1e-6)
        assert torch.allclose(elig.eps_a, eps_a_ref, atol=1e-6)
        # Eq. 25: ε = ψ·(ε_v - β·ε_a)
        assert torch.allclose(eps, psi * (eps_v_ref - beta * eps_a_ref), atol=1e-6)


def test_alif_eligibility_approx_drops_psi_beta():
    """approx=True uses Eq. 26: decay = ρ (drops the ψ·β term)."""
    c = cfg()
    n_post, n_pre, T = 3, 5, 8
    alpha, rho, beta = c.alpha, c.rho, c.beta
    elig = ALIFEligibility(n_post, n_pre, alpha, rho, beta, approx=True)
    torch.manual_seed(3)
    z_seq = (torch.rand(T, n_pre) > 0.5).float()
    psi_seq = torch.rand(T, n_post) * 0.3
    eps_a_ref = torch.zeros(n_post, n_pre)
    for t in range(T):
        elig.step(z_seq[t], psi_seq[t])
        zpre = z_seq[t].unsqueeze(0)
        psi = psi_seq[t].unsqueeze(1)
        eps_a_ref = psi * zpre + rho * eps_a_ref   # Eq. 26: decay = ρ
        assert torch.allclose(elig.eps_a, eps_a_ref, atol=1e-6)


# ------------------------------------------------------------ ALIF Eq. 1
def test_alif_eprop_gradient_matches_autograd():
    """Eq. 1 for a feedforward ALIF layer: Σ_t c·ε_ji^t == surrogate dE/dW."""
    c = cfg()
    torch.manual_seed(4)
    n_post, n_pre, T = 4, 6, 12
    alpha, rho, beta = c.alpha, c.rho, c.beta
    v_th, gamma_pd = c.v_threshold, c.gamma_pd

    W = torch.zeros(n_post, n_pre)
    x = (torch.rand(T, n_pre) > 0.5).float()
    cj = torch.randn(n_post)

    # ---- e-prop forward ----
    alif = ALIFNeurons(n_post, alpha, rho, v_th, beta, gamma_pd, refractory_steps=1)
    elig = ALIFEligibility(n_post, n_pre, alpha, rho, beta, approx=False)
    g_eprop = torch.zeros(n_post, n_pre)
    z_pre = x[0]
    for t in range(1, T):
        i_syn = F.linear(z_pre.unsqueeze(0), W).squeeze(0)
        z_post, psi = alif.step(i_syn)
        eps = elig.step(z_pre, psi)
        g_eprop += cj.unsqueeze(1) * eps
        z_pre = x[t]

    # ---- autograd reference ----
    W_ref = W.detach().clone().requires_grad_()
    v = torch.zeros(n_post); a = torch.zeros(n_post)
    ref = torch.zeros(n_post)
    z_pre = x[0]
    for t in range(1, T):
        i_syn = F.linear(z_pre.unsqueeze(0), W_ref).squeeze(0)
        v_new = alpha * v + i_syn
        thr = v_th + beta * a
        z = (v_new >= thr).float()
        psi = (gamma_pd / v_th) * F.relu(1.0 - (v_new - thr).abs() / v_th)
        z_st = z.detach() + psi * (v_new - v_new.detach())
        v_next = torch.where(z > 0.5, v_new - thr, v_new)
        a_next = rho * a + z
        ref = ref + cj * z_st
        v = v_next.detach(); a = a_next.detach()
        z_pre = x[t]
    g_auto = torch.autograd.grad(ref.sum(), W_ref)[0]

    assert torch.allclose(g_eprop, g_auto, atol=1e-5), \
        f"ALIF e-prop gradient mismatch:\n{g_eprop}\nvs autograd:\n{g_auto}"


# ------------------------------------------------------------ reset
def test_eligibility_reset():
    """reset() zeroes ε_vec and trace."""
    from model.eligibility import LIFEligibility
    c = cfg()
    elig = LIFEligibility(3, 4, c.alpha)
    elig.step(torch.ones(4), torch.ones(3) * 0.3)
    assert elig.eps_v.abs().sum() > 0
    elig.reset()
    assert elig.eps_v.abs().sum() == 0 and elig.trace.abs().sum() == 0
