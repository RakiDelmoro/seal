"""Stage 0 tests: LIF / ALIF spiking neuron dynamics (Eqs. 6-10).

Gates the neuron primitives before eligibility traces are built on top.
Run: pytest tests/test_neurons.py -v
"""
import math
import torch

from config import Config
from model.neurons import LIFNeurons, ALIFNeurons, pseudo_derivative


def cfg():
    c = Config()
    c.n_lif = 50
    c.n_alif = 50
    return c


# ----------------------------------------------------------------- LIF
def test_lif_membrane_decay_matches_alpha():
    """v_{t+1} = α·v_t with no input -> geometric decay with factor α."""
    c = cfg()
    alpha = c.alpha
    lif = LIFNeurons(n=1, alpha=alpha, v_th=c.v_threshold,
                     gamma_pd=c.gamma_pd, refractory_steps=c.refractory_steps)
    lif.v[0] = 1.0
    for _ in range(10):
        z, _ = lif.step(torch.zeros(1))
        assert z.item() == 0.0  # below threshold, no spike
    # after 10 steps v should be alpha^10
    assert abs(lif.v[0].item() - alpha ** 10) < 1e-6, \
        f"decay {lif.v[0].item()} != alpha^10={alpha**10}"


def test_lif_spike_and_reset():
    """v reaching threshold -> z=1, membrane reduced by v_th, refractory kicks in."""
    c = cfg()
    lif = LIFNeurons(n=1, alpha=c.alpha, v_th=c.v_threshold,
                     gamma_pd=c.gamma_pd, refractory_steps=2)
    # inject input so v_new = α·0 + i_syn lands just over threshold
    i_syn = c.v_threshold + 0.2
    z, psi = lif.step(torch.tensor([i_syn]))
    assert z.item() == 1.0, "should spike"
    # after subtractive reset: v = v_new - v_th = i_syn - v_th
    expected = i_syn - c.v_threshold
    assert abs(lif.v[0].item() - expected) < 1e-6
    # refractory active: even huge input cannot fire next step
    z2, _ = lif.step(torch.tensor([100.0]))
    assert z2.item() == 0.0, "refractory should block firing"
    # v did update though (no reset since no spike)
    assert lif.v[0].item() > 0


def test_lif_pseudo_derivative_window():
    """ψ is nonzero only within v_th of the threshold, zeroed in refractory."""
    c = cfg()
    lif = LIFNeurons(n=3, alpha=c.alpha, v_th=c.v_threshold,
                     gamma_pd=c.gamma_pd, refractory_steps=1)
    # inject inputs so v_new (= α·0 + i_syn) lands at known offsets from v_th.
    # neuron 0: AT threshold (max ψ), 1: 2*v_th away (ψ=0), 2: 0.4 above (partial)
    i_syn = torch.tensor([c.v_threshold, 3 * c.v_threshold, c.v_threshold + 0.4])
    z, psi = lif.step(i_syn)
    expected_max = c.gamma_pd / c.v_threshold
    assert abs(psi[0].item() - expected_max) < 1e-6, "ψ at threshold = γ_pd/v_th"
    assert psi[1].item() == 0.0, "ψ zero far from threshold"
    assert 0 < psi[2].item() < expected_max, "ψ decays linearly"
    # neuron 0 spiked -> refractory next step -> ψ forced to 0
    _, psi_ref = lif.step(torch.zeros(3))
    assert psi_ref[0].item() == 0.0, "ψ=0 during refractory"


# ----------------------------------------------------------------- ALIF
def test_alif_adaptation_grows_then_decays():
    """Spiking increases a by 1 each spike; with no spikes a decays as ρ^k."""
    c = cfg()
    rho = c.rho
    alif = ALIFNeurons(n=1, alpha=c.alpha, rho=rho, v_th=c.v_threshold,
                       beta=c.beta, gamma_pd=c.gamma_pd,
                       refractory_steps=c.refractory_steps)
    # force a spike by injecting large current
    alif.v[0] = c.v_threshold + 0.5
    z, _ = alif.step(torch.zeros(1))
    assert z.item() == 1.0
    assert abs(alif.a[0].item() - 1.0) < 1e-6, "a += z (should be 1)"
    # now no input for many steps; a should decay as ρ^k
    a0 = alif.a[0].item()
    for _ in range(20):
        alif.step(torch.zeros(1))
    assert abs(alif.a[0].item() - a0 * (rho ** 20)) < 1e-5, \
        f"adaptation decay {alif.a[0].item()} != {a0 * rho**20}"


def test_alif_threshold_raises_with_adaptation():
    """A_t = v_th + β·a_t: after a spike, threshold is higher than baseline."""
    c = cfg()
    alif = ALIFNeurons(n=1, alpha=c.alpha, rho=c.rho, v_th=c.v_threshold,
                       beta=c.beta, gamma_pd=c.gamma_pd,
                       refractory_steps=c.refractory_steps)
    thr0 = alif.threshold().item()
    assert abs(thr0 - c.v_threshold) < 1e-6, "baseline threshold = v_th"
    alif.v[0] = c.v_threshold + 1.0
    alif.step(torch.zeros(1))  # spike
    thr1 = alif.threshold().item()
    assert thr1 > thr0, "threshold should rise after spike"
    assert abs(thr1 - (c.v_threshold + c.beta * 1.0)) < 1e-6, \
        f"raised threshold {thr1} != v_th + β"


def test_alif_jacobian_diagonal_matches_eq24():
    """∂h/∂h_prev diagonal = [α, ρ - ψ·β], off-diag ∂a/∂v = ψ (Eq. 24)."""
    c = cfg()
    alif = ALIFNeurons(n=2, alpha=c.alpha, rho=c.rho, v_th=c.v_threshold,
                       beta=c.beta, gamma_pd=c.gamma_pd,
                       refractory_steps=c.refractory_steps)
    # set a known psi
    psi = torch.tensor([0.1, 0.2])
    J = alif.dh_dh_prev(psi)
    assert abs(J[0, 0, 0].item() - c.alpha) < 1e-6, "∂v/∂v = α"
    assert abs(J[0, 1, 1].item() - (c.rho - 0.1 * c.beta)) < 1e-6, "∂a/∂a = ρ-ψβ"
    assert abs(J[0, 1, 0].item() - 0.1) < 1e-6, "∂a/∂v = ψ"
    assert J[0, 0, 1].item() == 0.0, "∂v/∂a = 0"


def test_pseudo_derivative_function_exact():
    """ψ(v) = (γ_pd/v_th)·max(0, 1-|v-A|/v_th)."""
    c = cfg()
    v = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    thr = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    psi = pseudo_derivative(v, thr, gamma_pd=0.3, v_th=1.0)
    # at v=thr: 0.3; at |v-thr|=0.5: 0.15; at |v-thr|=1: 0
    expected = torch.tensor([0.0, 0.15, 0.3, 0.15, 0.0])
    assert torch.allclose(psi, expected, atol=1e-6), f"{psi} != {expected}"


def test_reset_clears_state():
    """reset() zeroes v, a, ref for both neuron types."""
    c = cfg()
    lif = LIFNeurons(n=2, alpha=c.alpha, v_th=c.v_threshold,
                     gamma_pd=c.gamma_pd, refractory_steps=2)
    lif.v[0] = 5.0; lif.ref[0] = 3
    lif.reset()
    assert lif.v.abs().sum() == 0 and lif.ref.sum() == 0

    alif = ALIFNeurons(n=2, alpha=c.alpha, rho=c.rho, v_th=c.v_threshold,
                       beta=c.beta, gamma_pd=c.gamma_pd,
                       refractory_steps=2)
    alif.v[0] = 5.0; alif.a[0] = 2.0; alif.ref[0] = 1
    alif.reset()
    assert alif.v.abs().sum() == 0 and alif.a.abs().sum() == 0
    assert alif.ref.sum() == 0
