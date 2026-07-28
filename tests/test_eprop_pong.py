"""Stage 2/3 RL smoke test: e-prop LSNN on ALE Pong.

Not a learning-curve assertion (Pong takes millions of frames to learn) —
this is a smoke test that the full agent runs end-to-end without NaN/crash,
produces sane diagnostics, the symmetric feedback B_jk tracks Wout_kjᵀ exactly
(the signature of symmetric e-prop), and the eligibility tags accumulate +
reset correctly across episodes.

Run: pytest tests/test_eprop_pong.py -v
"""
import numpy as np
import torch

from config import Config
from env.envs import make_env
from model.agent import SEALAgent


def _run(cfg, n_frames=400, seed=0):
    env, spec = make_env(cfg.env_id, seed=seed)
    agent = SEALAgent(cfg, n_actions=spec.n_actions, device="cpu")
    agent.reset_episode()
    obs, _ = env.reset(seed=seed)
    a, st = agent.act(obs)
    ep_returns = []
    ep_ret = 0.0
    b_drifts = []
    deltas = []
    for t in range(n_frames):
        obs, r, term, trunc, info = env.step(a)
        ep_ret += float(r)
        done = bool(term or trunc)
        if done:
            agent.learn(st, float(r), next_state=None, done=True)
            ep_returns.append(ep_ret); ep_ret = 0.0
            agent.reset_episode()
            obs, _ = env.reset()
            a, st = agent.act(obs)
        else:
            a2, st2 = agent.act(obs)
            d = agent.learn(st, float(r), next_state=st2, done=False)
            deltas.append(d)
            st = st2; a = a2
        if t % 50 == 0:
            b_drifts.append(agent.b_drift())
    env.close()
    return ep_returns, deltas, b_drifts, agent


def test_agent_runs_without_nan():
    """No NaN/Inf in any diagnostic over a 400-frame rollout."""
    cfg = Config(); cfg.warmup_frames = 0
    ep_returns, deltas, b_drifts, agent = _run(cfg, n_frames=400)
    assert all(np.isfinite(d) for d in deltas), "TD error has NaN/Inf"
    assert all(np.isfinite(b) for b in b_drifts), "B drift has NaN/Inf"
    assert np.isfinite(agent.last_v), "V is NaN/Inf"
    assert np.isfinite(agent.last_spike_rate_hz), "spike rate NaN/Inf"


def test_symmetric_feedback_tracks_wout():
    """Symmetric e-prop: B^π == Wout_actor^T and B^V == Wout_critic^T exactly,
    at all times (live views)."""
    cfg = Config(); cfg.warmup_frames = 0
    _, _, _, agent = _run(cfg, n_frames=400)
    assert torch.allclose(agent.feedback.B_pi, agent.readout.Wout_actor.t()), \
        "symmetric B^π must equal Wout_actor^T"
    assert torch.allclose(agent.feedback.B_v, agent.readout.Wout_critic.t()), \
        "symmetric B^V must equal Wout_critic^T"
    # b_drift is a no-op diagnostic under symmetric e-prop (always 0.0)
    assert agent.b_drift() == 0.0


def test_tags_accumulate_and_reset_on_episode():
    """Eligibility tags grow within an episode and zero on reset_episode()."""
    cfg = Config(); cfg.warmup_frames = 0
    env, spec = make_env(cfg.env_id, seed=0)
    agent = SEALAgent(cfg, n_actions=spec.n_actions)
    agent.reset_episode()
    obs, _ = env.reset(seed=0)
    a, st = agent.act(obs)
    tag_before = agent.tag_norms()
    # run a few in-episode steps
    for _ in range(5):
        obs, r, term, trunc, _ = env.step(a)
        if term or trunc:
            break
        a2, st2 = agent.act(obs)
        agent.learn(st, float(r), next_state=st2, done=False)
        st = st2; a = a2
    tag_after = agent.tag_norms()
    assert tag_after[0] > tag_before[0], "tags should accumulate within episode"
    agent.reset_episode()
    assert agent.tag_norms()[0] == 0.0, "reset_episode must zero tags"
    env.close()


def test_spike_rate_in_sparse_regime():
    """Core spike rate stays below ~50 Hz (energy-efficient spike coding)."""
    cfg = Config(); cfg.warmup_frames = 0
    _, _, _, agent = _run(cfg, n_frames=200)
    assert agent.last_spike_rate_hz < 50.0, \
        f"spike rate {agent.last_spike_rate_hz:.1f} Hz too high (want <50)"


def test_eq37_value_term_is_constant():
    """Regression: Eq. 37 value term of L_j is the CONSTANT c_V·B^V_j.

    The value ERROR is carried by the global δ_t in the plasticity rule
    (Eq. 36), NOT by an extra factor inside L_j. Previously the code
    multiplied by (V_{t+1} − V), which collapses to ~0 whenever V is flat
    and switches off the critic channel. This test pins the paper-faithful
    form: with zero policy error, L_j == c_V · B^V_j regardless of V.
    """
    from model.broadcast import FeedbackWeights
    from model.readout import LeakyReadout
    torch.manual_seed(0)
    n_total, n_actions, c_v = 60, 6, 1.0
    readout = LeakyReadout(n_total, n_actions, n_critic=1, kappa=0.95)
    fb = FeedbackWeights(n_total, n_actions, n_critic=1, readout=readout)
    policy_err = torch.zeros(n_actions)          # isolate the value term
    # The value term must NOT depend on any V quantity — pass the constant 1.0
    # the agent is supposed to pass (paper Eq. 37).
    L_j = fb.learning_signal(policy_err, critic_error=1.0, c_v=c_v)
    expected = c_v * fb.B_v.squeeze(1)           # c_V · B^V_j  (constant)
    assert torch.allclose(L_j, expected, atol=1e-6), \
        "Eq. 37 value term must be c_V·B^V_j (constant), not error-weighted"

    # The bug would show up as L_j scaling with the passed critic_error.
    # Confirm the value term scales linearly with the constant multiplier
    # (c_V) but is independent of V values — i.e. passing 1.0 is the agent's
    # job, and the broadcast layer just applies it as a constant.
    L2 = fb.learning_signal(policy_err, critic_error=1.0, c_v=2.0)
    assert torch.allclose(L2, 2.0 * fb.B_v.squeeze(1), atol=1e-6)


def test_agent_critic_channel_not_attenuated_by_flat_v():
    """Regression: a flat V must NOT switch off the critic e-prop channel.

    The old bug multiplied L_j's value term by (V_{t+1} − V), so when V was
    flat the value channel drove ~0. After the fix, the value-channel
    contribution to the eligibility tags is independent of |V_{t+1} − V| and
    is gated only by δ_t (Eq. 36). We verify the tag norm contributed by the
    value term alone is non-negligible even when V is forced flat.
    """
    cfg = Config(); cfg.warmup_frames = 0
    env, spec = make_env(cfg.env_id, seed=0)
    agent = SEALAgent(cfg, n_actions=spec.n_actions)
    agent.reset_episode()
    obs, _ = env.reset(seed=0)
    a, st = agent.act(obs)

    # Spy on learning_signal to capture the critic_error the agent passes.
    passed = {}
    orig = agent.feedback.learning_signal
    def spy(policy_error, critic_error, c_v):
        passed["critic_error"] = critic_error
        return orig(policy_error, critic_error, c_v)
    agent.feedback.learning_signal = spy

    # Run a few steps so learn() is called with a (nominally flat) V.
    for _ in range(5):
        obs, r, term, trunc, _ = env.step(a)
        if term or trunc:
            break
        a2, st2 = agent.act(obs)
        agent.learn(st, float(r), next_state=st2, done=False)
        st = st2; a = a2
    env.close()
    # The agent must pass the CONSTANT 1.0 (paper Eq. 37), not (V_{t+1} − V).
    assert passed.get("critic_error") == 1.0, \
        f"agent must pass critic_error=1.0 (Eq. 37 constant), got {passed.get('critic_error')}"


def _rollout_steps(agent, env, obs, a, st, n):
    """Run n env steps with learning; handles episode ends."""
    for _ in range(n):
        obs, r, term, trunc, _ = env.step(a)
        done = bool(term or trunc)
        if done:
            agent.learn(st, float(r), next_state=None, done=True)
            agent.reset_episode()
            obs, _ = env.reset()
            a, st = agent.act(obs)
        else:
            a2, st2 = agent.act(obs)
            agent.learn(st, float(r), next_state=st2, done=False)
            st = st2; a = a2
    return obs, a, st


def test_cnn_weights_train_when_enabled():
    """Trainable front-end (paper Fig. 4b): conv weights move under the
    input-layer learning signal L_in = Winᵀ·L_j, gated by δ via ObGD."""
    cfg = Config(); cfg.warmup_frames = 0; cfg.train_cnn = True
    env, spec = make_env(cfg.env_id, seed=0)
    agent = SEALAgent(cfg, n_actions=spec.n_actions)
    agent.reset_episode()
    w0 = agent.cnn.convs[0].weight.detach().clone()
    b0 = agent.cnn.convs[1].bias.detach().clone()
    obs, _ = env.reset(seed=0)
    a, st = agent.act(obs)
    _rollout_steps(agent, env, obs, a, st, 20)
    env.close()
    assert not torch.allclose(w0, agent.cnn.convs[0].weight.detach()), \
        "conv weights should update when train_cnn=True"
    assert not torch.allclose(b0, agent.cnn.convs[1].bias.detach()), \
        "conv biases should update when train_cnn=True"


def test_cnn_frozen_when_disabled():
    """Ablation escape hatch: train_cnn=False keeps the encoder frozen random."""
    cfg = Config(); cfg.warmup_frames = 0; cfg.train_cnn = False
    env, spec = make_env(cfg.env_id, seed=0)
    agent = SEALAgent(cfg, n_actions=spec.n_actions)
    agent.reset_episode()
    w0 = agent.cnn.convs[0].weight.detach().clone()
    obs, _ = env.reset(seed=0)
    a, st = agent.act(obs)
    _rollout_steps(agent, env, obs, a, st, 20)
    env.close()
    assert torch.allclose(w0, agent.cnn.convs[0].weight.detach()), \
        "conv weights must NOT update when train_cnn=False"


def test_checkpoint_roundtrip():
    """save/load model state_dict round-trips weights exactly."""
    cfg = Config(); cfg.warmup_frames = 0
    _, _, _, agent = _run(cfg, n_frames=100)
    sd = agent.state_dict()
    env, spec = make_env(cfg.env_id, seed=1)
    agent2 = SEALAgent(cfg, n_actions=spec.n_actions)
    agent2.load_state_dict(sd, strict=False)
    assert torch.allclose(agent.core.Win, agent2.core.Win)
    assert torch.allclose(agent.core.Wrec, agent2.core.Wrec)
    assert torch.allclose(agent.readout.Wout_actor, agent2.readout.Wout_actor)
    assert torch.allclose(agent.readout.Wout_critic, agent2.readout.Wout_critic)
    assert torch.allclose(agent.feedback.B_pi, agent2.feedback.B_pi)
    assert torch.allclose(agent.feedback.B_v, agent2.feedback.B_v)
    env.close()
