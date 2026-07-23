"""Probe harness: run N frames, print ret20 trajectory every `every` frames.

Sweeps the streaming-RL dials (λ, entropy_coeff, κ) to break the -20.5 plateau.
λ=0.8 gives trace decay 0.792^100≈1e-10 between Pong points → no credit reaches
the actions 100+ frames before a score. Now that z_sum is naturally bounded
(~1500, move A), we have headroom to raise λ for longer credit assignment.
"""
import argparse, numpy as np, torch, time, collections
from config import config_from_preset
from env.envs import make_env, warmup
from model.agent import StreamingActorCritic


def probe(frames, seed, lam, entropy_coeff, kappa, max_z_sum, every=10000,
         total_frames_budget=5_000_000):
    """total_frames_budget controls the epsilon schedule (5% of budget = decay
    duration). frames is how many steps to actually run. This lets a 150k probe
    simulate the early window of a 5M run."""
    cfg = config_from_preset("ALE/Pong-v5", total_frames=total_frames_budget,
                             run_name=f"probe_l{int(lam*100)}_e{int(entropy_coeff*1000)}_k{int(kappa*10)}")
    cfg.lam = lam
    cfg.entropy_coeff = entropy_coeff
    cfg.kappa = kappa
    cfg.max_z_sum = max_z_sum
    torch.manual_seed(seed); np.random.seed(seed)
    env, spec = make_env(cfg.env_id, seed=seed, scale_reward=cfg.scale_reward,
                         ema_alphas=cfg.ema_alphas, ema_lags=cfg.ema_lags)
    agent = StreamingActorCritic(cfg, spec.n_actions)
    warmup(env, agent, n_frames=1000, seed=seed)
    agent.reset_episode()
    obs, _ = env.reset(seed=seed)
    a, pending = agent.act(obs)
    rets = collections.deque(maxlen=20)
    ep_ret = 0.0; t0 = time.time()
    print(f"PROBE lam={lam} ent={entropy_coeff} kappa={kappa} max_z={max_z_sum}")
    for t in range(1, frames + 1):
        no, r, term, trunc, info = env.step(a)
        done = bool(term or trunc); ep_ret += float(r)
        if done:
            agent.learn(pending, float(r), v_next=0.0, done=True)
            rets.append(ep_ret)
            agent.reset_episode(); obs, _ = env.reset(); a, pending = agent.act(obs)
            ep_ret = 0.0
        else:
            a2, np_ = agent.act(no)
            agent.learn(pending, float(r), v_next=agent.bootstrap(np_, done), done=False)
            pending = np_; a = a2
        if t % every == 0:
            r20 = np.mean(rets) if rets else -21.0
            best = max(rets) if rets else -21
            z = agent.opt.last_z_sum; ae = agent.opt.last_step_size
            ent = getattr(agent, 'last_entropy', 0.0)
            rates = agent.encoder.event_rates()
            fps = t / max(1e-6, time.time() - t0)
            rstr = ' '.join(f'{x:.3f}' for x in rates)
            print(f'  t={t:>6} ret20={r20:6.2f} best={best:3.0f} z={z:7.0f} '
                  f'step={ae:.1e} ent={ent:.2f} eps={agent.epsilon:.3f} rates=[{rstr}] fps={fps:.0f}')
    env.close()
    return np.mean(rets) if rets else -21.0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.8)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument("--max-z-sum", type=float, default=10000.0)
    p.add_argument("--every", type=int, default=10000)
    p.add_argument("--total-frames-budget", type=int, default=5_000_000)
    a = p.parse_args()
    probe(a.frames, a.seed, a.lam, a.entropy_coeff, a.kappa, a.max_z_sum, a.every,
          a.total_frames_budget)
