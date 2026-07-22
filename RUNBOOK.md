# SEAL Pong — Runbook

How to run, watch, checkpoint, and resume the SEAL agent learning ALE Pong.

## Quick start

### Start fresh (500k run, with GUI, checkpointing every 50k)
```bash
cd /workspaces/seal
python -m seal.watch --frames 500000 --seed 0 --fps 60
```
- Opens a Pygame window showing the live Pong game + a metrics side panel.
- Saves `results/seal-pong_latest.pt` every 50k frames.
- Saves `results/seal-pong_best.pt` whenever the running return (ret20) improves.
- If it crashes or you Ctrl-C, progress is preserved.
- Prints one diagnostic line per episode to the terminal.

### Resume after a crash / Ctrl-C (continue from where it stopped)
```bash
python -m seal.watch --frames 500000 --seed 0 --fps 60 \
  --resume results/seal-pong_latest.pt
```
Prints `[ckpt] resumed from ... step=48000 episodes=... corrVr=...` and continues
from there — weights, traces, normalization, and corrVr history all restored.

### Load the BEST checkpoint (highest return so far)
```bash
python -m seal.watch --frames 500000 --seed 0 --fps 60 \
  --resume results/seal-pong_best.pt
```

### Headless (faster, no window) + more frequent checkpoints
```bash
python -m seal.watch --frames 500000 --seed 0 --no-render --ckpt-every 25000
```

## Checkpointing — what is saved

Each `seal-pong_*.pt` checkpoint (~3.8 MB) contains:
- **Model weights** (encoder + GRU + heads)
- **Optimizer eligibility traces** (so credit assignment continues, not reset to zero)
- **GRU hidden state** + per-unit silence counters
- **Welford normalization** mean/var/count (so observation scaling is continuous)
- **Episode history** (returns + V-at-episode) so corrVr is continuous across resume
- **Config snapshot** (λ, α, κ, seed, env_id) so you can't accidentally resume with mismatched hyperparameters

| save type | when | file |
|---|---|---|
| periodic | every `--ckpt-every` frames (default 50k) | `results/seal-pong_latest.pt` |
| best | when running return (ret20) improves | `results/seal-pong_best.pt` |
| final | on normal completion OR Ctrl-C | `results/seal-pong_latest.pt` |

## Reading the live log

### Pygame side panel
```
--- IS IT LEARNING? ---
corrVr  +0.04   <- want rising     (green if >0.15, red if <0.05)
return  -19.20  (last20 ep)
--- health ---
|delta| 0.13   V -7.20   entropy 1.78
act_med 0.30   |h| 0.33   z_sum 7400   a_eff 1.5e-09
```

### Per-episode terminal line
```
[EP   12] frame=  2367 ret=-21.00 ret20=-19.83 corrVr=+0.754 |d|=3.24 V=-4.24 ent=1.783 act=0.28 |h|=0.28 z=9469 a_eff=5.6e-09  V-tracking + policy-holding (promising)
```
| field | meaning |
|---|---|
| `EP N` | episode number |
| `frame` | env step reached |
| `ret` | this episode's return (Pong plays to ±21) |
| `ret20` | running mean of last 20 episodes — **the policy trend** |
| `corrVr` | **the key signal**: V-return correlation (want rising + stable > 0.15) |
| `\|d\|` | TD error magnitude |
| `V` | value estimate |
| `ent` | policy entropy (1.79 = pure random; lower = committing) |
| `act` | trunk activation magnitude (want ~0.3, not 0.1) |
| `\|h\|` | GRU hidden magnitude (want < 0.9, not saturated) |
| `z`, `a_eff` | trace sum and effective step size |
| flag | honest read: `V-tracking` / `corrVr flat` / `promising` |

## The decision gate (500k run)

Watch **corrVr** and **return** together — they measure different things:
- **corrVr > 0.15**: the *value function* is learning to predict returns. Leading indicator.
- **return trend**: is the *policy* actually scoring better? May stay flat at 500k even if healthy (the paper needs millions of frames).

| outcome | meaning | next step |
|---|---|---|
| corrVr climbs > ~0.2 + return stops worsening | agent is starting to learn | green light for 10M run |
| corrVr stays ~0.04 for hundreds of episodes | V isn't tracking reward; more frames won't fix | **stop**, go back to diagnosis |
| return worsening over hundreds of episodes | policy is degrading | **stop**, diagnosis |

You can **Ctrl-C anytime** — the CSV + checkpoint save. Kill it the moment corrVr
is obviously stuck flat for a long stretch; no need to wait the full run.

## All CLI flags
```
--frames N         total env frames to run (default 500000)
--seed N           random seed (default 0)
--lam L            trace decay λ (default 0.8 = paper value; spec was 0.95)
--alpha A          nominal learning rate (default 1.0; NOTE: cancels in ObGD,
                   the real dial is λ -- see optimizers.py docstring)
--kappa K          overshoot bound (default 2.0; do NOT tune for speed)
--fps N            cap display fps (0 = as fast as possible; default 0)
--no-render        headless: no Pygame window, ~2x faster
--log-every N      CSV row cadence in frames (default 5000)
--ckpt-every N     checkpoint cadence in frames (default 50000)
--resume PATH      resume from a seal-pong_*.pt checkpoint
```

## Output files
```
results/
  seal-pong_latest.pt      latest periodic checkpoint (auto-saved)
  seal-pong_best.pt        best-return checkpoint (auto-saved)
  watch_l80_s0.csv         per-5k-frame metrics CSV
  diag/                    50k-probe results (stability/learning diagnosis)
    l80_s0.csv ...         per-λ, per-seed probe logs
    h_mean.png             GRU saturation check
    headline_aux_vs_td.png aux-vs-TD (suspect #1 confirmation)
    probe.log              full probe console log
    archive/               pre-fix probe results (provenance)
```

## Time estimates
| mode | speed | 500k takes |
|---|---|---|
| `--fps 60` (watchable GUI) | ~30-50 fps | ~3-4 hours |
| no fps cap (fast GUI) | ~40 fps | ~3 hours |
| `--no-render` (headless) | ~90 fps | ~1.5 hours |

## The hyperparameters we landed on (and why)
- **λ = 0.8** (paper value). Spec was 0.95, which freezes the agent because
  ObGD's effective step = 1/(κ·δ̄·‖z‖₁) and ‖z‖₁ grows with λ. λ=0.8 gives
  ~3.5× larger steps than 0.95 and unfreezes learning. See `optimizers.py`
  docstring for the full α-cancels derivation.
- **α = 1.0, κ = 2.0** (paper values). α cancels in ObGD's bound-active regime;
  κ is the overshoot safety bound, never a tuning dial.
- **LayerNorm at every layer + sparse init 90%** (paper §3.3, Appendix F).
  Load-bearing for ObGD: normalized features → stable trace → stable α_eff.
- **Aux task OFF in dense baseline** (on only when event threshold > 0).
  At θ=0 the event mask is garbage and aux poisons the shared body.
- **Obs normalization clip ±5 + 1000-frame warmup** (fixes early Welford blowup).
- **No ScaleReward** for Pong (rewards are already ideal ±1; scaling was unstable).
- **Corrected loss formulation** (paper-exact, matches official stream_ac code):
  no `td_err` in the value/policy loss (ObGD supplies δ internally — putting it
  in the loss double-counts to δ²); entropy term uses `sign(δ)` not `|δ|` so the
  ObGD×δ product gives `|δ|·ascent` (always pushes entropy UP, scaled by
  surprise). The spec's §2.8 pseudocode had δ in the loss, which flipped the
  entropy sign when losing and caused entropy collapse. See `agent.py` docstring.
- **Homeostat dead-layer recovery** (spec-faithful): `adapt_rate` 1e-3→1e-2 (10×
  faster so it recovers from overshoot in ~700 frames not ~7000), plus a
  dead-layer safety net (if rate==0 for 100 steps, cut θ by 0.3× repeatedly
  until the layer wakes). Keeps deeper event layers alive.

## Why the GUI looks fast-forwarded (and why that's correct during training)

The `MaxAndSkipObservation(skip=4)` wrapper means **every `env.step()` advances
the ALE emulator by 4 raw frames** and returns the max of the last 2. So with
render-every-step + `--fps 60`, the displayed ball moves at 60×4 = 240 ALE
frames/sec = **4× real Pong speed**. This is **correct and desirable during
training** — action repeat (frame skip 4) is standard Atari RL (paper Appendix
F: "Each action taken by the agent is repeated 4 times"), and fast-forward =
more learning per second. The learning is correct: one decision + one update
per 4 ALE frames, exactly as the paper does it.

Do NOT slow training to make the GUI watchable — that would 4× the wall-clock
time for 5M frames (~37h instead of ~15h) for no learning benefit.

## Future plan: inference player (`seal/play.py` — to be built after training)

Training and inference want **opposite speeds**:

| | training | inference |
|---|---|---|
| goal | learn as fast as possible | watch the agent play |
| speed | fast-forward is *good* (more frames/sec = faster learning) | real-time (60 ALE fps, watchable) |
| learning updates | yes (every step) | no (just play, no updates) |
| rendering | optional, cosmetic | the whole point |

**Plan: build `seal/play.py` after training completes.** It will:
1. Load a trained checkpoint (`results/seal-pong_best.pt`).
2. **No learning** — just forward passes + action selection. No ObGD, no traces,
   no eligibility updates. Pure policy execution.
3. Render at **real Pong speed**: cap to 15 `env.step()`/sec = 60 ALE fps (with
   frame_skip=4), so the ball moves at normal speed and you can actually watch.
4. Show live score, episode return, and a "best episode so far" tracker.
5. Optional `--fps` override if you want to watch at 0.5× or 2× speed.

Usage will be roughly:
```bash
python -m seal.play --checkpoint results/seal-pong_best.pt       # real-time, watch it play
python -m seal.play --checkpoint results/seal-pong_best.pt --fps 30   # 0.5× slow-mo
```

This is the "show off" moment — after the agent has trained, watch it play Pong
at real speed and see if it's actually good. Build this only after training is
done or you have a checkpoint you're happy with (no point playing an untrained
agent).
