"""Plotting for SEAL e-prop LSNN results.

Reads the CSV written by train.py (columns: step, episode, return, td_err, v,
spike_rate_hz, policy_entropy, b_drift, tag_norm_win, tag_norm_wrec,
dormant_frac, max_episode_len) and produces:
  - return curve (episode return over training)
  - learning health (TD error, V, policy entropy, B_jk drift)
  - spiking activity (spike rate, dormant fraction, tag norms)
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(x, default=np.nan):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def plot_return_curve(out_dir, runs, save="return_curve.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, path in runs.items():
        if not os.path.exists(path):
            continue
        rows = _read_csv(path)
        eps = [int(r["episode"]) for r in rows if r["return"] not in ("", None)]
        rets = [_to_float(r["return"]) for r in rows if r["return"] not in ("", None)]
        if rets:
            window = max(1, len(rets) // 50)
            rets_smooth = np.convolve(rets, np.ones(window) / window, mode="valid")
            ax.plot(eps[:len(rets_smooth)], rets_smooth, label=name)
    ax.set_xlabel("episode"); ax.set_ylabel("episode return (smoothed)")
    ax.set_title("SEAL e-prop return curve"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, save), dpi=100); plt.close(fig)


def plot_health(out_dir, path, save="health.png"):
    if not os.path.exists(path):
        return
    rows = _read_csv(path)
    steps = [_to_float(r["step"]) for r in rows]
    td = [_to_float(r["td_err"]) for r in rows]
    v = [_to_float(r["v"]) for r in rows]
    ent = [_to_float(r["policy_entropy"]) for r in rows]
    bdrift = [_to_float(r["b_drift"]) for r in rows]
    fig, axs = plt.subplots(4, 1, figsize=(7, 9), sharex=True)
    axs[0].plot(steps, td); axs[0].set_ylabel("|TD error|"); axs[0].grid(True, alpha=0.3)
    axs[1].plot(steps, v); axs[1].set_ylabel("V (critic)"); axs[1].grid(True, alpha=0.3)
    axs[2].plot(steps, ent); axs[2].set_ylabel("policy entropy"); axs[2].grid(True, alpha=0.3)
    axs[3].plot(steps, bdrift); axs[3].set_ylabel("B_jk drift (no-op: symmetric)")
    axs[3].set_xlabel("env step"); axs[3].grid(True, alpha=0.3)
    fig.suptitle("SEAL e-prop learning health"); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, save), dpi=100); plt.close(fig)


def plot_spiking(out_dir, path, save="spiking.png"):
    if not os.path.exists(path):
        return
    rows = _read_csv(path)
    steps = [_to_float(r["step"]) for r in rows]
    hz = [_to_float(r["spike_rate_hz"]) for r in rows]
    dorm = [_to_float(r["dormant_frac"]) for r in rows]
    tw = [_to_float(r["tag_norm_win"]) for r in rows]
    tr = [_to_float(r["tag_norm_wrec"]) for r in rows]
    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axs[0].plot(steps, hz); axs[0].set_ylabel("spike rate (Hz)"); axs[0].grid(True, alpha=0.3)
    axs[1].plot(steps, dorm); axs[1].set_ylabel("dormant fraction"); axs[1].grid(True, alpha=0.3)
    axs[2].plot(steps, tw, label="Win tag"); axs[2].plot(steps, tr, label="Wrec tag")
    axs[2].set_ylabel("eligibility tag norm"); axs[2].set_xlabel("env step")
    axs[2].grid(True, alpha=0.3); axs[2].legend()
    fig.suptitle("SEAL e-prop spiking activity"); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, save), dpi=100); plt.close(fig)


def make_all(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "seal_eprop_s0.csv")
    runs = {"seal_eprop": path}
    plot_return_curve(out_dir, runs)
    plot_health(out_dir, path)
    plot_spiking(out_dir, path)
    print(f"plots written to {out_dir}/")


if __name__ == "__main__":
    make_all()
