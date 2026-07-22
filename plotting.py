"""Plotting for SEAL results (spec §5).

Produces:
  - return curve (dense vs seal)
  - FLOPs/step comparison (event vs dense over steps)
  - plasticity curves (dormant fraction, frac weights updated, feature rank)
  - event heatmap (already saved by Stage-1 test)

Reads the CSVs written by seal.train.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv


def _read_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


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
        steps = [int(r["step"]) for r in rows if r["return"] not in ("", None)]
        rets = [_to_float(r["return"]) for r in rows if r["return"] not in ("", None)]
        if rets:
            # rolling mean
            window = max(1, len(rets) // 50)
            rets_smooth = np.convolve(rets, np.ones(window) / window, mode="valid")
            ax.plot(steps[:len(rets_smooth)], rets_smooth, label=name)
    ax.set_xlabel("env step"); ax.set_ylabel("episode return (smoothed)")
    ax.set_title("SEAL return curve"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, save), dpi=100); plt.close(fig)


def plot_flops_comparison(out_dir, runs, save="flops_comparison.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, path in runs.items():
        if not os.path.exists(path):
            continue
        rows = _read_csv(path)
        steps = [int(r["step"]) for r in rows]
        ev = [_to_float(r["event_flops"]) for r in rows]
        dn = [_to_float(r["dense_flops"]) for r in rows]
        if ev:
            ax.plot(steps, ev, label=f"{name} event FLOPs")
        if dn:
            ax.plot(steps, dn, label=f"{name} dense FLOPs", ls="--")
    ax.set_xlabel("env step"); ax.set_ylabel("analytic FLOPs / step")
    ax.set_title("SEAL FLOPs/step (analytic)"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, save), dpi=100); plt.close(fig)


def plot_plasticity(out_dir, runs, save="plasticity.png"):
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    for name, path in runs.items():
        if not os.path.exists(path):
            continue
        rows = _read_csv(path)
        steps = [int(r["step"]) for r in rows]
        dormant = [_to_float(r["dormant_frac"]) for r in rows]
        fracupd = [_to_float(r["frac_weights_updated"]) for r in rows]
        rank = [_to_float(r["feat_rank"]) for r in rows]
        axs[0].plot(steps, dormant, label=name)
        axs[1].plot(steps, fracupd, label=name)
        axs[2].plot(steps, rank, label=name)
    axs[0].set_ylabel("dormant fraction"); axs[0].grid(True, alpha=0.3)
    axs[1].set_ylabel("frac weights updated"); axs[1].grid(True, alpha=0.3)
    axs[2].set_ylabel("feature rank"); axs[2].set_xlabel("env step")
    axs[2].grid(True, alpha=0.3)
    for a in axs: a.legend()
    fig.suptitle("SEAL plasticity curves"); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, save), dpi=100); plt.close(fig)


def make_all(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    runs = {
        "dense": os.path.join(out_dir, "dense_seed0.csv"),
        "seal": os.path.join(out_dir, "seal_seed0.csv"),
    }
    plot_return_curve(out_dir, runs)
    plot_flops_comparison(out_dir, runs)
    plot_plasticity(out_dir, runs)
    print(f"plots written to {out_dir}/")


if __name__ == "__main__":
    make_all()
