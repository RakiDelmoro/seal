"""SEAL run comparison — competitiveness metrics across A/B runs.

Win rate at ~0.5% is too sparse to trust at the 120k-frame budget (control
runs themselves vary 0.52–0.65% seed-to-seed). Episode survival is the
denser, steadier competitiveness signal, per the reading that a longer
episode means the agent is holding its own against the opponent:

  frames_per_loss = episode_length / losses   (how long each rally survived)
  episode_length                              (average rally length)
  win rate                                    (the goal, but noisy)

Handles appended CSVs (a restart resets step_count) by analyzing the LAST
segment of each file — so re-run experiments compare cleanly.

Usage:
  python analyze_runs.py results/run_a.csv results/run_b.csv ...
"""
from __future__ import annotations
import csv
import sys
import numpy as np


def load_last_segment(path: str) -> list[dict]:
    """Load a metrics CSV and keep only the last contiguous run segment."""
    rows = list(csv.DictReader(open(path)))
    idx, prev = 0, -1
    for i, r in enumerate(rows):
        sc = int(r["step_count"])
        if sc < prev - 1000:  # a big drop marks a restarted run
            idx = i
        prev = sc
    return rows[idx:]


def summarize(rows: list[dict]) -> dict:
    ep_len = np.array([int(r["episode_length"]) for r in rows])
    lost = np.array([int(r["lost"]) for r in rows])
    scored = np.array([int(r["scored"]) for r in rows])
    fpl = ep_len / np.maximum(lost, 1)
    return {
        "episodes": len(rows),
        "frames": int(rows[-1]["step_count"]),
        "episode_length": float(ep_len.mean()),
        "frames_per_loss": float(fpl.mean()),
        "wins": int(scored.sum()),
        "losses": int(lost.sum()),
        "win_rate": 100 * scored.sum() / max(scored.sum() + lost.sum(), 1),
    }


def buckets(rows: list[dict], key: str, width: int = 100) -> list[tuple]:
    """(start_episode, mean) buckets over the run for a trend view."""
    out = []
    for b in range(0, len(rows), width):
        chunk = rows[b:b + width]
        out.append((b, float(np.mean([float(r[key]) for r in chunk]))))
    return out


def main(paths: list[str]):
    if len(paths) < 1:
        print(__doc__)
        return
    data = {p: load_last_segment(p) for p in paths}

    print("=" * 90)
    print(f"{'run':<42}{'eps':>6}{'ep_len':>9}{'frm/loss':>10}"
          f"{'wins':>7}{'losses':>8}{'win%':>8}")
    print("=" * 90)
    for p, rows in data.items():
        s = summarize(rows)
        name = p.split("/")[-1][:40]
        print(f"{name:<42}{s['episodes']:>6}{s['episode_length']:>9.1f}"
              f"{s['frames_per_loss']:>10.2f}{s['wins']:>7}{s['losses']:>8}"
              f"{s['win_rate']:>8.3f}")

    print("\nTrend: frames_per_loss per 100-episode bucket "
          "(higher = holding out longer = more competitive)")
    for p, rows in data.items():
        name = p.split("/")[-1][:40]
        line = " ".join(f"{v:5.2f}" for _, v in buckets(rows, "episode_length"))
        print(f"  {name:<42} {line}")

    print("\nTrend: win rate % per 100-episode bucket")
    for p, rows in data.items():
        name = p.split("/")[-1][:40]
        vals = []
        for b in range(0, len(rows), 100):
            chunk = rows[b:b + 100]
            w = sum(int(r["scored"]) for r in chunk)
            l = sum(int(r["lost"]) for r in chunk)
            vals.append(100 * w / max(w + l, 1))
        print(f"  {name:<42} " + " ".join(f"{v:5.2f}" for v in vals))


if __name__ == "__main__":
    main(sys.argv[1:])
