#!/usr/bin/env python3
"""Plot the EvalCallback curves for each curriculum stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STAGE_NAMES = {0: "balance", 1: "reach", 2: "lift", 3: "carry"}
COLORS = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/smoke")
    p.add_argument("--out", default="assets/training_curve.png")
    args = p.parse_args()

    runs = Path(args.runs)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax2 = ax.twinx()
    plotted = 0

    for stage in range(4):
        path = runs / f"stage{stage}" / "eval" / "evaluations.npz"
        if not path.exists():
            continue
        data = np.load(path)
        steps = data["timesteps"]
        rewards = data["results"]
        lengths = data["ep_lengths"]
        color = COLORS[stage]
        label = f"stage {stage}: {STAGE_NAMES[stage]}"
        ax.plot(steps, rewards.mean(axis=1), color=color, lw=2, label=label)
        ax.fill_between(
            steps,
            rewards.mean(axis=1) - rewards.std(axis=1),
            rewards.mean(axis=1) + rewards.std(axis=1),
            color=color,
            alpha=0.15,
        )
        ax2.plot(
            steps, lengths.mean(axis=1), color=color, lw=1.2,
            linestyle="--", alpha=0.75)
        plotted += 1

    if plotted == 0:
        raise SystemExit(f"no evaluations.npz found under {runs}")

    ax.set_xlabel("environment steps within stage")
    ax.set_ylabel("episode return (mean ± std)")
    ax2.set_ylabel("episode length (dashed)")
    ax.set_title("Hulkbuster staged SAC — evaluation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
