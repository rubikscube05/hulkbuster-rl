#!/usr/bin/env python3
"""Evaluate a trained SAC policy on the Hulkbuster warehouse stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import HulkbusterLiftEnv, MAX_EPISODE_STEPS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--stage", type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--kp", type=float, default=5.0)
    p.add_argument("--kv", type=float, default=0.30)
    p.add_argument("--action-scale", type=float, default=0.5)
    p.add_argument("--render", action="store_true")
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.add_argument("--no-gravity-comp", dest="gravity_comp", action="store_false")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(deterministic=True, gravity_comp=True)
    return p.parse_args()


def default_model(args: argparse.Namespace) -> Path:
    run = Path(__file__).resolve().parent / "runs"
    candidates = [
        run / f"sac_stage{args.stage}" / "best" / "best_model.zip",
        run / f"sac_stage{args.stage}" / f"sac_stage{args.stage}.zip",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "no model found; pass --model or train first: " + ", ".join(map(str, candidates))
    )


def main() -> None:
    args = parse_args()
    model_path = args.model or default_model(args)
    model = SAC.load(model_path, device=args.device)
    env = HulkbusterLiftEnv(
        stage=args.stage,
        randomize=False,
        push_prob=0.0,
        render_mode="human" if args.render else None,
        kp_scale=args.kp,
        kv_scale=args.kv,
        gravity_comp=args.gravity_comp,
        action_scale_mult=args.action_scale,
    )

    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ret = 0.0
        min_hand = float("inf")
        max_box_z = -float("inf")
        info = {}
        for step in range(1, MAX_EPISODE_STEPS + 1):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, term, trunc, info = env.step(action)
            ret += float(reward)
            min_hand = min(min_hand, info["hand_dist"])
            max_box_z = max(max_box_z, info["box_z"])
            if term or trunc:
                break
        row = {
            "episode": ep,
            "return": round(ret, 2),
            "length": step,
            "final_root_z": round(info["root_z"], 3),
            "final_up": round(info["up"], 3),
            "min_hand_dist": round(min_hand, 3),
            "final_hand_dist": round(info["hand_dist"], 3),
            "max_box_z": round(max_box_z, 3),
            "final_box_z": round(info["box_z"], 3),
            "box_goal_dist": round(info["box_goal_dist"], 3),
            "grasped": int(info["grasped"]),
            "success": int(info["success"]),
        }
        rows.append(row)
        print(
            f"ep {ep:2d} ret {row['return']:8.2f} len {row['length']:3d} "
            f"root_z {row['final_root_z']:.2f} up {row['final_up']:.2f} "
            f"hand {row['min_hand_dist']:.2f} box_z {row['max_box_z']:.2f} "
            f"goal {row['box_goal_dist']:.2f} grasp {row['grasped']} "
            f"success {row['success']}",
            flush=True,
        )

    summary = {
        "model": str(model_path),
        "stage": args.stage,
        "episodes": args.episodes,
        "mean_return": round(float(np.mean([r["return"] for r in rows])), 3),
        "mean_length": round(float(np.mean([r["length"] for r in rows])), 2),
        "success_rate": round(float(np.mean([r["success"] for r in rows])), 3),
        "grasp_rate": round(float(np.mean([r["grasped"] for r in rows])), 3),
        "mean_max_box_z": round(float(np.mean([r["max_box_z"] for r in rows])), 3),
        "mean_min_hand_dist": round(float(np.mean([r["min_hand_dist"] for r in rows])), 3),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
