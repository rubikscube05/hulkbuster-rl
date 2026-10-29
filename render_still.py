#!/usr/bin/env python3
"""Render a still of the nominal Hulkbuster in the training scene."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco

from env import HulkbusterLiftEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="assets/design.png")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--lookat", type=float, nargs=3, default=[0.60, 0.0, 0.78])
    p.add_argument("--distance", type=float, default=3.1)
    p.add_argument("--azimuth", type=float, default=140.0)
    p.add_argument("--elevation", type=float, default=-12.0)
    args = p.parse_args()

    env = HulkbusterLiftEnv(stage=0, randomize=False, push_prob=0.0)
    env.model.vis.global_.offwidth = max(1280, args.width)
    env.model.vis.global_.offheight = max(720, args.height)
    renderer = mujoco.Renderer(
        env.model, height=args.height, width=args.width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = args.lookat
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation

    try:
        env.reset(seed=0)
        renderer.update_scene(env.data, camera=cam)
        frame = renderer.render()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out, frame)
        print(f"wrote {out} ({frame.shape[1]}x{frame.shape[0]})")
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
