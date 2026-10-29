#!/usr/bin/env python3
"""Render a deterministic policy rollout to MP4 (and optionally GIF).

The offscreen renderer uses EGL by default.  If EGL is unavailable, try
``MUJOCO_GL=osmesa`` or run on a machine with a display.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
from stable_baselines3 import SAC

from env import CONTROL_DT, HulkbusterLiftEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stage", type=int, required=True, choices=(0, 1, 2, 3))
    p.add_argument("--out", default=None)
    p.add_argument("--gif", default=None)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-scale", type=float, default=0.5)
    p.add_argument("--lookat", type=float, nargs=3, default=None)
    p.add_argument("--distance", type=float, default=None)
    p.add_argument("--azimuth", type=float, default=140.0)
    p.add_argument("--elevation", type=float, default=-12.0)
    p.add_argument("--stochastic", action="store_true")
    return p.parse_args()


def ffmpeg_run(cmd: list[str]) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out = Path(args.out or f"assets/stage{args.stage}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    env = HulkbusterLiftEnv(
        stage=args.stage,
        randomize=False,
        push_prob=0.0,
        action_scale_mult=args.action_scale,
    )
    env.model.vis.global_.offwidth = max(1280, args.width)
    env.model.vis.global_.offheight = max(720, args.height)
    renderer = mujoco.Renderer(
        env.model, height=args.height, width=args.width)
    policy = SAC.load(args.model, device="cpu")

    if args.lookat is None:
        lookat = [0.55, 0.0, 0.80]
        if args.stage == 2:
            lookat = [0.75, 0.0, 0.70]
        elif args.stage == 3:
            lookat = [0.95, 0.0, 0.65]
    else:
        lookat = args.lookat
    if args.distance is None:
        distance = 3.1 if args.stage == 3 else 2.9
    else:
        distance = args.distance

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation

    obs, info = env.reset(seed=args.seed)
    frames_dir = Path(tempfile.mkdtemp(prefix="hulk_video_"))
    frames = 0
    try:
        for i in range(int(args.seconds / CONTROL_DT)):
            action, _ = policy.predict(
                obs, deterministic=not args.stochastic)
            obs, _, terminated, truncated, info = env.step(action)
            renderer.update_scene(env.data, camera=cam)
            imageio.imwrite(frames_dir / f"{i:05d}.png", renderer.render())
            frames += 1
            if terminated or truncated:
                break

        ffmpeg_run([
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(frames_dir / "%05d.png"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out),
        ])
        print(f"wrote {out} ({frames} frames)")
        if args.gif:
            gif = Path(args.gif)
            gif.parent.mkdir(parents=True, exist_ok=True)
            ffmpeg_run([
                "ffmpeg", "-y", "-i", str(out),
                "-vf", "fps=15,scale=640:-1:flags=lanczos,"
                       "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                "-loop", "0", str(gif),
            ])
            print(f"wrote {gif}")
        print("final info:", {
            k: info.get(k) for k in (
                "stage", "box_z", "hand_dist", "grasped", "success")
        })
    finally:
        renderer.close()
        env.close()
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
