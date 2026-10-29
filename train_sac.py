#!/usr/bin/env python3
"""Train SAC on the Hulkbuster balance/reach/lift/carry stages."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import HulkbusterLiftEnv

INFO_KEYS = ("success", "hand_dist", "box_z", "grasped")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--kp", type=float, default=5.0)
    p.add_argument("--kv", type=float, default=0.30)
    p.add_argument("--action-scale", type=float, default=0.5)
    p.add_argument("--push-prob", type=float, default=0.0015)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gradient-steps", type=int, default=-1)
    p.add_argument("--target-entropy", type=float, default=-8.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--no-randomize", dest="randomize", action="store_false")
    p.add_argument("--no-gravity-comp", dest="gravity_comp", action="store_false")
    p.set_defaults(randomize=True, gravity_comp=True)
    args = p.parse_args()
    if args.run_name is None:
        args.run_name = f"sac_stage{args.stage}"
    if args.out is None:
        args.out = Path(__file__).resolve().parent / "runs" / args.run_name
    return args


def make_env(args: argparse.Namespace, rank: int):
    def _init():
        env = HulkbusterLiftEnv(
            stage=args.stage,
            randomize=args.randomize,
            push_prob=args.push_prob,
            kp_scale=args.kp,
            kv_scale=args.kv,
            gravity_comp=args.gravity_comp,
            action_scale_mult=args.action_scale,
        )
        env.reset(seed=args.seed + rank)
        return env

    return _init


def make_eval_env(args: argparse.Namespace) -> DummyVecEnv:
    def _init():
        env = HulkbusterLiftEnv(
            stage=args.stage,
            randomize=False,
            push_prob=0.0,
            kp_scale=args.kp,
            kv_scale=args.kv,
            gravity_comp=args.gravity_comp,
            action_scale_mult=args.action_scale,
        )
        return env

    return VecMonitor(DummyVecEnv([_init]), info_keywords=INFO_KEYS)


def last_progress_values(log_dir: Path) -> dict[str, str]:
    path = log_dir / "progress.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, str] = {}
    for row in rows:
        for key, value in row.items():
            if value not in ("", None):
                out[key] = value
    return out


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    ckpt_dir = out / "checkpoints"
    best_dir = out / "best"
    eval_dir = out / "eval"

    venv = SubprocVecEnv(
        [make_env(args, i) for i in range(args.n_envs)], start_method="fork"
    ) if args.n_envs > 1 else DummyVecEnv([make_env(args, 0)])
    venv = VecMonitor(venv, info_keywords=INFO_KEYS)
    eval_env = make_eval_env(args)

    if args.resume is not None:
        print(f"loading {args.resume}", flush=True)
        model = SAC.load(args.resume, env=venv, device=args.device)
        model.set_logger(configure(str(log_dir), ["stdout", "csv"]))
    else:
        model = SAC(
            "MlpPolicy",
            venv,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=1,
            gradient_steps=args.gradient_steps,
            ent_coef="auto",
            target_entropy=args.target_entropy,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
        with torch.no_grad():
            model.policy.actor.log_std.weight.zero_()
            model.policy.actor.log_std.bias.fill_(-1.5)
        model.set_logger(configure(str(log_dir), ["stdout", "csv"]))

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(eval_dir),
            eval_freq=max(1, args.eval_freq // args.n_envs),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
            warn=False,
        ),
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.n_envs),
            save_path=str(ckpt_dir),
            name_prefix=f"sac_stage{args.stage}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
    ]

    print(
        f"stage={args.stage} timesteps={args.timesteps} envs={args.n_envs} "
        f"action_scale={args.action_scale} kp={args.kp} kv={args.kv} "
        f"out={out}",
        flush=True,
    )
    total_timesteps = args.timesteps
    if args.resume is not None:
        total_timesteps += model.num_timesteps
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        reset_num_timesteps=args.resume is None,
        progress_bar=False,
    )
    final_path = out / f"sac_stage{args.stage}"
    model.save(final_path)
    print(f"saved {final_path}.zip", flush=True)

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    print(f"final eval reward {mean_reward:.2f} +/- {std_reward:.2f}", flush=True)
    row = last_progress_values(log_dir)
    for key in (
        "rollout/ep_rew_mean", "rollout/ep_len_mean", "rollout/success",
        "rollout/hand_dist", "rollout/box_z", "rollout/grasped",
        "eval/mean_reward", "eval/mean_ep_length",
    ):
        if key in row:
            print(f"  {key}: {row[key]}", flush=True)


if __name__ == "__main__":
    main()
