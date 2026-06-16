# Copyright (c) 2026 - K1 Walk live viewer
# SPDX-License-Identifier: BSD-3-Clause

"""Live viewer for an RSL-RL run that is still training.

Runs a small number of envs (default 8) in a viewer and **hot-reloads the newest
checkpoint** as the background trainer writes them — so you keep one window open
and it keeps updating to the latest policy without ever restarting the sim or
touching the training process.

Example:
    ~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_live.py --device cuda:0
    # or pin a specific run / env count / reload cadence:
    ~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_live.py --device cuda:0 \
        --num_envs 8 --load_run 2026-06-14_21-00-00 --reload_interval_s 5
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Live-following viewer for an RSL-RL run.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to display.")
parser.add_argument("--task", type=str, default="Booster-K1-Walk-v0-Play", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--reload_interval_s", type=float, default=5.0,
    help="How often (seconds) to check for and load a newer checkpoint.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments (adds --load_run, --checkpoint, --experiment_name, etc.)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import re
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_train.tasks  # noqa: F401


_CKPT_RE = re.compile(r"model_(\d+)\.pt$")


def _find_latest_run_dir(log_root_path: str) -> str:
    """Newest timestamped run directory under the experiment log root."""
    candidates = [
        os.path.join(log_root_path, d)
        for d in os.listdir(log_root_path)
        if os.path.isdir(os.path.join(log_root_path, d)) and d != "exported"
    ]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {log_root_path}")
    return max(candidates, key=os.path.getmtime)


def _latest_checkpoint(run_dir: str) -> str | None:
    """Highest-iteration model_<N>.pt in a run directory, or None if there are none."""
    best_path, best_n = None, -1
    for name in os.listdir(run_dir):
        m = _CKPT_RE.search(name)
        if m and int(m.group(1)) > best_n:
            best_n, best_path = int(m.group(1)), os.path.join(run_dir, name)
    return best_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Cluster the few viewer envs together on flat ground so they're all in one
    # frame. The policy is "blind" (terrain never enters its observations), so it
    # behaves the same here as on the training terrain.
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.env_spacing = 1.5
    env_cfg.curriculum.terrain_levels = None

    # resolve the run directory we will follow
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.load_run and args_cli.load_run not in (".*", ""):
        run_dir = os.path.join(log_root_path, args_cli.load_run)
    else:
        run_dir = _find_latest_run_dir(log_root_path)
    print(f"[INFO] Following run directory: {run_dir}")

    # wait for the first checkpoint to exist (trainer may have just started)
    resume_path = _latest_checkpoint(run_dir)
    while resume_path is None and simulation_app.is_running():
        print(f"[INFO] No checkpoint yet in {run_dir} — waiting...")
        time.sleep(args_cli.reload_interval_s)
        resume_path = _latest_checkpoint(run_dir)
    if resume_path is None:
        return

    # create environment + rsl-rl wrapper
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading initial checkpoint: {resume_path}", flush=True)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    last_loaded = resume_path

    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()

    last_check = time.time()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # periodically pull in a newer checkpoint without restarting the sim
        if time.time() - last_check >= args_cli.reload_interval_s:
            last_check = time.time()
            newest = _latest_checkpoint(run_dir)
            if newest is not None and newest != last_loaded:
                try:
                    ppo_runner.load(newest)  # loads weights + normalizer in place
                    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
                    last_loaded = newest
                    print(f"[INFO] Hot-reloaded checkpoint: {os.path.basename(newest)}", flush=True)
                except Exception as e:  # partially-written file, etc. — retry next interval
                    print(f"[WARN] Could not load {os.path.basename(newest)} yet ({e}); will retry.", flush=True)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
