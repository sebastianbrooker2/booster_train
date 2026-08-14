# Copyright (c) 2026 - K1 Ball Control (RL, robot-soccer first touch)
# SPDX-License-Identifier: BSD-3-Clause

"""Viewer sanity-check for the k1_ball_control spawn/reset geometry (NOT trained
behavior - the policy is untrained, so actions are held at zero/default pose).
Confirms visually: robot scatter within the spawn disk, facing window relative
to the ball, ball launched from a fixed point toward the disk CENTER (not at
the robot), and per-env ball size variety.

Usage:
    ./isaaclab.sh -p scripts/demo_ball_control_viewer.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="K1 ball-control spawn/reset geometry sanity check.")
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import gymnasium as gym

import booster_train.tasks  # noqa: F401 - registers gym task ids
from booster_train.tasks.manager_based.locomotion.k1_ball_control.ball_control_env_cfg import (
    K1BallControlEnvCfg_PLAY,
)


def main():
    cfg = K1BallControlEnvCfg_PLAY()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = 8.0  # roomy, so the 3 m ball throw doesn't cross into neighboring envs

    env = gym.make("Booster-K1-BallControl-v0-Play", cfg=cfg)
    obs_dict, _ = env.reset()

    action_dim = env.unwrapped.action_manager.total_action_dim
    zero_actions = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)

    print("STARTING_VIEWER (close the window to stop)")
    step = 0
    while simulation_app.is_running():
        obs_dict, _, terminated, truncated, _ = env.step(zero_actions)
        if step % 100 == 0:
            robot = env.unwrapped.scene["robot"]
            ball = env.unwrapped.scene["ball"]
            print(
                f"STEP {step:5d}  robot_z_mean={robot.data.root_pos_w[:, 2].mean().item():.3f}  "
                f"resets_this_tick={int((terminated | truncated).sum().item())}"
            )
        step += 1

    print("DEMO_DONE")


if __name__ == "__main__":
    main()
    os._exit(0)
