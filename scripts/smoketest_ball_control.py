# Copyright (c) 2026 - K1 Ball Control (RL, robot-soccer first touch)
# SPDX-License-Identifier: BSD-3-Clause

"""Fast headless smoke test: build the env, reset, step a few times with
random actions, print obs/reward shapes and any per-term errors."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

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
    cfg.scene.num_envs = 16
    env = gym.make("Booster-K1-BallControl-v0-Play", cfg=cfg)

    obs_dict, _ = env.reset()
    print("RESET_OK")
    print("ball_target_dir", env.unwrapped._ball_target_dir.tolist())
    print("policy_obs_shape", obs_dict["policy"].shape)
    print("critic_obs_shape", obs_dict["critic"].shape)
    print("action_dim", env.unwrapped.action_manager.total_action_dim)

    ball = env.unwrapped.scene["ball"]
    robot = env.unwrapped.scene["robot"]
    print("env_origins", env.unwrapped.scene.env_origins.tolist())
    print("robot_default_root_state_pos", robot.data.default_root_state[:, :3].tolist())
    print("ball_default_root_state_pos", ball.data.default_root_state[:, :3].tolist())
    print("ball_pos_all", ball.data.root_pos_w.tolist())
    print("robot_pos_all", robot.data.root_pos_w.tolist())
    print("ball_vel0", ball.data.root_lin_vel_w[0].tolist())

    action_dim = env.unwrapped.action_manager.total_action_dim
    for i in range(60):
        actions = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)
        obs_dict, rew, terminated, truncated, info = env.step(actions)
        if i % 20 == 0:
            print(
                f"STEP {i:3d} reward_mean={rew.mean().item():.4f} "
                f"ball_speed={torch.norm(ball.data.root_lin_vel_w[0, :2]).item():.3f} "
                f"robot_z={robot.data.root_pos_w[0, 2].item():.3f}"
            )

    print("SMOKETEST_DONE")


if __name__ == "__main__":
    main()
    os._exit(0)
