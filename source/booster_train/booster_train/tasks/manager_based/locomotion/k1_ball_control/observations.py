# Copyright (c) 2026 - K1 Ball Control (RL, robot-soccer first touch)
# SPDX-License-Identifier: BSD-3-Clause

"""Ball-state observations, expressed relative to the robot's own body frame
(egocentric bearing/distance/approach-velocity - not a privileged world-frame
readout), matching the mock of a real vision-based ball estimate."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


def ball_pos_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    rel_w = ball.data.root_pos_w - robot.data.root_pos_w
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_w)


def ball_lin_vel_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    rel_vel_w = ball.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_vel_w)
