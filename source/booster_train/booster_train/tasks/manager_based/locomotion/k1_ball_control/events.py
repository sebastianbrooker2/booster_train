# Copyright (c) 2026 - K1 Ball Control (RL, robot-soccer first touch)
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reset event for the K1 ball-control task.

Spawns the robot at a random point in a small disk, facing mostly square-on to
the ball with some spread either side (a downstream walk-to-ball controller is
expected to square the robot up before handing off to this policy, so training
only needs to cover realistic heading error, not the full side-on case). The
ball is always aimed at the CENTER of that spawn disk (not at the robot's
actual randomized position) - the robot has to use its own footwork/positioning
to actually get a foot on the ball, rather than the ball conveniently homing in
on wherever it stands.

Also samples a per-episode TARGET DIRECTION - where the ball's controlled touch
should send it (biased toward continuing the way it came, since that's the most
likely next-pass direction, with some fully-random episodes for generalization).
Facing, the ball's aim, and the target direction all key off the robot's sampled
position, so this has to be one function, not several independent
`reset_root_state_uniform` terms.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils


def reset_robot_and_ball(
    env,
    env_ids: torch.Tensor,
    spawn_radius: float = 0.5,
    facing_window: float = 0.7854,  # +/- rad around "directly facing the ball" (pi/4 = 45 deg)
    square_on_fraction: float = 0.4,  # fraction of resets that start (near-)square-on
    ball_speed_range: tuple[float, float] = (1.5, 4.5),
    target_dir_spread: float = 0.5236,  # +/- rad around the incoming direction (pi/6 = 30 deg)
    target_dir_random_fraction: float = 0.25,  # fraction of episodes with a fully random target
    robot_name: str = "robot",
    ball_name: str = "ball",
) -> None:
    robot: Articulation = env.scene[robot_name]
    ball: RigidObject = env.scene[ball_name]
    n = len(env_ids)
    device = env.device

    # --- robot: uniform point in a disk of radius `spawn_radius` ------------
    robot_state = robot.data.default_root_state[env_ids].clone()
    center_pos = robot_state[:, 0:3] + env.scene.env_origins[env_ids]  # disk center = nominal spawn point

    r = spawn_radius * torch.sqrt(torch.rand(n, device=device))
    theta = torch.rand(n, device=device) * 2 * torch.pi
    robot_pos = center_pos.clone()
    robot_pos[:, 0] += r * torch.cos(theta)
    robot_pos[:, 1] += r * torch.sin(theta)

    # --- ball: fixed local spawn point (no randomization on the ball itself) -
    ball_state = ball.data.default_root_state[env_ids].clone()
    ball_pos = ball_state[:, 0:3] + env.scene.env_origins[env_ids]

    # --- robot yaw: mostly square-on to the ball, some spread either side ----
    to_ball = ball_pos[:, :2] - robot_pos[:, :2]
    face_ball_yaw = torch.atan2(to_ball[:, 1], to_ball[:, 0])
    square_on = torch.rand(n, device=device) < square_on_fraction
    wide_offset = (torch.rand(n, device=device) * 2 - 1) * facing_window
    narrow_offset = (torch.rand(n, device=device) * 2 - 1) * (facing_window * 0.15)
    yaw_offset = torch.where(square_on, narrow_offset, wide_offset)
    yaw = face_ball_yaw + yaw_offset
    zeros = torch.zeros(n, device=device)
    quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    robot_state[:, 0:3] = robot_pos
    robot_state[:, 3:7] = quat
    robot.write_root_pose_to_sim(robot_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(robot_state[:, 7:], env_ids=env_ids)

    # --- ball: launched at the CENTER of the spawn disk, not at the robot's
    # actual (randomized) position -------------------------------------------
    to_center = center_pos[:, :2] - ball_pos[:, :2]
    dist = torch.norm(to_center, dim=-1).clamp(min=1.0e-6)
    direction = to_center / dist.unsqueeze(-1)
    speed = torch.empty(n, device=device).uniform_(*ball_speed_range)

    ball_state[:, 0:3] = ball_pos
    ball_state[:, 7] = direction[:, 0] * speed
    ball_state[:, 8] = direction[:, 1] * speed
    ball_state[:, 9] = 0.0
    ball.write_root_pose_to_sim(ball_state[:, :7], env_ids=env_ids)
    ball.write_root_velocity_to_sim(ball_state[:, 7:], env_ids=env_ids)

    # remember each env's incoming direction - ball_overshoot_penalty uses it
    # to tell "got away past the robot" from "still nearby"
    if not hasattr(env, "_ball_incoming_dir") or env._ball_incoming_dir.shape[0] != env.num_envs:
        env._ball_incoming_dir = torch.zeros(env.num_envs, 2, device=device)
    env._ball_incoming_dir[env_ids] = direction

    # --- target direction: where the controlled touch should send the ball --
    # biased toward continuing the incoming direction (most likely next-pass
    # direction); a fraction of episodes get a fully random target instead, so
    # the policy generalizes rather than only ever learning "let it through"
    incoming_angle = torch.atan2(direction[:, 1], direction[:, 0])
    biased_angle = incoming_angle + (torch.rand(n, device=device) * 2 - 1) * target_dir_spread
    random_angle = torch.rand(n, device=device) * 2 * torch.pi
    use_random = torch.rand(n, device=device) < target_dir_random_fraction
    target_angle = torch.where(use_random, random_angle, biased_angle)
    target_dir = torch.stack([torch.cos(target_angle), torch.sin(target_angle)], dim=-1)

    if not hasattr(env, "_ball_target_dir") or env._ball_target_dir.shape[0] != env.num_envs:
        env._ball_target_dir = torch.zeros(env.num_envs, 2, device=device)
    env._ball_target_dir[env_ids] = target_dir
