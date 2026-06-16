# Copyright (c) 2026 - K1 ParameterWalk (HTWK port)
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions ported from HTWK Gym's K1 ParameterWalk (envs/K1/parameter_walk.py).

Faithful re-implementation in IsaacLab's manager-based API. Velocity tracking,
posture (base height + orientation-to-target), energy, smoothness, foot orientation,
foot-yaw / foot-offset tracking, and a gait-phase swing reward.

All "dof_*" sums are scoped to the leg joints via `asset_cfg` (HTWK is legs-only).
The parametric targets and gait phase are read from the ParameterWalkCommand term.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat

CMD = "base_velocity"  # registered name of the ParameterWalkCommand term

# command index layout
C_VX, C_VY, C_WZ, C_FREQ = 0, 1, 2, 3
C_FYAW_L, C_FYAW_R, C_PITCH, C_ROLL, C_OFF_X, C_OFF_Y = 4, 5, 6, 7, 8, 9


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def _cmd(env: ManagerBasedRLEnv):
    return env.command_manager.get_term(CMD)


def _foot_euler(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """roll, pitch, yaw of each foot (wrapped to [-pi, pi]); shape (N, n_feet)."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (N, nf, 4)
    n, nf = quat.shape[0], quat.shape[1]
    roll, pitch, yaw = euler_xyz_from_quat(quat.reshape(-1, 4))
    roll = _wrap_pi(roll).reshape(n, nf)
    pitch = _wrap_pi(pitch).reshape(n, nf)
    yaw = _wrap_pi(yaw).reshape(n, nf)
    return roll, pitch, yaw


def _base_yaw(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    return yaw


def _feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids]  # (N, hist, nf, 3)
    return forces.norm(dim=-1).max(dim=1)[0] > threshold  # (N, nf) bool


# --------------------------------------------------------------------------- #
# Task tracking
# --------------------------------------------------------------------------- #
def survival(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def tracking_lin_vel_x(env: ManagerBasedRLEnv, sigma: float = 0.25,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err = _cmd(env).cmd[:, C_VX] - asset.data.root_lin_vel_b[:, 0]
    return torch.exp(-torch.square(err) / sigma)


def tracking_lin_vel_y(env: ManagerBasedRLEnv, sigma: float = 0.25,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err = _cmd(env).cmd[:, C_VY] - asset.data.root_lin_vel_b[:, 1]
    return torch.exp(-torch.square(err) / sigma)


def tracking_ang_vel(env: ManagerBasedRLEnv, sigma: float = 0.25,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err = _cmd(env).cmd[:, C_WZ] - asset.data.root_ang_vel_b[:, 2]
    return torch.exp(-torch.square(err) / sigma)


# --------------------------------------------------------------------------- #
# Posture
# --------------------------------------------------------------------------- #
def base_height(env: ManagerBasedRLEnv, target_height: float = 0.52,
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """roll/pitch error vs the commanded body_roll/body_pitch targets."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    roll, pitch = _wrap_pi(roll), _wrap_pi(pitch)
    cmd = _cmd(env).cmd
    roll_err = roll - cmd[:, C_ROLL]
    pitch_err = pitch - cmd[:, C_PITCH]
    return torch.square(roll_err) + torch.square(pitch_err)


# --------------------------------------------------------------------------- #
# Velocity penalties
# --------------------------------------------------------------------------- #
def lin_vel_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=-1)


# --------------------------------------------------------------------------- #
# Energy / effort (leg joints)
# --------------------------------------------------------------------------- #
def torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=-1)


def dof_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


def dof_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=-1)


def root_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # spatial acceleration of the root/trunk body (6-dim: lin + ang)
    body_acc = asset.data.body_acc_w[:, asset_cfg.body_ids]  # (N, 1, 6)
    return torch.sum(torch.square(body_acc), dim=(1, 2))


def torque_tiredness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    lim = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(tau / lim).clip(max=1.0), dim=-1)


def power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum((tau * vel).clip(min=0.0), dim=-1)


# --------------------------------------------------------------------------- #
# Smoothness / limits
# --------------------------------------------------------------------------- #
def action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.prev_action - env.action_manager.action), dim=-1)


def dof_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    jids = asset_cfg.joint_ids
    pos = asset.data.joint_pos[:, jids]
    lower = asset.data.soft_joint_pos_limits[:, jids, 0]
    upper = asset.data.soft_joint_pos_limits[:, jids, 1]
    return torch.sum(((pos < lower) | (pos > upper)).float(), dim=-1)


# --------------------------------------------------------------------------- #
# Feet: contact / orientation
# --------------------------------------------------------------------------- #
def feet_slip(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]  # (N, nf, 3)
    contact = _feet_contact(env, sensor_cfg).float()             # (N, nf)
    return torch.sum(torch.sum(torch.square(feet_vel), dim=-1) * contact, dim=-1)


def feet_roll(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    roll, _, _ = _foot_euler(env, asset_cfg)
    return torch.sum(torch.square(roll), dim=-1)


def feet_pitch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    _, pitch, _ = _foot_euler(env, asset_cfg)
    return torch.sum(torch.square(pitch), dim=-1)


def _feet_yaw_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """foot yaw relative to base yaw, ordered [L, R]."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, fyaw = _foot_euler(env, asset_cfg)            # (N, nf) ordered as body_ids (L,R)
    byaw = _base_yaw(env, asset).unsqueeze(-1)
    return _wrap_pi(fyaw - byaw)                        # (N, nf)


def feet_yaw_diff(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd = _cmd(env).cmd
    commanded = cmd[:, C_FYAW_R] - cmd[:, C_FYAW_L]
    fyaw = _feet_yaw_rel(env, asset_cfg)
    actual = fyaw[:, 1] - fyaw[:, 0]
    return torch.square(_wrap_pi(actual - commanded))


def feet_yaw_mean(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd = _cmd(env).cmd
    commanded = (cmd[:, C_FYAW_R] + cmd[:, C_FYAW_L]) * 0.5
    fyaw = _feet_yaw_rel(env, asset_cfg)
    actual = fyaw.mean(dim=-1)
    return torch.square(_wrap_pi(actual - commanded))


def foot_yaw_l(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    fyaw = _feet_yaw_rel(env, asset_cfg)
    return torch.square(fyaw[:, 0] - _cmd(env).cmd[:, C_FYAW_L])


def foot_yaw_r(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    fyaw = _feet_yaw_rel(env, asset_cfg)
    return torch.square(fyaw[:, 1] - _cmd(env).cmd[:, C_FYAW_R])


def _feet_offset(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """x/y offset between L and R feet expressed in the base yaw frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (N, 2, 3) [L, R]
    byaw = _base_yaw(env, asset)
    dx = feet_pos[:, 0, 0] - feet_pos[:, 1, 0]
    dy = feet_pos[:, 0, 1] - feet_pos[:, 1, 1]
    x_off = torch.cos(byaw) * dx + torch.sin(byaw) * dy
    y_off = -torch.sin(byaw) * dx + torch.cos(byaw) * dy
    return x_off, y_off


def _vel_scale(cmd_vel: torch.Tensor, max_vel: float) -> torch.Tensor:
    return torch.clamp((1.0 - torch.abs(cmd_vel) / max_vel) ** 2, min=0.0, max=1.0)


def feet_offset_x(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_vel: float = 1.0) -> torch.Tensor:
    x_off, _ = _feet_offset(env, asset_cfg)
    cmd = _cmd(env).cmd
    err = torch.clip(torch.abs(x_off - cmd[:, C_OFF_X]), min=0.0, max=0.1)
    return err * _vel_scale(cmd[:, C_VX], max_vel)


def feet_offset_y(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_vel: float = 1.0) -> torch.Tensor:
    _, y_off = _feet_offset(env, asset_cfg)
    cmd = _cmd(env).cmd
    err = torch.clip(torch.abs(y_off - cmd[:, C_OFF_Y]), min=0.0, max=0.1)
    return err * _vel_scale(cmd[:, C_VY], max_vel)


def feet_swing(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, swing_period: float = 0.2) -> torch.Tensor:
    """Reward the swing foot being airborne at the right point of the gait phase."""
    term = _cmd(env)
    gp = term.gait_process
    freq = term.gait_frequency
    contact = _feet_contact(env, sensor_cfg)  # (N, 2) [L, R]
    left_swing = (torch.abs(gp - 0.25) < 0.5 * swing_period) & (freq > 1e-8)
    right_swing = (torch.abs(gp - 0.75) < 0.5 * swing_period) & (freq > 1e-8)
    return (left_swing & ~contact[:, 0]).float() + (right_swing & ~contact[:, 1]).float()
