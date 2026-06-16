# Copyright (c) 2026 - K1 Get-up (curriculum RL, discovery-based)
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the K1 get-up task — built for a FAST rise from any posture.

A single policy gets up from supine / prone / side. The motion (e.g. a backward
roll from supine, a one-motion push-up from prone) is not scripted — it emerges
from: a big standing bonus, dense height shaping, time pressure (= fast), a
continuous-progress term (= one fluid motion), a light supine backward-roll
nudge, and motor-friendly regularization.

Body-frame projected gravity tells the policy (and these rewards) the posture:
  standing  -> proj_grav_b ~ ( 0, 0, -1)
  supine    -> proj_grav_b ~ (-1, 0,  0)   (face up)
  prone     -> proj_grav_b ~ (+1, 0,  0)   (face down)
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

HEAD_BODY = "Head_2"
TRUNK_BODY = "Trunk"
HIP_BODIES = ["Left_Hip_Pitch", "Right_Hip_Pitch"]   # pelvis/hip region — must rise first
FEET_BODIES = ["left_foot_link", "right_foot_link"]


def _robot(env) -> Articulation:
    return env.scene["robot"]


def _head_height(env) -> torch.Tensor:
    asset = _robot(env)
    idx = asset.find_bodies(HEAD_BODY)[0][0]
    return asset.data.body_pos_w[:, idx, 2]


def _is_standing(env, target_height: float) -> torch.Tensor:
    """Upright + head near target height + feet down + low speed -> standing."""
    asset = _robot(env)
    head_h = _head_height(env)
    upright = asset.data.projected_gravity_b[:, 2] < -0.9   # body z ~ world up
    high = head_h > target_height * 0.9
    slow = torch.norm(asset.data.root_lin_vel_w, dim=-1) < 0.6
    return upright & high & slow


# --------------------------------------------------------------------------- #
# Task: stand up (the goal)
# --------------------------------------------------------------------------- #
def standing_bonus(env: ManagerBasedRLEnv, target_height: float = 0.72) -> torch.Tensor:
    """Big sparse reward for being in the standing state."""
    return _is_standing(env, target_height).float()


def head_height(env: ManagerBasedRLEnv, min_height: float = 0.5) -> torch.Tensor:
    """Head height ABOVE a threshold (default 0.5 m). Below it -> 0, so there's no
    reward for flicking the head up while still down; it only rewards finishing the
    stand once the body is genuinely up."""
    return torch.clamp(_head_height(env) - min_height, min=0.0)


def hip_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense: mean hip/pelvis height — the PRIMARY climb signal. From prone or
    supine the hips must come off the ground first; rewarding this (not head
    height) avoids the 'fling the head up' exploit."""
    asset = _robot(env)
    lh = asset.find_bodies(HIP_BODIES[0])[0][0]
    rh = asset.find_bodies(HIP_BODIES[1])[0][0]
    return 0.5 * (asset.data.body_pos_w[:, lh, 2] + asset.data.body_pos_w[:, rh, 2])


def upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward gravity-aligned torso (1 when fully upright, 0 when horizontal)."""
    return torch.clamp(-_robot(env).data.projected_gravity_b[:, 2], min=0.0)


# --------------------------------------------------------------------------- #
# FAST: time pressure + continuous (one-motion) progress
# --------------------------------------------------------------------------- #
def not_standing_time_penalty(env: ManagerBasedRLEnv, target_height: float = 0.72) -> torch.Tensor:
    """1.0 every step it is NOT yet standing -> with a negative, ramped weight this
    forces it to minimise time-to-stand (explosive, athletic rise)."""
    return (~_is_standing(env, target_height)).float()


def rising_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward upward head velocity -> continuous progress = one fluid motion,
    not stop-start (no reward for parking in a stable intermediate)."""
    asset = _robot(env)
    idx = asset.find_bodies(HEAD_BODY)[0][0]
    head_vel_z = asset.data.body_lin_vel_w[:, idx, 2]
    return torch.clamp(head_vel_z, min=0.0)


# --------------------------------------------------------------------------- #
# Supine backward-roll nudge (light; the roll mostly emerges on its own)
# --------------------------------------------------------------------------- #
def hip_extension_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalise BACKWARD hip bending (extension = positive Hip_Pitch). Pushes it to
    FOLD instead — a tuck from supine, a pike from prone — not arch into a bridge.
    (Sign assumes positive Hip_Pitch = extension; verify in the viewer and flip if not.)"""
    asset = _robot(env)
    hip_pitch = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(hip_pitch, min=0.0), dim=1)


def hip_flexion_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward hip FLEXION (knees toward chest / in front of the body = negative
    Hip_Pitch). Prefers folding the legs forward (tuck / pike) over extending them
    behind. (Sign assumes negative Hip_Pitch = flexion; verify and flip if needed.)"""
    asset = _robot(env)
    hip_pitch = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(-hip_pitch, min=0.0), dim=1)


def supine_hips_over_head(env: ManagerBasedRLEnv) -> torch.Tensor:
    """When on its back, reward the hips being HIGHER than the head — i.e. throwing
    the pelvis up and over the head (the shoulder-stand / rock-back that starts a
    backward roll). Gated to supine so it doesn't affect prone get-ups."""
    asset = _robot(env)
    on_back = (asset.data.projected_gravity_b[:, 0] < -0.5).float()   # face up
    lh = asset.find_bodies(HIP_BODIES[0])[0][0]
    rh = asset.find_bodies(HIP_BODIES[1])[0][0]
    hip_z = 0.5 * (asset.data.body_pos_w[:, lh, 2] + asset.data.body_pos_w[:, rh, 2])
    return torch.clamp(hip_z - _head_height(env), min=0.0) * on_back


def supine_backward_roll(env: ManagerBasedRLEnv) -> torch.Tensor:
    """When on its back (face up), reward pitch angular velocity that rocks it back
    toward standing. NOTE: sign assumes rolling 'back over the shoulders'; verify in
    the viewer and flip the weight sign if it rolls the wrong way."""
    asset = _robot(env)
    pg = asset.data.projected_gravity_b
    on_back = (pg[:, 0] < -0.5).float()           # face pointing up
    pitch_rate = asset.data.root_ang_vel_b[:, 1]  # body-y angular velocity
    return torch.clamp(pitch_rate, min=0.0) * on_back


# --------------------------------------------------------------------------- #
# Stability once up
# --------------------------------------------------------------------------- #
def standing_stability(env: ManagerBasedRLEnv, target_height: float = 0.72) -> torch.Tensor:
    """When standing, reward low base velocity (hold the pose, don't topple)."""
    asset = _robot(env)
    standing = _is_standing(env, target_height).float()
    vel = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=-1) + \
        torch.sum(torch.square(asset.data.root_ang_vel_w), dim=-1)
    return torch.exp(-1.0 * vel) * standing


def feet_under_body(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward feet being below the trunk (encourages getting feet underneath)."""
    asset = _robot(env)
    trunk_idx = asset.find_bodies(TRUNK_BODY)[0][0]
    trunk_h = asset.data.body_pos_w[:, trunk_idx, 2]
    lf = asset.find_bodies(FEET_BODIES[0])[0][0]
    rf = asset.find_bodies(FEET_BODIES[1])[0][0]
    feet_h = 0.5 * (asset.data.body_pos_w[:, lf, 2] + asset.data.body_pos_w[:, rf, 2])
    return torch.clamp(trunk_h - feet_h, min=0.0)


# --------------------------------------------------------------------------- #
# Assist force (applied here as a side-effect every step; level set by curriculum)
# --------------------------------------------------------------------------- #
def apply_assist_force(env: ManagerBasedRLEnv, body_names: list[str] = HIP_BODIES) -> torch.Tensor:
    """Apply a WORLD-up assist force on the HIPS (split across both hip bodies), so the
    pelvis is lifted first — magnitude = env._assist_force (set by the curriculum).
    Returns 0 (no direct reward)."""
    from isaaclab.utils.math import quat_apply_inverse

    asset = _robot(env)
    body_ids = [asset.find_bodies(n)[0][0] for n in body_names]
    nb = len(body_ids)
    torques = torch.zeros(env.num_envs, nb, 3, device=env.device)
    force_mag = float(getattr(env, "_assist_force", 0.0))
    if force_mag < 1.0:
        asset.set_external_force_and_torque(
            torch.zeros(env.num_envs, nb, 3, device=env.device), torques, body_ids=body_ids
        )
        return torch.zeros(env.num_envs, device=env.device)

    quats = asset.data.body_quat_w[:, body_ids]                       # (N, nb, 4)
    world_up = torch.zeros(env.num_envs, nb, 3, device=env.device)
    world_up[:, :, 2] = force_mag / nb                               # split evenly across the hips
    local_force = quat_apply_inverse(quats, world_up)                # (N, nb, 3) world-up in each body frame
    asset.set_external_force_and_torque(local_force, torques, body_ids=body_ids)
    return torch.zeros(env.num_envs, device=env.device)
