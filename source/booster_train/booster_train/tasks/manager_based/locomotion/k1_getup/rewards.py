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


def _base_height(env) -> torch.Tensor:
    """World-Z of the base/pelvis. K1's verified standing base height is 0.57 m
    (from the asset init_state) — so this is the one climb metric grounded in the
    real robot geometry, not a guessed head height."""
    return _robot(env).data.root_pos_w[:, 2]


def _upright_factor(env) -> torch.Tensor:
    """0 when horizontal (lying / downward-dog), 1 when fully upright."""
    return torch.clamp(-_robot(env).data.projected_gravity_b[:, 2], min=0.0, max=1.0)


def _is_standing(env, stand_height: float) -> torch.Tensor:
    """Upright AND base near standing height AND slow -> standing. Uses BASE height
    (verified 0.57 m), gated by uprightness, so a butt-in-air downward-dog (pelvis
    high but body horizontal) does NOT count as standing."""
    asset = _robot(env)
    upright = asset.data.projected_gravity_b[:, 2] < -0.9      # torso z ~ world up
    high = asset.data.root_pos_w[:, 2] > stand_height * 0.88   # ~0.50 m of 0.57
    slow = torch.norm(asset.data.root_lin_vel_w, dim=-1) < 0.6
    return upright & high & slow


# --------------------------------------------------------------------------- #
# Task: stand up (the goal)
# --------------------------------------------------------------------------- #
def standing_bonus(env: ManagerBasedRLEnv, stand_height: float = 0.57) -> torch.Tensor:
    """Big sparse reward for being in the standing state."""
    return _is_standing(env, stand_height).float()


def head_height(env: ManagerBasedRLEnv, max_diff: float = 0.7) -> torch.Tensor:
    """Dense head-ABOVE-feet height (HoST-style), capped at `max_diff`. This is LOW
    in a downward-dog (head near the ground between the feet) and HIGH only when the
    body is genuinely upright — so it pulls the head up without rewarding the park,
    and needs no guessed absolute head height."""
    asset = _robot(env)
    head_z = _head_height(env)
    lf = asset.find_bodies(FEET_BODIES[0])[0][0]
    rf = asset.find_bodies(FEET_BODIES[1])[0][0]
    feet_z = 0.5 * (asset.data.body_pos_w[:, lf, 2] + asset.data.body_pos_w[:, rf, 2])
    return torch.clamp(head_z - feet_z, min=0.0, max=max_diff)


def upright_climb(env: ManagerBasedRLEnv, stand_height: float = 0.57) -> torch.Tensor:
    """PRIMARY dense climb signal: base height (toward standing 0.57 m) GATED by
    uprightness. = clamp(base_z, 0, 0.57) * upright_factor. Crucially this is ~0 in a
    butt-in-air downward-dog (pelvis high but body horizontal -> upright_factor ~ 0),
    which is what killed the previous run. Reward only grows when the robot rises AND
    becomes upright — exactly the get-up we want."""
    base_h = torch.clamp(_base_height(env), min=0.0, max=stand_height)
    return base_h * _upright_factor(env)


def upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward gravity-aligned torso (1 when fully upright, 0 when horizontal)."""
    return torch.clamp(-_robot(env).data.projected_gravity_b[:, 2], min=0.0)


# --------------------------------------------------------------------------- #
# FAST: time pressure + continuous (one-motion) progress
# --------------------------------------------------------------------------- #
def not_standing_time_penalty(env: ManagerBasedRLEnv, stand_height: float = 0.57) -> torch.Tensor:
    """1.0 every step it is NOT yet standing -> with a negative, ramped weight this
    forces it to minimise time-to-stand (explosive, athletic rise)."""
    return (~_is_standing(env, stand_height)).float()


def rising_velocity(env: ManagerBasedRLEnv, max_vel: float = 1.0) -> torch.Tensor:
    """Reward upward head velocity -> continuous progress = one fluid motion,
    not stop-start (no reward for parking in a stable intermediate). CAPPED at
    `max_vel`: without the cap an ever-faster fling farms this reward (a cheat);
    capping rewards steady progress, not violent launches."""
    asset = _robot(env)
    idx = asset.find_bodies(HEAD_BODY)[0][0]
    head_vel_z = asset.data.body_lin_vel_w[:, idx, 2]
    return torch.clamp(head_vel_z, min=0.0, max=max_vel)


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


def arms_at_side(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_height: float = 0.57) -> torch.Tensor:
    """When standing, reward arm joints near neutral so the get-up ENDS with the arms down
    at the sides (not flailing / overhead). Only active once standing, so arms are still
    free to push off during the rise. Target = 0 for shoulder+elbow joints; verify 'down at
    sides' in the viewer and adjust the target if 0 isn't arms-down for this URDF."""
    asset = _robot(env)
    standing = _is_standing(env, stand_height).float()
    jp = asset.data.joint_pos[:, asset_cfg.joint_ids]
    err = torch.sum(torch.square(jp), dim=1)
    return torch.exp(-err) * standing


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
def standing_stability(env: ManagerBasedRLEnv, stand_height: float = 0.57) -> torch.Tensor:
    """When standing, reward low base velocity (hold the pose, don't topple)."""
    asset = _robot(env)
    standing = _is_standing(env, stand_height).float()
    vel = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=-1) + \
        torch.sum(torch.square(asset.data.root_ang_vel_w), dim=-1)
    return torch.exp(-1.0 * vel) * standing


def feet_under_body(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward feet below the trunk (feet getting underneath), GATED by uprightness so a
    downward-dog (trunk also above the feet, but body horizontal) doesn't earn it."""
    asset = _robot(env)
    trunk_idx = asset.find_bodies(TRUNK_BODY)[0][0]
    trunk_h = asset.data.body_pos_w[:, trunk_idx, 2]
    lf = asset.find_bodies(FEET_BODIES[0])[0][0]
    rf = asset.find_bodies(FEET_BODIES[1])[0][0]
    feet_h = 0.5 * (asset.data.body_pos_w[:, lf, 2] + asset.data.body_pos_w[:, rf, 2])
    return torch.clamp(trunk_h - feet_h, min=0.0) * _upright_factor(env)


# --------------------------------------------------------------------------- #
# Assist force (applied here as a side-effect every step; level set by curriculum)
# --------------------------------------------------------------------------- #
def _ensure_assist_state(env, initial: float) -> None:
    """Lazily create the PER-ENV assist state (force magnitude + per-episode peak base
    height). Per-env (shape [N]) is the whole point: each robot weans off its own assist
    when IT succeeds, so the population transitions gradually from assisted to unassisted."""
    if not hasattr(env, "_assist_force") or env._assist_force.shape[0] != env.num_envs:
        env._assist_force = torch.full((env.num_envs,), float(initial), device=env.device)
        env._epi_max_base = torch.zeros(env.num_envs, device=env.device)


def apply_assist_force(
    env: ManagerBasedRLEnv, body_names: list[str] = (TRUNK_BODY,), initial: float = 120.0
) -> torch.Tensor:
    """PER-ENV world-up assist on the TRUNK (not the hips — pulling the hips up from a
    face-down pose just lifts the butt into a downward-dog; pulling the TORSO up rotates
    the body toward upright, HoST-style). Magnitude = env._assist_force[env] (a tensor,
    weaned to 0 per-env by assist_force_curriculum), so the FINAL policy stands with NO
    assist -> not a cheat. Also tracks each episode's peak base height for the curriculum.
    Returns 0 (no reward)."""
    from isaaclab.utils.math import quat_apply_inverse

    asset = _robot(env)
    _ensure_assist_state(env, initial)

    # track the episode's peak base height -> the curriculum decays force per-env on success
    env._epi_max_base = torch.maximum(env._epi_max_base, _base_height(env))

    body_ids = [asset.find_bodies(n)[0][0] for n in body_names]
    nb = len(body_ids)
    torques = torch.zeros(env.num_envs, nb, 3, device=env.device)
    world_up = torch.zeros(env.num_envs, nb, 3, device=env.device)
    world_up[:, :, 2] = (env._assist_force / nb).unsqueeze(-1)        # (N, nb): per-env, split across bodies
    quats = asset.data.body_quat_w[:, body_ids]                       # (N, nb, 4)
    local_force = quat_apply_inverse(quats, world_up)                # (N, nb, 3) world-up in each body frame
    asset.set_external_force_and_torque(local_force, torques, body_ids=body_ids)
    return torch.zeros(env.num_envs, device=env.device)
