# Copyright (c) 2026 - K1 Ball Control (RL, robot-soccer first touch)
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the K1 ball-control task.

Two skills at once: (1) hold a standing balance (same posture/effort shaping
style as k1_walk_htwk's stand-still mode) and (2) kill the ball's velocity on
contact - either a dead stop or a slow roll back the way it came (a soft
"first touch"), mirroring a real football trap.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
import isaaclab.utils.math as math_utils

FEET_BODIES = ["left_foot_link", "right_foot_link"]
SHANK_BODIES = ["Left_Shank", "Right_Shank"]  # knee-to-ankle segment


def _robot(env) -> Articulation:
    return env.scene["robot"]


def _ball(env) -> RigidObject:
    return env.scene["ball"]


def _draw_target_dir_arrow(env) -> None:
    """Debug-vis: a red arrow anchored AT THE BALL, pointing along its commanded
    target direction (`_ball_target_dir`, world-frame, set per-episode by
    reset_robot_and_ball) - "which way the ball wants to go next"."""
    if not hasattr(env, "_ball_target_dir"):
        return
    if not hasattr(env, "_target_dir_visualizer"):
        cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/BallTargetDir")
        cfg.markers["arrow"].scale = (1.25, 0.05, 0.05)  # long and thin - exaggerated length, not girth
        env._target_dir_visualizer = VisualizationMarkers(cfg)

    ball_pos = _ball(env).data.root_pos_w.clone()
    ball_pos[:, 2] += 0.05
    heading = torch.atan2(env._ball_target_dir[:, 1], env._ball_target_dir[:, 0])
    zeros = torch.zeros_like(heading)
    quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading)
    env._target_dir_visualizer.visualize(ball_pos, quat)


# --------------------------------------------------------------------------- #
# Ball control (the goal)
# --------------------------------------------------------------------------- #
def ball_speed_penalty(env: ManagerBasedRLEnv, sigma: float = 0.4) -> torch.Tensor:
    """Dense: exp(-speed^2/sigma^2) peaks at 1 when the ball is fully stopped.
    Direction-agnostic, so it already gives partial credit for ANY slow ball -
    covers both the "dead stop" and "slow roll back" outcomes, since both are
    slow. Near-zero while the ball is still crossing the pitch at launch speed."""
    speed = torch.norm(_ball(env).data.root_lin_vel_w[:, :2], dim=-1)
    return torch.exp(-torch.square(speed) / (sigma**2))


def ball_target_direction_bonus(env: ManagerBasedRLEnv, sigma: float = 0.6) -> torch.Tensor:
    """On top of ball_speed_penalty: extra credit when the ball's (slow) residual
    velocity is aligned with the commanded target direction `_ball_target_dir`
    (set per-episode by reset_robot_and_ball, biased toward the incoming
    direction - i.e. "continue the way it came" is the most likely target, with
    some fully-random episodes for generalization). This is the DOMINANT shaping
    for where the touch sends the ball - a controlled send toward where the next
    action should go, not just any slow ball."""
    if not hasattr(env, "_ball_target_dir"):
        return torch.zeros(env.num_envs, device=env.device)
    _draw_target_dir_arrow(env)
    vel = _ball(env).data.root_lin_vel_w[:, :2]
    speed = torch.norm(vel, dim=-1)
    vel_dir = vel / speed.clamp(min=1.0e-6).unsqueeze(-1)
    aligned = torch.clamp(torch.sum(vel_dir * env._ball_target_dir, dim=-1), min=0.0)
    return aligned * torch.exp(-torch.square(speed) / (sigma**2))


def ball_target_position_bonus(env: ManagerBasedRLEnv, target_dist: float = 0.4, sigma: float = 0.15) -> torch.Tensor:
    """Secondary to ball_target_direction_bonus (lower weight by design - the
    direction of the touch matters more than landing on an exact spot): reward
    the ball coming to REST near a point `target_dist` from the robot, along
    the same commanded target direction."""
    if not hasattr(env, "_ball_target_dir"):
        return torch.zeros(env.num_envs, device=env.device)
    ball_xy = _ball(env).data.root_pos_w[:, :2]
    robot_xy = _robot(env).data.root_pos_w[:, :2]
    target_xy = robot_xy + env._ball_target_dir * target_dist
    dist_err = torch.norm(ball_xy - target_xy, dim=-1)
    speed = torch.norm(_ball(env).data.root_lin_vel_w[:, :2], dim=-1)
    return torch.exp(-torch.square(dist_err) / (sigma**2)) * torch.exp(-torch.square(speed) / (0.4**2))


def ball_overshoot_penalty(env: ManagerBasedRLEnv, max_overshoot: float = 1.0) -> torch.Tensor:
    """Penalize the ball rolling more than `max_overshoot` past the robot's
    current position, measured along the direction it was originally travelling
    - a clean miss, the robot let it go straight through. Zero while the ball is
    still in front of / near the robot; grows linearly once it's gotten away.
    A ball that bounces BACK past the robot doesn't trigger this (that's the
    opposite direction along `_ball_incoming_dir` - a separate concern from the
    target-direction bonuses, which judge the touch on its own terms)."""
    if not hasattr(env, "_ball_incoming_dir"):
        return torch.zeros(env.num_envs, device=env.device)
    ball_xy = _ball(env).data.root_pos_w[:, :2]
    robot_xy = _robot(env).data.root_pos_w[:, :2]
    overshoot = torch.sum((ball_xy - robot_xy) * env._ball_incoming_dir, dim=-1)
    return torch.clamp(overshoot - max_overshoot, min=0.0)


def inside_foot_contact(env: ManagerBasedRLEnv, contact_dist: float = 0.15) -> torch.Tensor:
    """Reward the ball being near a foot AND on that foot's INSIDE (medial) side -
    a real football "inside of the foot" first touch, not the shin or the
    outside of the foot. Proximity-based proxy (ball-center-to-foot-center
    distance), not a true contact-patch sensor.

    Sign-convention-free: "inside" is decided purely relative to each foot's own
    lateral offset from the robot's midline (y=0 in the base frame) - the ball
    counts as inside if it sits BETWEEN that foot and the midline, whichever
    foot that is and whatever the URDF's left/right sign convention turns out
    to be, so there's no left/right sign to get wrong here."""
    asset = _robot(env)
    ball_pos = _ball(env).data.root_pos_w
    base_pos = asset.data.root_pos_w
    base_quat = asset.data.root_quat_w

    def local_y(world_pos: torch.Tensor) -> torch.Tensor:
        rel_b = math_utils.quat_apply_inverse(base_quat, world_pos - base_pos)
        return rel_b[:, 1]

    lf_idx = asset.find_bodies(FEET_BODIES[0])[0][0]
    rf_idx = asset.find_bodies(FEET_BODIES[1])[0][0]
    lf_pos = asset.data.body_pos_w[:, lf_idx]
    rf_pos = asset.data.body_pos_w[:, rf_idx]

    ball_y = local_y(ball_pos)
    lf_y = local_y(lf_pos)
    rf_y = local_y(rf_pos)

    dist_l = torch.norm(ball_pos - lf_pos, dim=-1)
    dist_r = torch.norm(ball_pos - rf_pos, dim=-1)

    inside_l = (dist_l < contact_dist) & ((ball_y - lf_y) * lf_y < 0)
    inside_r = (dist_r < contact_dist) & ((ball_y - rf_y) * rf_y < 0)
    return (inside_l | inside_r).float()


def lifted_touching_foot(
    env: ManagerBasedRLEnv, contact_dist: float = 0.15, target_lift: float = 0.05, sigma: float = 0.03
) -> torch.Tensor:
    """Reward the ball-touching foot being lifted ABOVE the other (supporting)
    foot - so weight stays on the grounded foot and the touching foot is free
    (not weight-bearing) for a soft, controlled touch rather than stomping the
    ball with full load on it. Self-referential (touching foot's height
    relative to the OTHER foot, not an assumed absolute ground height), so it's
    robust regardless of the exact foot-geometry baseline."""
    asset = _robot(env)
    ball_pos = _ball(env).data.root_pos_w
    lf_idx = asset.find_bodies(FEET_BODIES[0])[0][0]
    rf_idx = asset.find_bodies(FEET_BODIES[1])[0][0]
    lf_pos = asset.data.body_pos_w[:, lf_idx]
    rf_pos = asset.data.body_pos_w[:, rf_idx]

    dist_l = torch.norm(ball_pos - lf_pos, dim=-1)
    dist_r = torch.norm(ball_pos - rf_pos, dim=-1)
    near_l = dist_l < contact_dist
    near_r = dist_r < contact_dist

    lift_l = torch.exp(-torch.square((lf_pos[:, 2] - rf_pos[:, 2]) - target_lift) / (sigma**2)) * near_l.float()
    lift_r = torch.exp(-torch.square((rf_pos[:, 2] - lf_pos[:, 2]) - target_lift) / (sigma**2)) * near_r.float()
    count = near_l.float() + near_r.float()
    return (lift_l + lift_r) / count.clamp(min=1.0)


def contact_shin_uprightness(env: ManagerBasedRLEnv, contact_dist: float = 0.15) -> torch.Tensor:
    """For whichever leg's foot is currently near the ball, reward that leg's
    SHIN being vertical rather than angled - discourages the wide-stance,
    bent-knee "block it with an angled shin" technique in favor of reaching
    with a straighter leg. Scaled 0 (horizontal) to 1 (perfectly vertical).
    The other (non-contacting) leg's angle doesn't matter here.

    Geometric, not orientation-based, so there's no local-axis sign to get
    wrong: verticality = |z-drop from knee to ankle| / |knee-to-ankle length|,
    using the Shank and foot body WORLD positions as knee/ankle proxies."""
    asset = _robot(env)
    ball_pos = _ball(env).data.root_pos_w

    lf_idx = asset.find_bodies(FEET_BODIES[0])[0][0]
    rf_idx = asset.find_bodies(FEET_BODIES[1])[0][0]
    lf_pos = asset.data.body_pos_w[:, lf_idx]
    rf_pos = asset.data.body_pos_w[:, rf_idx]

    dist_l = torch.norm(ball_pos - lf_pos, dim=-1)
    dist_r = torch.norm(ball_pos - rf_pos, dim=-1)
    near_l = dist_l < contact_dist
    near_r = dist_r < contact_dist

    ls_idx = asset.find_bodies(SHANK_BODIES[0])[0][0]
    rs_idx = asset.find_bodies(SHANK_BODIES[1])[0][0]
    ls_pos = asset.data.body_pos_w[:, ls_idx]
    rs_pos = asset.data.body_pos_w[:, rs_idx]

    def verticality(shank_pos: torch.Tensor, foot_pos: torch.Tensor) -> torch.Tensor:
        vec = foot_pos - shank_pos
        return torch.abs(vec[:, 2]) / (torch.norm(vec, dim=-1) + 1.0e-6)

    vert_l = verticality(ls_pos, lf_pos) * near_l.float()
    vert_r = verticality(rs_pos, rf_pos) * near_r.float()
    count = near_l.float() + near_r.float()
    return (vert_l + vert_r) / count.clamp(min=1.0)


# --------------------------------------------------------------------------- #
# Standing / balance
# --------------------------------------------------------------------------- #
def base_height(
    env: ManagerBasedRLEnv, target_height: float = 0.55, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)


def lin_vel_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=-1)


# --------------------------------------------------------------------------- #
# Effort / smoothness
# --------------------------------------------------------------------------- #
def torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=-1)


def dof_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=-1)


def action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.prev_action - env.action_manager.action), dim=-1)
