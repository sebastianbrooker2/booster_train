# Copyright (c) 2026 - K1 Getup Environment for RoboCup
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from booster_train.assets.robots.booster import BOOSTER_K1_CFG
import booster_train.tasks.manager_based.locomotion.k1_walk.rewards as custom_rewards


@configclass
class K1GetupRewards(RewardsCfg):
    # PRIMARY: linear head height — higher is always better
    head_height = RewTerm(
        func=custom_rewards.head_height_linear,
        weight=50.0,
        params={},
    )
    # Torso rotating — pitch magnitude regardless of direction
    torso_pitch_up = RewTerm(
        func=custom_rewards.torso_pitch_up_reward,
        weight=15.0,
        params={},
    )
    # Penalise yaw in robot LOCAL frame — spinning around own vertical axis
    body_yaw_penalty = RewTerm(
        func=custom_rewards.penalty_body_yaw_local,
        weight=20.0,
        params={},
    )
    # Disable everything else
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    lin_vel_z_l2 = None
    feet_air_time = None
    feet_slide = None
    undesired_contacts = None
    termination_penalty = None
    flat_orientation_l2 = None
    action_rate_penalty = None
    standing_stable = None
    feet_ground = None
    joint_deviation_hip = None
    joint_deviation_head = None
    joint_deviation_arms = None


@configclass
class K1GetupEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: K1GetupRewards = K1GetupRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # Short episodes — 4 seconds
        self.episode_length_s = 4.0

        # No velocity commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Spawn in any orientation — full roll range
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (1.57, 1.57),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.30)

        # No pushes, no external forces
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None

        # Allow explosive movement
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0

        # Don't terminate on trunk contact
        self.terminations.base_contact = None


@configclass
class K1GetupEnvCfg_PLAY(K1GetupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 15.0
        self.observations.policy.enable_corruption = False
