# Copyright (c) 2026 - K1 Walk Environment for RoboCup
# SPDX-License-Identifier: BSD-3-Clause

"""K1 omnidirectional locomotion for RoboCup soccer.

Cost function is seeded from the Unitree G1 flat velocity-tracking task
(isaaclab_tasks .../velocity/config/g1/flat_env_cfg.py) and re-mapped onto the
K1 joint/body names. Goal: a fast, agile policy that can walk forward (biased),
backward, strafe, and turn quickly on the spot — all in a single policy.

Notes:
  * Flat terrain, no pushes, no mass/COM randomization — clean signal to start.
  * Servo delay is kept (it lives in BOOSTER_K1_CFG's DelayedPD actuators).
  * Turn-on-the-spot is trained inside the same policy: lin/ang velocity are
    sampled independently, so the command distribution already contains
    near-zero linear velocity with large yaw rate (= turn in place).
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from booster_train.assets.robots.booster import BOOSTER_K1_CFG
import booster_train.tasks.manager_based.locomotion.k1_walk.rewards as custom_rewards


# K1 foot bodies and non-foot bodies (anything below that touching ground = fall)
FEET_BODIES = ["left_foot_link", "right_foot_link"]
NON_FOOT_BODIES = [
    "Trunk",
    "Left_Hip_Pitch", "Right_Hip_Pitch",
    "Left_Shank", "Right_Shank",
    "Left_Arm_1", "Right_Arm_1",
    "Left_Arm_2", "Right_Arm_2",
    "Left_Arm_3", "Right_Arm_3",
    "Head_1", "Head_2",
]


@configclass
class K1WalkRewards(RewardsCfg):
    """G1-flat cost function re-mapped to K1."""

    # -- task: track commanded velocity (yaw-frame lin vel + world yaw rate) -----
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # -- gait --------------------------------------------------------------------
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES),
        },
    )

    # -- gait shaping ------------------------------------------------------------
    # DISABLED: the swing-knee-flexion reward was the wrong direction — it shaped
    # the gait pattern while the robot still tracked velocity fine on its toes.
    # Stability pressure (uneven terrain + friction randomization) is the fix
    # instead. Left here, commented, in case we want to revisit.
    # swing_knee_flexion = RewTerm(
    #     func=custom_rewards.swing_knee_flexion,
    #     weight=0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["Left_Knee_Pitch", "Right_Knee_Pitch"]),
    #     },
    # )

    # -- posture / joint regularisation ------------------------------------------
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_Ankle_Pitch", ".*_Ankle_Roll",
        ])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_Hip_Yaw", ".*_Hip_Roll",
        ])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_Shoulder_Pitch", ".*_Shoulder_Roll",
            ".*_Elbow_Pitch", ".*_Elbow_Yaw",
        ])},
    )
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "AAHead_yaw", "Head_pitch",
        ])},
    )

    # Disable terms that don't apply / aren't wanted at the start
    undesired_contacts = None


@configclass
class K1WalkEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: K1WalkRewards = K1WalkRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---- Flat terrain ------------------------------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # ---- Commands: forward-biased omnidirectional + fast turn --------------
        # Independent sampling means the policy also sees near-zero linear vel with
        # large yaw rate -> "turn on the spot" trained within the same policy.
        # heading_command=False so the yaw rate is commanded directly (sustained
        # spinning) rather than derived from a heading set-point.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 1.2)   # forward bias
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)   # strafe
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)   # fast turn

        # ---- Reward weights (G1 flat values) -----------------------------------
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch"]
        )
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch"]
        )

        # ---- Terminate on any non-foot body touching the ground ----------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = NON_FOOT_BODIES

        # ---- Events: no external forces, no domain randomization ---------------
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        # ---- Stable bent-knee standing pose ------------------------------------
        self.scene.robot.init_state.joint_pos["Left_Hip_Pitch"] = -0.2
        self.scene.robot.init_state.joint_pos["Right_Hip_Pitch"] = -0.2
        self.scene.robot.init_state.joint_pos["Left_Knee_Pitch"] = 0.4
        self.scene.robot.init_state.joint_pos["Right_Knee_Pitch"] = 0.4
        self.scene.robot.init_state.joint_pos["Left_Ankle_Pitch"] = -0.2
        self.scene.robot.init_state.joint_pos["Right_Ankle_Pitch"] = -0.2


@configclass
class K1WalkEnvCfg_PLAY(K1WalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # smaller, non-curriculum terrain grid for play
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
