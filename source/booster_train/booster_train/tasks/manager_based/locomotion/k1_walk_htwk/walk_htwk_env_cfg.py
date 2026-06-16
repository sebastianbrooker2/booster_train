# Copyright (c) 2026 - K1 ParameterWalk (HTWK port)
# SPDX-License-Identifier: BSD-3-Clause

"""Faithful IsaacLab port of HTWK Gym's K1 ParameterWalk task.

Parametric, gait-clock-driven walk: 10-dim command (lin/ang vel, gait frequency,
foot yaw L/R, body pitch/roll targets, feet x/y offset targets) + a gait phase
clock, legs-only 12-DOF action, a 54-dim observation that is BLIND to linear
velocity, and the full HTWK reward set / weights.

We keep the Booster K1 actuator model (servo delay) rather than HTWK's PD gains,
so action scale and a few weights may need a tuning pass.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from booster_train.assets.robots.booster import BOOSTER_K1_CFG
import booster_train.tasks.manager_based.locomotion.k1_walk_htwk.rewards as R
import booster_train.tasks.manager_based.locomotion.k1_walk_htwk.curriculum as C
from booster_train.tasks.manager_based.locomotion.k1_walk_htwk.commands import ParameterWalkCommandCfg


# Mild terrain (gentle noise + shallow slopes, NO stairs). With max_init_terrain_level=0
# everyone starts on the flattest row; the gated curriculum only ramps difficulty once
# the speed curriculum is satisfied.
MILD_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(0.0, 0.06), noise_step=0.01, border_width=0.25
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
        ),
    },
)


# 6 joints/leg x 2 = 12 leg DOF
LEG_JOINTS = [
    ".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw",
    ".*_Knee_Pitch", ".*_Ankle_Pitch", ".*_Ankle_Roll",
]
# shoulders actuated but pinned early by a decaying penalty (curriculum), freed later
SHOULDER_JOINTS = [".*_Shoulder_Pitch", ".*_Shoulder_Roll"]
ACTION_JOINTS = LEG_JOINTS + SHOULDER_JOINTS  # 16 DOF (elbows/head stay locked at default)
FEET_BODIES = ["left_foot_link", "right_foot_link"]
# bodies that should never touch the ground (collision penalty)
COLLISION_BODIES = ["Trunk", ".*_Shank", ".*_Arm_.*", "Head_.*"]
LEGS = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
ACTUATED = SceneEntityCfg("robot", joint_names=ACTION_JOINTS)
FEET = SceneEntityCfg("robot", body_names=FEET_BODIES)
FEET_SENSOR = SceneEntityCfg("contact_forces", body_names=FEET_BODIES)


@configclass
class CommandsCfg:
    base_velocity = ParameterWalkCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        still_proportion=0.1,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    # legs + shoulders, position control (elbows/head stay at their default targets)
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ACTION_JOINTS, scale=0.8, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # order matches HTWK: proj_grav, ang_vel, command(10+phase=12), dof_pos, dof_vel, actions  -> 54
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": ACTUATED}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": ACTUATED}, scale=0.1, noise=Unoise(n_min=-0.15, n_max=0.15)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # asymmetric actor-critic: critic sees everything the actor does PLUS
        # privileged signals the actor is blind to (true linear velocity + height).
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ACTUATED})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ACTUATED}, scale=0.1)
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)   # privileged
        base_height = ObsTerm(func=mdp.base_pos_z)      # privileged

        def __post_init__(self):
            self.enable_corruption = False  # critic gets clean obs
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HTWKRewards:
    # -- task tracking -------------------------------------------------------
    survival = RewTerm(func=R.survival, weight=0.25)
    tracking_lin_vel_x = RewTerm(func=R.tracking_lin_vel_x, weight=2.0, params={"sigma": 0.25})
    tracking_lin_vel_y = RewTerm(func=R.tracking_lin_vel_y, weight=2.0, params={"sigma": 0.25})
    tracking_ang_vel = RewTerm(func=R.tracking_ang_vel, weight=1.5, params={"sigma": 0.25})
    # -- posture -------------------------------------------------------------
    # eased from -20 to let it lean/lower into a more dynamic, faster gait
    base_height = RewTerm(func=R.base_height, weight=-8.0, params={"target_height": 0.52})
    orientation = RewTerm(func=R.orientation, weight=-8.0)
    # -- velocity penalties --------------------------------------------------
    lin_vel_z = RewTerm(func=R.lin_vel_z, weight=-2.0)
    ang_vel_xy = RewTerm(func=R.ang_vel_xy, weight=-0.2)
    # -- energy / effort (leg joints) ---------------------------------------
    # effort/energy penalties slashed ~10x so it freely USES the actuators (torque + joint speed)
    torques = RewTerm(func=R.torques, weight=-3.0e-5, params={"asset_cfg": ACTUATED})
    torque_tiredness = RewTerm(func=R.torque_tiredness, weight=-1.0e-3, params={"asset_cfg": ACTUATED})
    power = RewTerm(func=R.power, weight=-3.0e-4, params={"asset_cfg": ACTUATED})
    dof_vel = RewTerm(func=R.dof_vel, weight=-2.0e-5, params={"asset_cfg": ACTUATED})
    dof_acc = RewTerm(func=R.dof_acc, weight=-1.0e-7, params={"asset_cfg": ACTUATED})
    root_acc = RewTerm(func=R.root_acc, weight=-1.0e-5, params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")})
    # -- smoothness / limits -------------------------------------------------
    # starts loose (move explosively, nail the speed) -> ramped up by the
    # action_rate_curriculum once velocity tracking is good (then it smooths out)
    action_rate = RewTerm(func=R.action_rate, weight=-0.1)
    dof_pos_limits = RewTerm(func=R.dof_pos_limits, weight=-1.0, params={"asset_cfg": ACTUATED})
    collision = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=COLLISION_BODIES), "threshold": 1.0},
    )
    # -- feet: contact / orientation ----------------------------------------
    feet_slip = RewTerm(func=R.feet_slip, weight=-0.1, params={"sensor_cfg": FEET_SENSOR, "asset_cfg": FEET})
    feet_roll = RewTerm(func=R.feet_roll, weight=-0.2, params={"asset_cfg": FEET})
    feet_pitch = RewTerm(func=R.feet_pitch, weight=-0.1, params={"asset_cfg": FEET})
    # -- feet: parametric placement (foot yaw / offset / swing) -------------
    foot_yaw_l = RewTerm(func=R.foot_yaw_l, weight=-1.0, params={"asset_cfg": FEET})
    foot_yaw_r = RewTerm(func=R.foot_yaw_r, weight=-1.0, params={"asset_cfg": FEET})
    feet_yaw_diff = RewTerm(func=R.feet_yaw_diff, weight=-0.5, params={"asset_cfg": FEET})
    feet_yaw_mean = RewTerm(func=R.feet_yaw_mean, weight=-0.5, params={"asset_cfg": FEET})
    feet_offset_x = RewTerm(func=R.feet_offset_x, weight=-12.0, params={"asset_cfg": FEET, "max_vel": 1.0})
    feet_offset_y = RewTerm(func=R.feet_offset_y, weight=-12.0, params={"asset_cfg": FEET, "max_vel": 1.0})
    feet_swing = RewTerm(func=R.feet_swing, weight=3.0, params={"sensor_cfg": FEET_SENSOR, "swing_period": 0.2})
    # -- arms: penalty starts strong (pins shoulders ~locked) and is decayed by the
    #    shoulder_release curriculum so arm-swing is freed once the legs gait forms.
    shoulder_deviation = RewTerm(
        func=mdp.joint_deviation_l1, weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=SHOULDER_JOINTS)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.35})


@configclass
class CurriculumCfg:
    # grow the commanded velocity range as the policy tracks the current one well
    velocity = CurrTerm(func=C.velocity_levels)
    # tighten action-rate smoothness only once it's tracking well (move fast first)
    action_rate = CurrTerm(func=C.action_rate_curriculum,
                           params={"term_name": "action_rate", "start_weight": -0.1,
                                   "end_weight": -1.0, "error_thresh": 0.45})
    # hold terrain flat until speed curriculum is satisfied, then ramp difficulty
    terrain = CurrTerm(func=C.gated_terrain_levels, params={"vel_scale_gate": 0.95})
    # keep shoulders pinned early, then decay the penalty to free arm-swing
    shoulder_release = CurrTerm(
        func=C.decay_shoulder_penalty,
        params={"term_name": "shoulder_deviation", "start_weight": -3.0, "end_weight": -0.1,
                "start_step": 60000, "end_step": 200000},
    )


@configclass
class K1WalkHTWKEnvCfg(LocomotionVelocityRoughEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: HTWKRewards = HTWKRewards()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.episode_length_s = 30.0

        # ---- flat terrain (HTWK K1 trains on plane) ----------------------------
        # flat plane: get 2 m/s solid on flat first (warm-start preserves the gait);
        # terrain is a separate follow-up run. (generator terrain face-planted the
        # flat-trained warm-start on contact.)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        # ---- HTWK init pose ----------------------------------------------------
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.58)
        self.scene.robot.init_state.joint_pos["Left_Hip_Pitch"] = -0.2
        self.scene.robot.init_state.joint_pos["Right_Hip_Pitch"] = -0.2
        self.scene.robot.init_state.joint_pos["Left_Knee_Pitch"] = 0.4
        self.scene.robot.init_state.joint_pos["Right_Knee_Pitch"] = 0.4
        self.scene.robot.init_state.joint_pos["Left_Ankle_Pitch"] = -0.25
        self.scene.robot.init_state.joint_pos["Right_Ankle_Pitch"] = -0.25

        # ---- domain randomization (HTWK-style) --------------------------------
        self.events.physics_material.params["static_friction_range"] = (0.1, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 1.5)
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Trunk"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
        self.events.base_com.params["asset_cfg"].body_names = ["Trunk"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.1), "y": (0.0, 0.1), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        }
        # HTWK pushes every ~5 s
        self.events.push_robot.interval_range_s = (5.0, 5.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}


@configclass
class K1WalkHTWKEnvCfg_PLAY(K1WalkHTWKEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        # mix of fast forward (up to 1.5 m/s) and turn-in-place (yaw with ~zero lin vel)
        self.commands.base_velocity.still_proportion = 0.05
        self.commands.base_velocity.vel_curriculum = False  # full commanded speed
        self.commands.base_velocity.debug_vis = True  # green = target vel, blue = actual vel
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_yaw = (-1.6, 1.6)
