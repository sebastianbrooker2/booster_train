# Copyright (c) 2026 - K1 Get-up (curriculum RL, discovery-based)
# SPDX-License-Identifier: BSD-3-Clause

"""K1 get-up: ONE policy that rises FAST from any fallen posture (supine/prone/side).

Discovery-based (no reference motion). The motion emerges from a big standing
bonus + dense height + time pressure (fast) + continuous-progress (one motion) +
a light supine backward-roll nudge, with motor-friendly regularization. Two
curricula: a success-gated upward assist force, and a ramping speed pressure.
"""

import math
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from booster_train.assets.robots.booster import BOOSTER_K1_CFG
import booster_train.tasks.manager_based.locomotion.k1_getup.rewards as R
import booster_train.tasks.manager_based.locomotion.k1_getup.curriculum as C

# legs (12) + arms (8) — arms actuated for push-off; head locked
ACTION_JOINTS = [
    ".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw", ".*_Knee_Pitch", ".*_Ankle_Pitch", ".*_Ankle_Roll",
    ".*_Shoulder_Pitch", ".*_Shoulder_Roll", ".*_Elbow_Pitch", ".*_Elbow_Yaw",
]
ACTUATED = SceneEntityCfg("robot", joint_names=ACTION_JOINTS)
HIP_PITCH = SceneEntityCfg("robot", joint_names=[".*_Hip_Pitch"])
TARGET_HEIGHT = 0.72  # head height when standing (K1 ~0.58 base -> head ~0.72)


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ACTION_JOINTS, scale=0.8, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # posture-aware proprioception (projected gravity tells it which way it fell)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ACTUATED}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ACTUATED}, scale=0.1, noise=Unoise(n_min=-0.15, n_max=0.15))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ACTUATED})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ACTUATED}, scale=0.1)
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_height = ObsTerm(func=mdp.base_pos_z)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class GetupRewards:
    # -- task: stand up ------------------------------------------------------
    standing_bonus = RewTerm(func=R.standing_bonus, weight=10.0, params={"target_height": TARGET_HEIGHT})
    hip_height = RewTerm(func=R.hip_height, weight=10.0)   # PRIMARY dense climb (pelvis rises first)
    head_height = RewTerm(func=R.head_height, weight=3.0, params={"min_height": 0.5})  # only above 0.5m (no head-spring), drives the final stand
    upright = RewTerm(func=R.upright, weight=2.0)
    feet_under_body = RewTerm(func=R.feet_under_body, weight=1.0)
    # -- FAST: time pressure (weight ramped by speed_pressure curriculum) -----
    time_penalty = RewTerm(func=R.not_standing_time_penalty, weight=-0.2, params={"target_height": TARGET_HEIGHT})
    rising_velocity = RewTerm(func=R.rising_velocity, weight=1.0)
    # -- prefer hip FLEXION (knees in front / tuck / pike) over extension (arch) -
    hip_flexion = RewTerm(func=R.hip_flexion_reward, weight=2.0, params={"asset_cfg": HIP_PITCH})
    hip_extension = RewTerm(func=R.hip_extension_penalty, weight=-3.0, params={"asset_cfg": HIP_PITCH})
    # -- supine: throw the hips up and over the head (starts the backward roll) -
    supine_hips_over_head = RewTerm(func=R.supine_hips_over_head, weight=4.0)
    # -- supine backward-roll nudge (light; verify sign in viewer) ------------
    supine_backward_roll = RewTerm(func=R.supine_backward_roll, weight=0.5)
    # -- stability once up ---------------------------------------------------
    standing_stability = RewTerm(func=R.standing_stability, weight=2.0, params={"target_height": TARGET_HEIGHT})
    # -- assist force application (side-effect; returns 0) -------------------
    assist_force = RewTerm(func=R.apply_assist_force, weight=0.0)
    # -- deployability / motor-friendly regularization (light) ---------------
    dof_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4, params={"asset_cfg": ACTUATED})
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3, params={"asset_cfg": ACTUATED})
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7, params={"asset_cfg": ACTUATED})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": ACTUATED})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # no contact termination — being on the ground is the whole point


@configclass
class CurriculumCfg:
    assist_force = CurrTerm(func=C.assist_force_curriculum,
                            params={"initial": 120.0, "target_height": TARGET_HEIGHT, "force_step": 0.06})
    speed_pressure = CurrTerm(func=C.speed_pressure_curriculum,
                              params={"term_name": "time_penalty", "start_weight": -0.2, "end_weight": -2.0,
                                      "start_step": 40000, "end_step": 160000})


@configclass
class K1GetupEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: GetupRewards = GetupRewards()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.episode_length_s = 5.0


        # flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        # no velocity command (getup is posture, not locomotion) — keep but zero & don't observe
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ---- spawn in DIVERSE fallen postures (supine / prone / side) ----------
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.35)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.3, 0.3), "y": (-0.3, 0.3),
                "roll": (-math.pi, math.pi), "pitch": (-math.pi, math.pi), "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        }
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)  # varied limb configs

        # ---- light domain randomization (sim2real), no push, we apply our own force
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # no contact termination
        self.terminations.base_contact = None


@configclass
class K1GetupEnvCfg_PLAY(K1GetupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 8.0
        self.observations.policy.enable_corruption = False
