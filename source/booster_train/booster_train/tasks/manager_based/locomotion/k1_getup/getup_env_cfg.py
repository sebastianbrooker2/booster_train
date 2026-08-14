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
ARM_JOINTS = SceneEntityCfg("robot", joint_names=[".*_Shoulder_Pitch", ".*_Shoulder_Roll", ".*_Elbow_Pitch", ".*_Elbow_Yaw"])
STAND_HEIGHT = 0.57  # VERIFIED K1 standing base/pelvis height (asset init_state pos z)


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
    standing_bonus = RewTerm(func=R.standing_bonus, weight=10.0, params={"stand_height": STAND_HEIGHT})
    upright_climb = RewTerm(func=R.upright_climb, weight=10.0, params={"stand_height": STAND_HEIGHT})  # PRIMARY dense climb: base height GATED by uprightness (no downward-dog farm)
    head_height = RewTerm(func=R.head_height, weight=3.0, params={"max_diff": 0.7})  # dense head-above-feet: high only when upright
    upright = RewTerm(func=R.upright, weight=3.0)
    feet_under_body = RewTerm(func=R.feet_under_body, weight=2.0)  # reward planting feet under (knee->stand)
    # -- FAST: time pressure (weight ramped by speed_pressure curriculum) -----
    time_penalty = RewTerm(func=R.not_standing_time_penalty, weight=-0.2, params={"stand_height": STAND_HEIGHT})
    rising_velocity = RewTerm(func=R.rising_velocity, weight=1.0, params={"max_vel": 1.0})
    # NOTE: the supine-specific "do a backward roll" nudges (hip flexion/extension,
    # hips-over-head, backward-roll) were REMOVED. They prescribed one gymnastic trick,
    # were sign-uncertain (could fight the motion), and didn't generalize to prone/side.
    # The rise now EMERGES from the task reward + the decaying per-env assist (HoST-style),
    # which works from any posture (supine / prone / side).
    # -- stability once up ---------------------------------------------------
    standing_stability = RewTerm(func=R.standing_stability, weight=2.0, params={"stand_height": STAND_HEIGHT})
    arms_at_side = RewTerm(func=R.arms_at_side, weight=1.0, params={"asset_cfg": ARM_JOINTS, "stand_height": STAND_HEIGHT})  # end with arms down at sides
    # (assist force REMOVED — it was a weight-0 reward term, which IsaacLab skips, so it
    #  never ran. Get-ups have been fully unassisted the whole time; keeping it out = clean.)
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
    # TIME IS THE PRIORITY: ramp the not-standing time penalty hard and early. A flip
    # (prone->supine to reuse the supine skill) earns ~0 shaping (body stays low/horizontal)
    # so a heavy per-step time cost makes the flip strictly wasteful — the fastest path from
    # EACH posture (direct prone push-up, direct supine rise) wins instead.
    speed_pressure = CurrTerm(func=C.speed_pressure_curriculum,
                              params={"term_name": "time_penalty", "start_weight": -0.3, "end_weight": -5.0,
                                      "start_step": 8000, "end_step": 50000})


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
        self.episode_length_s = 8.0   # room to discover the full rise (was 5.0)


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
        # HIGH floor grip so the feet can't SLIDE — kills the "splits then squeeze the legs
        # back in" exploit (which needs the feet to slide on the ground) and is more
        # sim2real-honest. Band kept for robustness but the slippery low end (0.4) is gone.
        self.events.physics_material.params["static_friction_range"] = (0.9, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.3)
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
