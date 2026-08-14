# Copyright (c) 2026 - K1 Ball Control (RL, robot-soccer first touch)
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ball control: stand and kill an incoming ball's velocity (a football "trap").

The robot spawns at a random point in a 1 m disk, facing anywhere from directly
at the ball to side-on to it (never back-on - a +/-90 deg window around facing
the ball). The ball is launched at wherever the robot landed. Reward is
dominated by driving the ball's post-contact speed toward zero (or a slow roll
back the way it came), layered on the same stand-still posture/effort shaping
used by k1_walk_htwk's zero-velocity mode.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from booster_train.assets.robots.booster import BOOSTER_K1_CFG
import booster_train.tasks.manager_based.locomotion.k1_ball_control.rewards as R
import booster_train.tasks.manager_based.locomotion.k1_ball_control.observations as O
import booster_train.tasks.manager_based.locomotion.k1_ball_control.events as E

# legs (12) + shoulders (4) = 16 DOF, same actuated set as k1_walk_htwk
LEG_JOINTS = [
    ".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw",
    ".*_Knee_Pitch", ".*_Ankle_Pitch", ".*_Ankle_Roll",
]
SHOULDER_JOINTS = [".*_Shoulder_Pitch", ".*_Shoulder_Roll"]
ACTION_JOINTS = LEG_JOINTS + SHOULDER_JOINTS
ACTUATED = SceneEntityCfg("robot", joint_names=ACTION_JOINTS)
STAND_HEIGHT = 0.55

BALL_RADIUS = 0.11 * 0.8  # 20% smaller than the viewer-demo ball; nominal, scale is randomized per env
BALL_SIZE_SCALE_RANGE = (0.85, 1.2)  # per-env ball-size variety (different match balls), fixed for that env's life
BALL_SPAWN_X = 3.0
SPAWN_RADIUS = 0.5  # robot spawns within this radius of the env origin
FACING_WINDOW = math.pi / 4  # +/-45 deg around directly facing the ball (a walk-to-ball
                              # controller is expected to square the robot up beforehand)
SQUARE_ON_FRACTION = 0.4  # fraction of resets that start (near-)square-on rather than spread
BALL_SPEED_RANGE = (1.5, 4.5)  # resampled every episode
TARGET_DIR_SPREAD = math.pi / 6  # unused while TARGET_DIR_RANDOM_FRACTION=1.0; kept for later tuning
TARGET_DIR_RANDOM_FRACTION = 1.0  # fraction of episodes with a fully random target direction
TARGET_DIST = 0.4  # meters from the robot the controlled touch should aim to settle at


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ACTION_JOINTS, scale=0.8, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ACTUATED}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": ACTUATED}, scale=0.1, noise=Unoise(n_min=-0.15, n_max=0.15)
        )
        actions = ObsTerm(func=mdp.last_action)
        # egocentric ball state - the new input this task needs to "see" the ball
        ball_pos_b = ObsTerm(func=O.ball_pos_b, noise=Unoise(n_min=-0.02, n_max=0.02))
        ball_lin_vel_b = ObsTerm(func=O.ball_lin_vel_b, noise=Unoise(n_min=-0.05, n_max=0.05))

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
        ball_pos_b = ObsTerm(func=O.ball_pos_b)
        ball_lin_vel_b = ObsTerm(func=O.ball_lin_vel_b)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)   # privileged
        base_height = ObsTerm(func=mdp.base_pos_z)      # privileged

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class BallControlRewards:
    # -- task: kill the ball's velocity (dominant baseline) -------------------
    ball_speed_penalty = RewTerm(func=R.ball_speed_penalty, weight=15.0, params={"sigma": 0.4})
    # -- controlled touch toward a target direction (emphasized over the exact
    # settling spot) / a specific spot to settle at (secondary, lower weight) -
    ball_target_direction_bonus = RewTerm(func=R.ball_target_direction_bonus, weight=8.0, params={"sigma": 0.6})
    ball_target_position_bonus = RewTerm(
        func=R.ball_target_position_bonus, weight=3.0, params={"target_dist": TARGET_DIST, "sigma": 0.15}
    )
    inside_foot_contact = RewTerm(func=R.inside_foot_contact, weight=3.0, params={"contact_dist": 0.15})
    contact_shin_uprightness = RewTerm(func=R.contact_shin_uprightness, weight=4.0, params={"contact_dist": 0.15})
    lifted_touching_foot = RewTerm(
        # tighter contact_dist than inside_foot_contact/contact_shin_uprightness on purpose:
        # this one should only fire right at genuine contact, not during an extended hover
        # near the ball - a loose gate let the policy camp on one leg early and fall over
        func=R.lifted_touching_foot, weight=4.0, params={"contact_dist": 0.08, "target_lift": 0.05, "sigma": 0.03}
    )
    ball_overshoot_penalty = RewTerm(func=R.ball_overshoot_penalty, weight=-20.0, params={"max_overshoot": 1.0})
    # -- standing / balance ---------------------------------------------------
    base_height = RewTerm(func=R.base_height, weight=-8.0, params={"target_height": STAND_HEIGHT})
    orientation = RewTerm(func=R.orientation, weight=-8.0)
    lin_vel_z = RewTerm(func=R.lin_vel_z, weight=-2.0)
    ang_vel_xy = RewTerm(func=R.ang_vel_xy, weight=-0.2)
    # -- effort / smoothness / limits -----------------------------------------
    torques = RewTerm(func=R.torques, weight=-3.0e-5, params={"asset_cfg": ACTUATED})
    dof_acc = RewTerm(func=R.dof_acc, weight=-1.0e-7, params={"asset_cfg": ACTUATED})
    action_rate = RewTerm(func=R.action_rate, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": ACTUATED})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.35})


@configclass
class CurriculumCfg:
    # none yet - fixed spawn radius / ball speed range for this first version
    pass


@configclass
class K1BallControlEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: BallControlRewards = BallControlRewards()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # bent-knee standing pose (same as k1_walk_htwk) - the raw default (straight
        # legs, all-zero) is not a stable equilibrium under pure PD hold
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.58)
        self.scene.robot.init_state.joint_pos["Left_Hip_Pitch"] = -0.2
        self.scene.robot.init_state.joint_pos["Right_Hip_Pitch"] = -0.2
        self.scene.robot.init_state.joint_pos["Left_Knee_Pitch"] = 0.4
        self.scene.robot.init_state.joint_pos["Right_Knee_Pitch"] = 0.4
        self.scene.robot.init_state.joint_pos["Left_Ankle_Pitch"] = -0.25
        self.scene.robot.init_state.joint_pos["Right_Ankle_Pitch"] = -0.25

        self.episode_length_s = 6.0

        # wide enough that a 3 m ball throw + 0.5 m spawn disk never crosses into
        # the neighboring env's cell (base default of 2.5 m was too tight - the
        # ball visually spawned inside the next env over, though physics itself
        # stays correctly isolated between envs regardless)
        self.scene.env_spacing = 8.0

        # flat pitch - a rolling ball needs a flat plane, not rough terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        # no velocity command - this is a stand-and-react skill, not locomotion
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ---- the ball -----------------------------------------------------------
        self.scene.ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=BALL_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.1)),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.6
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.43),  # ~soccer ball
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(BALL_SPAWN_X, 0.0, BALL_RADIUS)),
        )
        # ball size is baked into the collider at spawn - can't be resampled every
        # episode like position/velocity, so each of the N parallel envs gets a
        # fixed-but-different ball size for its lifetime (still gives the policy a
        # variety of ball sizes across the training population). Requires
        # replicate_physics=False so envs aren't cloned identically.
        self.scene.replicate_physics = False
        self.events.randomize_ball_size = EventTerm(
            func=mdp.randomize_rigid_body_scale,
            mode="usd",
            params={"asset_cfg": SceneEntityCfg("ball"), "scale_range": BALL_SIZE_SCALE_RANGE},
        )

        # ---- reset: robot in a 0.5 m disk mostly square-on to the ball, ball
        # launched at wherever the robot landed with a per-episode target
        # direction for the controlled touch (speed/target resampled every ep) --
        self.events.reset_base = EventTerm(
            func=E.reset_robot_and_ball,
            mode="reset",
            params={
                "spawn_radius": SPAWN_RADIUS,
                "facing_window": FACING_WINDOW,
                "square_on_fraction": SQUARE_ON_FRACTION,
                "ball_speed_range": BALL_SPEED_RANGE,
                "target_dir_spread": TARGET_DIR_SPREAD,
                "target_dir_random_fraction": TARGET_DIR_RANDOM_FRACTION,
            },
        )
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)

        # ---- light domain randomization (sim2real), high floor grip --------------
        self.events.physics_material.params["static_friction_range"] = (0.9, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.3)
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1BallControlEnvCfg_PLAY(K1BallControlEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
