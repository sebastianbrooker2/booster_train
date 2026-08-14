# Copyright (c) 2026 - K1 ball-stop task prototyping
# SPDX-License-Identifier: BSD-3-Clause

"""K1 standing (via the trained walk policy, commanded to zero velocity) while a
ball rolls at it.

Reuses the real Booster-K1-WalkHTWK-v0-Play env/policy so the observation and
action pipeline exactly matches training (no hand-rolled joint order/obs risk).
The velocity/gait command ranges are forced to zero so the policy sits in its
"stand still" mode (the same mode `still_proportion` samples during training),
and a ball is added to the scene and relaunched at the robot on every reset.

Usage:
    ./isaaclab.sh -p scripts/demo_ball_stand.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="K1 standing + ball rolling at it.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

import booster_train.tasks  # noqa: F401 - registers the gym task ids
from booster_train.tasks.manager_based.locomotion.k1_walk_htwk.walk_htwk_env_cfg import (
    K1WalkHTWKEnvCfg_PLAY,
)

POLICY_PATH = "logs/rsl_rl/k1_walk_htwk/exported/k1_walk_htwk_k1_walk_htwk.pt"
BALL_RADIUS = 0.11 * 0.8  # 20% smaller than the first ball-roll demo
BALL_SPEED = 2.0  # m/s toward the robot
BALL_SPAWN_X = 3.0


def build_cfg() -> K1WalkHTWKEnvCfg_PLAY:
    cfg = K1WalkHTWKEnvCfg_PLAY()
    cfg.scene.num_envs = 1

    # force the "stand still" command: zero every param so the gait clock freezes
    # (same mode `still_proportion` already samples during training)
    zero = (0.0, 0.0)
    r = cfg.commands.base_velocity.ranges
    r.lin_vel_x = zero
    r.lin_vel_y = zero
    r.ang_vel_yaw = zero
    r.gait_frequency = zero
    r.foot_yaw_l = zero
    r.foot_yaw_r = zero
    r.body_pitch_target = zero
    r.body_roll_target = zero
    r.feet_offset_x_target = zero
    r.feet_offset_y_target = zero
    cfg.commands.base_velocity.still_proportion = 1.0
    cfg.commands.base_velocity.vel_curriculum = False

    # ball: spawn ahead of the robot, relaunched at BALL_SPEED on every env reset
    cfg.scene.ball = RigidObjectCfg(
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
    cfg.events.reset_ball = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {},
            "velocity_range": {"x": (-BALL_SPEED, -BALL_SPEED)},
        },
    )
    return cfg


def main():
    env_cfg = build_cfg()
    env = gym.make("Booster-K1-WalkHTWK-v0-Play", cfg=env_cfg, render_mode=None)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    policy = torch.jit.load(POLICY_PATH, map_location=obs.device)
    policy.eval()

    print("STARTING_STAND_DEMO (close the viewer window to stop)")
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            obs = obs_dict["policy"]
        if step % 100 == 0:
            base_z = env.unwrapped.scene["robot"].data.root_pos_w[0, 2].item()
            ball_pos = env.unwrapped.scene["ball"].data.root_pos_w[0]
            print(
                f"STEP {step:5d}  robot_base_z={base_z:.3f}m  "
                f"ball_pos=({ball_pos[0]:+.2f},{ball_pos[1]:+.2f},{ball_pos[2]:.2f})  "
                f"terminated={bool(terminated[0])} truncated={bool(truncated[0])}"
            )
        step += 1

    print("DEMO_DONE")


if __name__ == "__main__":
    main()
    # bypass Isaac Sim's slow/hanging graceful shutdown - we already have what we need
    os._exit(0)
