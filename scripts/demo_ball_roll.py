# Copyright (c) 2026 - K1 ball-stop task prototyping
# SPDX-License-Identifier: BSD-3-Clause

"""Quick standalone scene check: K1 standing, a ball rolling toward it.

Prints ball distance-to-robot and robot base height over time so the ball's
approach and the robot's standing stability can be verified without a GUI.

Usage:
    ./isaaclab.sh -p scripts/demo_ball_roll.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Ball-rolls-toward-standing-K1 scene check.")
parser.add_argument("--reset_every", type=int, default=360, help="Steps between ball resets.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

from booster_train.assets.robots.booster import BOOSTER_K1_CFG


def design_scene():
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    )

    robot_cfg = BOOSTER_K1_CFG.replace(prim_path="/World/Robot")
    # bent-knee standing pose (same as k1_walk_htwk) - the raw default (straight legs,
    # all-zero) is not a stable equilibrium and the robot buckles under pure PD hold
    robot_cfg.init_state.joint_pos["Left_Hip_Pitch"] = -0.2
    robot_cfg.init_state.joint_pos["Right_Hip_Pitch"] = -0.2
    robot_cfg.init_state.joint_pos["Left_Knee_Pitch"] = 0.4
    robot_cfg.init_state.joint_pos["Right_Knee_Pitch"] = 0.4
    robot_cfg.init_state.joint_pos["Left_Ankle_Pitch"] = -0.25
    robot_cfg.init_state.joint_pos["Right_Ankle_Pitch"] = -0.25
    robot = Articulation(robot_cfg)

    ball_cfg = RigidObjectCfg(
        prim_path="/World/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.5, restitution=0.6
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),  # ~soccer ball
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        # spawn 3m in front of the robot (+x), rolling toward it at 2 m/s
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 0.11), lin_vel=(-2.0, 0.0, 0.0)),
    )
    ball = RigidObject(ball_cfg)

    return robot, ball


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, -3.5, 2.0], target=[1.0, 0.0, 0.5])

    robot, ball = design_scene()
    sim.reset()

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_ball_state = ball.data.default_root_state.clone()
    sim_dt = sim.get_physics_dt()

    def reset_ball():
        ball.write_root_pose_to_sim(default_ball_state[:, :7])
        ball.write_root_velocity_to_sim(default_ball_state[:, 7:])
        ball.reset()

    reset_ball()

    print("STARTING_DEMO (close the viewer window to stop)")
    i = 0
    while simulation_app.is_running():
        if i > 0 and i % args_cli.reset_every == 0:
            reset_ball()

        # hold the robot's default standing pose (mannequin PD hold, no controller yet)
        robot.set_joint_position_target(default_joint_pos)
        robot.write_data_to_sim()
        ball.write_data_to_sim()

        sim.step()

        robot.update(sim_dt)
        ball.update(sim_dt)

        if i % 40 == 0:
            bp = ball.data.root_pos_w[0]
            rp = robot.data.root_pos_w[0]
            dist = torch.norm(bp[:2] - rp[:2])
            print(
                f"STEP {i:5d} t={i * sim_dt:6.2f}s  ball_pos=({bp[0]:+.2f},{bp[1]:+.2f},{bp[2]:.2f})  "
                f"dist_to_robot={dist.item():.2f}m  robot_base_z={rp[2].item():.3f}m"
            )
        i += 1

    print("DEMO_DONE")


if __name__ == "__main__":
    main()
    # bypass Isaac Sim's slow/hanging graceful shutdown - we already have what we need
    os._exit(0)
