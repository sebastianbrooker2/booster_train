# Copyright (c) 2026 - K1 scripted walk demo env
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from booster_train.tasks.manager_based.locomotion.k1_walk_htwk.walk_htwk_env_cfg import K1WalkHTWKEnvCfg
from booster_train.tasks.manager_based.locomotion.k1_walk_htwk.commands_script import ScriptedWalkCommandCfg


@configclass
class K1WalkHTWKScriptEnvCfg(K1WalkHTWKEnvCfg):
    """One robot following the fixed choreography (load a trained checkpoint into it)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.episode_length_s = 120.0          # long enough to loop the ~64s script

        # chase camera locked onto the robot (for the live view AND --video recording)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (2.5, 2.5, 1.2)      # offset behind/above the robot
        self.viewer.lookat = (0.0, 0.0, 0.4)
        self.viewer.resolution = (1920, 1080)
        self.actions.joint_pos.scale = 0.5     # MATCH the demo checkpoint (iter-12700 was scale 0.5)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        # swap the random command for the scripted choreography (with vel arrows)
        self.commands.base_velocity = ScriptedWalkCommandCfg(
            asset_name="robot",
            resampling_time_range=(1000.0, 1000.0),   # never resample mid-episode
            debug_vis=True,                            # green=target, blue=actual
            vel_curriculum=False,
        )


@configclass
class K1WalkHTWKScriptFastEnvCfg(K1WalkHTWKScriptEnvCfg):
    """Same choreography, scale 0.8 — for the unleashed (fast) walker checkpoints."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = 0.8   # match the unleashed walker
