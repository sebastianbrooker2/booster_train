# Copyright (c) 2026 - K1 scripted walk demo
# SPDX-License-Identifier: BSD-3-Clause

"""A ParameterWalkCommand that follows a fixed choreography instead of sampling.

Timeline (body-frame velocity): walk fwd -> run fast -> slow+stand -> walk back ->
strafe L -> strafe R -> stop -> circle (same facing) -> walk + figure-8 -> run + figure-8.
"""

from __future__ import annotations

import math
import torch
from isaaclab.utils import configclass

from booster_train.tasks.manager_based.locomotion.k1_walk_htwk.commands import (
    ParameterWalkCommand,
    ParameterWalkCommandCfg,
)

# each phase: (duration_s, fn(u) -> (vx, vy, yaw)) where u = seconds into the phase
_TWO_PI = 2.0 * math.pi
_SCRIPT = [
    (4.0, lambda u: (0.6, 0.0, 0.0)),                                   # walk forward
    (5.0, lambda u: (min(0.6 + 0.9 * u / 2.0, 1.5), 0.0, 0.0)),         # ramp to run fast
    (3.0, lambda u: (max(1.5 * (1.0 - u / 2.0), 0.0), 0.0, 0.0)),       # slow down
    (2.0, lambda u: (0.0, 0.0, 0.0)),                                   # stand still
    (4.0, lambda u: (-0.6, 0.0, 0.0)),                                  # walk backward
    (3.0, lambda u: (0.0, 0.6, 0.0)),                                   # strafe left
    (3.0, lambda u: (0.0, -0.6, 0.0)),                                  # strafe right
    (2.0, lambda u: (0.0, 0.0, 0.0)),                                   # stop
    (4.0, lambda u: (0.0, 0.0, 1.3)),                                   # turn left on the spot
    (4.0, lambda u: (0.0, 0.0, -1.3)),                                  # turn right on the spot
    (2.0, lambda u: (0.0, 0.0, 0.0)),                                   # stop
    (8.0, lambda u: (0.5 * math.cos(_TWO_PI * u / 8.0),                 # circle, same facing
                     0.5 * math.sin(_TWO_PI * u / 8.0), 0.0)),
    (10.0, lambda u: (0.5, 0.0, 1.2 * math.sin(_TWO_PI * u / 5.0))),    # walk + figure-8
    (10.0, lambda u: (1.0, 0.0, 1.5 * math.sin(_TWO_PI * u / 5.0))),    # run + figure-8
]
_TOTAL = sum(d for d, _ in _SCRIPT)


def _script_vel(t: float) -> tuple[float, float, float]:
    t = t % _TOTAL  # loop the choreography
    for dur, fn in _SCRIPT:
        if t < dur:
            return fn(t)
        t -= dur
    return (0.0, 0.0, 0.0)


class ScriptedWalkCommand(ParameterWalkCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.script_t = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        # fixed gait clock + neutral foot/body params; velocity comes from the script
        self.cmd[env_ids, :] = 0.0
        self.cmd[env_ids, 3] = 2.0      # gait_frequency
        self.gait_frequency[env_ids] = 2.0
        self.script_t[env_ids] = 0.0    # restart the script on (rare) episode reset

    def _update_command(self):
        super()._update_command()       # advance the gait clock
        self.script_t += self._env.step_dt
        for i in range(self.num_envs):
            vx, vy, yaw = _script_vel(float(self.script_t[i]))
            self.cmd[i, 0], self.cmd[i, 1], self.cmd[i, 2] = vx, vy, yaw


@configclass
class ScriptedWalkCommandCfg(ParameterWalkCommandCfg):
    class_type: type = ScriptedWalkCommand
    vel_curriculum: bool = False
