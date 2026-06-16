# Copyright (c) 2026 - K1 ParameterWalk (HTWK port)
# SPDX-License-Identifier: BSD-3-Clause

"""Custom env that clips the TOTAL reward at zero (HTWK `only_positive_rewards`).

With this reward set the penalties are large (base_height -20, orientation -20,
feet_offset -12, ...). Clipping the summed reward at 0 stops the policy from
"falling on purpose" to escape accumulating negative reward — the classic
early-termination exploit. Per-term values are still logged unclipped.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv


class K1WalkHTWKEnv(ManagerBasedRLEnv):
    def load_managers(self):
        super().load_managers()
        # wrap reward_manager.compute so the per-step total is clipped at 0
        _orig_compute = self.reward_manager.compute

        def _positive_compute(dt: float) -> torch.Tensor:
            return torch.clip(_orig_compute(dt), min=0.0)

        self.reward_manager.compute = _positive_compute
