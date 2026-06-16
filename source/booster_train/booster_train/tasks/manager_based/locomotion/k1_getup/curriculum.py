# Copyright (c) 2026 - K1 Get-up (curriculum RL)
# SPDX-License-Identifier: BSD-3-Clause

"""Two curricula for the get-up task:
  1. assist_force  — strong upward pull on the trunk early, weaned to 0 as it
                     succeeds (HoST-style, success-gated).
  2. speed_pressure — ramp the time-penalty weight up over training so the rise
                     gets faster and faster once it can stand at all.
"""

from __future__ import annotations

import booster_train.tasks.manager_based.locomotion.k1_getup.rewards as R


def assist_force_curriculum(
    env, env_ids,
    initial: float = 120.0,
    target_height: float = 0.72,
    force_step: float = 0.06,
) -> float:
    """Hold a strong upward assist early; decay it while it's reaching standing height."""
    if not hasattr(env, "_assist_force"):
        env._assist_force = float(initial)
    mean_head = R._head_height(env).mean().item()
    if mean_head > target_height and env._assist_force > 0.0:
        env._assist_force = max(0.0, env._assist_force - force_step)
    return env._assist_force


def speed_pressure_curriculum(
    env, env_ids,
    term_name: str = "time_penalty",
    start_weight: float = -0.2,
    end_weight: float = -2.0,
    start_step: int = 40000,
    end_step: int = 160000,
) -> float:
    """Linearly ramp the not-standing time-penalty weight (in env-steps) so the
    rise is rewarded for getting faster once it can stand at all."""
    s = env.common_step_counter
    if s <= start_step:
        w = start_weight
    elif s >= end_step:
        w = end_weight
    else:
        f = (s - start_step) / (end_step - start_step)
        w = start_weight + f * (end_weight - start_weight)
    cfg = env.reward_manager.get_term_cfg(term_name)
    cfg.weight = w
    env.reward_manager.set_term_cfg(term_name, cfg)
    return w
