# Copyright (c) 2026 - K1 Get-up (curriculum RL)
# SPDX-License-Identifier: BSD-3-Clause

"""Two curricula for the get-up task:
  1. assist_force  — strong upward pull on the trunk early, weaned to 0 as it
                     succeeds (HoST-style, success-gated).
  2. speed_pressure — ramp the time-penalty weight up over training so the rise
                     gets faster and faster once it can stand at all.
"""

from __future__ import annotations

import torch

import booster_train.tasks.manager_based.locomotion.k1_getup.rewards as R


def assist_force_curriculum(
    env, env_ids,
    initial: float = 120.0,
    threshold_height: float = 0.50,
    force_step: float = 4.0,
) -> float:
    """PER-ENV wean of the upward assist (HoST-style vertical-pull curriculum).

    Runs at reset (curriculum_manager.compute fires BEFORE scene.reset), so
    env._epi_max_base still holds the just-finished episode's PEAK BASE height for
    each resetting env. For every resetting env whose base got above `threshold_height`
    (0.50 m, i.e. ~88% of K1's verified 0.57 m standing height), drop THAT env's assist
    by `force_step` (monotonic -> 0). Envs that never got up keep their assist, so the
    population transitions gradually from assisted to fully unassisted — and the final
    policy stands with no assist.
    """
    R._ensure_assist_state(env, initial)
    reached = (env._epi_max_base[env_ids] > threshold_height).float()
    env._assist_force[env_ids] = torch.clamp(
        env._assist_force[env_ids] - force_step * reached, min=0.0
    )
    env._epi_max_base[env_ids] = 0.0   # reset peak tracking for the new episode
    return env._assist_force.mean().item()


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
