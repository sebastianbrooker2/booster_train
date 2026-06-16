# Copyright (c) 2026 - K1 ParameterWalk (HTWK port)
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms: grow the commanded velocity range, and free the shoulders late."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


def velocity_levels(env: ManagerBasedRLEnv, env_ids) -> float:
    """Grow the command velocity scale toward 1.0 while xy tracking error is low.

    The ParameterWalkCommand scales its lin/ang velocity ranges by `vel_scale`.
    We nudge it up a touch each step the policy is tracking well, so it starts
    slow/easy and only gets asked for higher speeds once it can hit the current ones.
    """
    term = env.command_manager.get_term("base_velocity")
    if not getattr(term.cfg, "vel_curriculum", False):
        return getattr(term, "vel_scale", 1.0)
    err = term.metrics["error_vel_xy"].mean().item()
    if err < term.cfg.vel_scale_error_thresh and term.vel_scale < 1.0:
        term.vel_scale = min(1.0, term.vel_scale + term.cfg.vel_scale_step)
    return term.vel_scale


def gated_terrain_levels(env: ManagerBasedRLEnv, env_ids, vel_scale_gate: float = 0.95) -> float:
    """Keep terrain flat (level 0) until the speed curriculum is satisfied, then ramp it.

    While `vel_scale` < gate, robots stay on the easiest (≈flat) terrain so they can
    learn to go fast on flat ground first. Once the velocity curriculum has reached
    near-max (it only does so while tracking well), the standard distance-based
    terrain progression kicks in and difficulty ramps automatically.
    """
    terrain = env.scene.terrain
    # flat plane (e.g. the viewer overrides to plane) -> nothing to ramp
    if getattr(terrain.cfg, "terrain_generator", None) is None:
        return 0.0
    term = env.command_manager.get_term("base_velocity")
    if getattr(term, "vel_scale", 1.0) < vel_scale_gate:
        return float(torch.mean(terrain.terrain_levels.float()))  # hold flat, no promotion
    return float(mdp.terrain_levels_vel(env, env_ids))


def action_rate_curriculum(
    env, env_ids,
    term_name: str = "action_rate",
    start_weight: float = -0.1,
    end_weight: float = -1.0,
    error_thresh: float = 0.45,
    step: float = 3.0e-4,
) -> float:
    """Tighten the action-rate (smoothness) penalty ONLY while velocity tracking is
    good — so it first learns to move fast, then gets smoothed for deployability."""
    term = env.command_manager.get_term("base_velocity")
    err = term.metrics["error_vel_xy"].mean().item()
    cfg = env.reward_manager.get_term_cfg(term_name)
    w = cfg.weight
    if err < error_thresh and w > end_weight:
        w = max(end_weight, w - step)   # more negative = tighter smoothness
        cfg.weight = w
        env.reward_manager.set_term_cfg(term_name, cfg)
    return w


def decay_shoulder_penalty(
    env: ManagerBasedRLEnv,
    env_ids,
    term_name: str = "shoulder_deviation",
    start_weight: float = -3.0,
    end_weight: float = -0.1,
    start_step: int = 60000,
    end_step: int = 200000,
) -> float:
    """Linearly relax the shoulder-movement penalty over training.

    Early on the strong penalty pins the shoulders at default (effectively locked,
    so the legs gait forms cleanly without the one-arm-swing exploit). It then
    decays so the policy can start using arm-swing for balance later.
    `start_step`/`end_step` are in env-steps (= iterations * num_steps_per_env).
    """
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
