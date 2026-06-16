# Copyright (c) 2026 - K1 ParameterWalk (HTWK port)
# SPDX-License-Identifier: BSD-3-Clause

"""Parametric walk command with a gait clock, ported from HTWK Gym's K1 ParameterWalk.

The command vector exposed to the policy is 12-dim:
    [0] lin_vel_x        [1] lin_vel_y       [2] ang_vel_yaw
    [3] gait_frequency   [4] foot_yaw_L      [5] foot_yaw_R
    [6] body_pitch_tgt   [7] body_roll_tgt   [8] feet_offset_x_tgt  [9] feet_offset_y_tgt
    [10] cos(2*pi*gait_process)   [11] sin(2*pi*gait_process)

A fraction `still_proportion` of envs are set to "stand" at each resample: the first
four entries (lin/ang vel + gait_frequency) are zeroed, which freezes the gait clock.
"""

from __future__ import annotations

from dataclasses import MISSING
import torch
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils


class ParameterWalkCommand(CommandTerm):
    """10 parametric command params + an internal gait-phase clock (cos/sin appended)."""

    cfg: "ParameterWalkCommandCfg"

    def __init__(self, cfg: "ParameterWalkCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        # 10 raw command params
        self.cmd = torch.zeros(self.num_envs, 10, device=self.device)
        # velocity-range curriculum scalar (grows 0..1 as tracking improves)
        self.vel_scale = float(cfg.init_vel_scale) if cfg.vel_curriculum else 1.0
        # gait clock state
        self.gait_frequency = torch.zeros(self.num_envs, device=self.device)
        # decorrelate initial phase across envs
        self.gait_process = torch.rand(self.num_envs, device=self.device)
        # logging
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"ParameterWalkCommand: {self.num_envs} envs, 10 params + gait clock"

    # -- the tensor the policy observes (generated_commands reads this) ----------
    @property
    def command(self) -> torch.Tensor:
        phase = 2.0 * torch.pi * self.gait_process
        return torch.cat(
            (self.cmd, torch.cos(phase).unsqueeze(-1), torch.sin(phase).unsqueeze(-1)), dim=-1
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = self.cmd  # alias
        rng = self.cfg.ranges
        n = len(env_ids)

        def U(lo_hi):
            lo, hi = lo_hi
            return torch.empty(n, device=self.device).uniform_(lo, hi)

        # velocity ranges shrink by the curriculum scalar (gait freq stays full)
        s = self.vel_scale
        r[env_ids, 0] = U(rng.lin_vel_x) * s
        r[env_ids, 1] = U(rng.lin_vel_y) * s
        r[env_ids, 2] = U(rng.ang_vel_yaw) * s
        r[env_ids, 3] = U(rng.gait_frequency)
        r[env_ids, 4] = U(rng.foot_yaw_l)
        r[env_ids, 5] = U(rng.foot_yaw_r)
        r[env_ids, 6] = U(rng.body_pitch_target)
        r[env_ids, 7] = U(rng.body_roll_target)
        r[env_ids, 8] = U(rng.feet_offset_x_target)
        r[env_ids, 9] = U(rng.feet_offset_y_target)

        self.gait_frequency[env_ids] = r[env_ids, 3]

        # a subset stand still: zero the first 4 (vel + gait freq) -> frozen clock
        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        k = int(self.cfg.still_proportion * n)
        if k > 0:
            still = env_ids_t[torch.randperm(n, device=self.device)[:k]]
            self.cmd[still, :4] = 0.0
            self.gait_frequency[still] = 0.0

    def _update_command(self):
        # advance the gait clock every control step
        self.gait_process[:] = torch.fmod(self.gait_process + self._env.step_dt * self.gait_frequency, 1.0)

    def _update_metrics(self):
        vel_b = self.robot.data.root_lin_vel_b
        ang_b = self.robot.data.root_ang_vel_b
        self.metrics["error_vel_xy"] = torch.norm(self.cmd[:, :2] - vel_b[:, :2], dim=-1)
        self.metrics["error_vel_yaw"] = torch.abs(self.cmd[:, 2] - ang_b[:, 2])

    # -- debug visualization: green = target velocity, blue = actual velocity ----
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        elif hasattr(self, "goal_vel_visualizer"):
            self.goal_vel_visualizer.set_visibility(False)
            self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        des_scale, des_quat = self._resolve_xy_velocity_to_arrow(self.cmd[:, :2])
        cur_scale, cur_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, des_quat, des_scale)
        self.current_vel_visualizer.visualize(base_pos_w, cur_quat, cur_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor):
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class ParameterWalkCommandCfg(CommandTermCfg):
    """Configuration for the HTWK parametric walk command."""

    class_type: type = ParameterWalkCommand
    asset_name: str = MISSING
    still_proportion: float = 0.1
    # velocity curriculum: start commands at init_vel_scale of full range, grow to 1.0
    vel_curriculum: bool = True
    init_vel_scale: float = 0.5  # warm-start already handles ~1 m/s, so start there
    vel_scale_step: float = 2.0e-4      # growth per step when tracking is good
    vel_scale_error_thresh: float = 0.4  # m/s mean xy error below which we grow

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-1.0, 2.0)  # forward-biased, fast; curriculum finds the real ceiling
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_yaw: tuple[float, float] = (-1.6, 1.6)
        gait_frequency: tuple[float, float] = (1.5, 3.0)  # allow faster cadence
        foot_yaw_l: tuple[float, float] = (-0.7, 0.7)
        foot_yaw_r: tuple[float, float] = (-0.7, 0.7)
        body_pitch_target: tuple[float, float] = (-0.1, 0.3)
        body_roll_target: tuple[float, float] = (-0.1, 0.1)
        feet_offset_x_target: tuple[float, float] = (-0.15, 0.15)
        feet_offset_y_target: tuple[float, float] = (-0.08, 0.15)

    ranges: Ranges = Ranges()

    # debug-vis arrows: green = target velocity, blue = actual velocity
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
