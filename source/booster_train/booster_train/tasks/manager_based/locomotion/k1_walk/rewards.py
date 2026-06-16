import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse


# ---------------------------------------------------------------------------
# Gait-shaping rewards: encourage an alternating, knee-driven bipedal gait
# instead of a stiff-knee, ankle-push tip-toe gait.
# ---------------------------------------------------------------------------

# Pair each foot with its own knee explicitly so left/right can never cross.
_FOOT_KNEE_PAIRS = (
    ("left_foot_link", "Left_Knee_Pitch"),
    ("right_foot_link", "Right_Knee_Pitch"),
)


def swing_knee_flexion(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_flex: float = 1.0,
) -> torch.Tensor:
    """Reward extra knee flexion of the leg whose foot is in swing (off ground).

    Gated by per-foot contact so only the airborne leg's knee earns reward — in a
    normal walk that is one leg at a time, which drives an alternating bipedal
    gait that lifts the foot with the knee rather than tip-toeing off the ankle.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    air_time = contact_sensor.data.current_air_time
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos

    reward = torch.zeros(env.num_envs, device=asset.device)
    for foot_name, knee_name in _FOOT_KNEE_PAIRS:
        foot_idx = contact_sensor.find_bodies(foot_name)[0][0]
        knee_idx = asset.find_joints(knee_name)[0][0]
        in_swing = (air_time[:, foot_idx] > 0.0).float()
        flex = torch.clamp((joint_pos[:, knee_idx] - default_pos[:, knee_idx]).abs(), max=max_flex)
        reward = reward + flex * in_swing
    return reward


def feet_flat_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalise the foot soles not being parallel to the ground (anti tip-toe).

    Gravity expressed in each foot frame has zero xy-component when the sole is
    flat; a pitched/rolled foot grows the xy magnitude. Sum of squared xy over
    both feet -> larger when the robot points its toes.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]                      # (N, n, 4)
    grav = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, quat.shape[1], -1)  # (N, n, 3)
    proj_grav = quat_apply_inverse(quat, grav)                               # (N, n, 3)
    return torch.sum(torch.square(proj_grav[..., :2]), dim=(1, 2))


# ---------------------------------------------------------------------------
# Assistive force — decaying upward push to help robot discover standing
# Called as an event every step, force decays linearly to zero by max_iter
# ---------------------------------------------------------------------------

def apply_knee_assist_force(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    force_z: float = 20.0,
    max_iter: int = 3000,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply upward force to knee/shank bodies — creates rotational moment to tip hips up.
    Decays to zero by max_iter same as main assist force.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_iter = getattr(env, "common_step_counter", 0) // env.max_episode_length
    decay = max(0.0, 1.0 - current_iter / max_iter)
    actual_force = force_z * decay

    if actual_force < 1.0:
        return

    num_envs = len(env_ids) if env_ids is not None else env.scene.num_envs
    if env_ids is None:
        env_ids = torch.arange(num_envs, device=asset.device)

    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 1
    forces = torch.zeros(num_envs, num_bodies, 3, device=asset.device)
    torques = torch.zeros(num_envs, num_bodies, 3, device=asset.device)
    forces[:, :, 2] = actual_force  # Z = upward in world frame

    asset.set_external_force_and_torque(
        forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
    )


def ankle_movement_penalty(env, asset_cfg=None):
    """Penalise ankle joint velocity — stops twitching exploit."""
    asset = env.scene["robot"]
    joint_vel = asset.data.joint_vel
    left_ankle_pitch_idx = asset.find_joints("Left_Ankle_Pitch")[0][0]
    left_ankle_roll_idx = asset.find_joints("Left_Ankle_Roll")[0][0]
    right_ankle_pitch_idx = asset.find_joints("Right_Ankle_Pitch")[0][0]
    right_ankle_roll_idx = asset.find_joints("Right_Ankle_Roll")[0][0]
    ankle_vel = (
        torch.square(joint_vel[:, left_ankle_pitch_idx]) +
        torch.square(joint_vel[:, left_ankle_roll_idx]) +
        torch.square(joint_vel[:, right_ankle_pitch_idx]) +
        torch.square(joint_vel[:, right_ankle_roll_idx])
    )
    return -ankle_vel


def apply_hip_flexion_torque(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    torque_y: float = 60.0,
    max_iter: int = 3000,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply hip flexion torque around Y axis — makes hips want to flex and lift torso.
    Strong enough to straighten from prone without full assistance. Decays over training.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_iter = getattr(env, "common_step_counter", 0) // env.max_episode_length
    decay = max(0.0, 1.0 - current_iter / max_iter)
    actual_torque = torque_y * decay

    if actual_torque < 1.0:
        return

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # Resolve body IDs manually — SceneEntityCfg.body_ids may not be resolved
    body_ids = [asset.find_bodies("Left_Hip_Pitch")[0][0],
                asset.find_bodies("Right_Hip_Pitch")[0][0]]
    num_bodies = len(body_ids)
    forces = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    torques = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    torques[:, :, 1] = actual_torque

    asset.set_external_force_and_torque(
        forces, torques, env_ids=env_ids, body_ids=body_ids
    )


def apply_assist_force(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    force_z: float = 150.0,
    max_iter: int = 3000,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply a decaying upward assistive force to the head.
    Full force at iteration 0, linearly decays to zero at max_iter.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_iter = getattr(env, "common_step_counter", 0) // env.max_episode_length
    decay = max(0.0, 1.0 - current_iter / max_iter)
    actual_force = force_z * decay

    if actual_force < 1.0:
        return

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # Resolve body ID manually — find Head_2 explicitly
    body_ids = [asset.find_bodies("Head_2")[0][0]]
    num_bodies = len(body_ids)
    forces = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    torques = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    forces[:, :, 2] = actual_force  # Z = upward in world frame

    asset.set_external_force_and_torque(
        forces, torques, env_ids=env_ids, body_ids=body_ids
    )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def knee_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward knee joint velocity — encourages active knee bending during walking."""
    asset: Articulation = env.scene[asset_cfg.name]
    knee_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.sum(torch.clamp(knee_vel, max=5.0), dim=1)


def knee_bend_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min_bend: float = 0.2) -> torch.Tensor:
    """Reward when at least one knee is bent beyond min_bend radians."""
    asset: Articulation = env.scene[asset_cfg.name]
    knee_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    max_bend = torch.max(torch.abs(knee_pos), dim=1)[0]
    return torch.clamp(max_bend - min_bend, min=0.0)


def joint_pos_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target: float = 0.3) -> torch.Tensor:
    """Penalise deviation from a target joint position."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos - target), dim=1)


def getup_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_height: float = 0.52) -> torch.Tensor:
    """Reward trunk height only when at least one foot is below trunk."""
    asset: Articulation = env.scene[asset_cfg.name]
    trunk_height = asset.data.root_pos_w[:, 2]
    body_pos = asset.data.body_pos_w
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]
    left_foot_height = body_pos[:, left_foot_idx, 2]
    right_foot_height = body_pos[:, right_foot_idx, 2]
    min_foot_height = torch.min(left_foot_height, right_foot_height)
    feet_below = (min_foot_height < trunk_height - 0.1).float()
    height_reward = torch.exp(-3.0 * torch.abs(trunk_height - target_height))
    return height_reward * feet_below


def hip_height_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_height: float = 0.45) -> torch.Tensor:
    """Reward hip height specifically."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w
    left_hip_idx = asset.find_bodies("Left_Hip_Pitch")[0][0]
    right_hip_idx = asset.find_bodies("Right_Hip_Pitch")[0][0]
    hip_height = (body_pos[:, left_hip_idx, 2] + body_pos[:, right_hip_idx, 2]) / 2.0
    return torch.exp(-3.0 * torch.abs(hip_height - target_height))


def head_height_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_height: float = 0.75) -> torch.Tensor:
    """Reward head height — only kicks in when hips are already high."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w
    head_idx = asset.find_bodies("Head_2")[0][0]
    left_hip_idx = asset.find_bodies("Left_Hip_Pitch")[0][0]
    right_hip_idx = asset.find_bodies("Right_Hip_Pitch")[0][0]
    head_height = body_pos[:, head_idx, 2]
    hip_height = (body_pos[:, left_hip_idx, 2] + body_pos[:, right_hip_idx, 2]) / 2.0
    hips_up = (hip_height > 0.3).float()
    return torch.exp(-3.0 * torch.abs(head_height - target_height)) * hips_up


def hip_height_staged(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_height: float = 0.45) -> torch.Tensor:
    """Hip height reward that reduces once hips are above 0.3m."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w
    left_hip_idx = asset.find_bodies("Left_Hip_Pitch")[0][0]
    right_hip_idx = asset.find_bodies("Right_Hip_Pitch")[0][0]
    hip_height = (body_pos[:, left_hip_idx, 2] + body_pos[:, right_hip_idx, 2]) / 2.0
    below = (hip_height < 0.3).float()
    above = (hip_height >= 0.3).float()
    reward = below * torch.exp(-3.0 * torch.abs(hip_height - target_height))
    reward += above * 0.2
    return reward


def height_velocity_reward(env, asset_cfg=None):
    """Reward upward velocity of trunk."""
    asset = env.scene["robot"]
    trunk_vel_z = asset.data.root_lin_vel_w[:, 2]
    return torch.clamp(trunk_vel_z, min=0.0)


def standing_stability_reward(env, asset_cfg=None):
    """Reward low velocity when standing — stay still once up."""
    asset = env.scene["robot"]
    body_pos = asset.data.body_pos_w
    head_idx = asset.find_bodies("Head_2")[0][0]
    head_h = body_pos[:, head_idx, 2]
    standing = (head_h > 0.5).float()
    lin_vel = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)
    ang_vel = torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)
    stability = torch.exp(-2.0 * (lin_vel + ang_vel))
    return stability * standing


def head_height_linear(env, asset_cfg=None):
    """Reward head height linearly — higher is always better, no ceiling."""
    asset = env.scene["robot"]
    body_pos = asset.data.body_pos_w
    head_idx = asset.find_bodies("Head_2")[0][0]
    return body_pos[:, head_idx, 2]


def feet_low_penalty(env, asset_cfg=None):
    """Penalise feet above 0.06m only when head is above 0.4m."""
    asset = env.scene["robot"]
    body_pos = asset.data.body_pos_w
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]
    head_idx = asset.find_bodies("Head_2")[0][0]
    left_h = body_pos[:, left_foot_idx, 2]
    right_h = body_pos[:, right_foot_idx, 2]
    head_h = body_pos[:, head_idx, 2]
    head_up = (head_h > 0.4).float()
    penalty = torch.clamp(left_h - 0.06, min=0.0) + torch.clamp(right_h - 0.06, min=0.0)
    return penalty * head_up


def torso_pitch_up_reward(env, asset_cfg=None):
    """Reward torso pitch rotation magnitude — works front AND back spawn."""
    from isaaclab.utils.math import quat_apply_inverse
    asset = env.scene["robot"]
    quat = asset.data.root_quat_w
    ang_vel_world = asset.data.root_ang_vel_w
    ang_vel_local = quat_apply_inverse(quat, ang_vel_world)
    pitch_vel = torch.abs(ang_vel_local[:, 1])
    return pitch_vel


def symmetrical_legs_reward(env, asset_cfg=None):
    """Reward symmetrical leg joint positions."""
    asset = env.scene["robot"]
    joint_pos = asset.data.joint_pos
    left_hip_pitch_idx = asset.find_joints("Left_Hip_Pitch")[0][0]
    right_hip_pitch_idx = asset.find_joints("Right_Hip_Pitch")[0][0]
    left_knee_idx = asset.find_joints("Left_Knee_Pitch")[0][0]
    right_knee_idx = asset.find_joints("Right_Knee_Pitch")[0][0]
    left_hip_roll_idx = asset.find_joints("Left_Hip_Roll")[0][0]
    right_hip_roll_idx = asset.find_joints("Right_Hip_Roll")[0][0]
    hip_diff = torch.square(joint_pos[:, left_hip_pitch_idx] - joint_pos[:, right_hip_pitch_idx])
    knee_diff = torch.square(joint_pos[:, left_knee_idx] - joint_pos[:, right_knee_idx])
    roll_diff = torch.square(joint_pos[:, left_hip_roll_idx] + joint_pos[:, right_hip_roll_idx])
    return -(hip_diff + knee_diff + roll_diff)


def feet_stationary_penalty(env, asset_cfg=None):
    """Penalise feet moving horizontally — anchor them so CoM must pass over."""
    asset = env.scene["robot"]
    body_pos = asset.data.body_pos_w
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]
    if not hasattr(env, "_foot_anchor_l"):
        env._foot_anchor_l = body_pos[:, left_foot_idx, :2].clone()
        env._foot_anchor_r = body_pos[:, right_foot_idx, :2].clone()
    if hasattr(env, "episode_length_buf"):
        reset_mask = (env.episode_length_buf == 0)
        env._foot_anchor_l[reset_mask] = body_pos[reset_mask, left_foot_idx, :2]
        env._foot_anchor_r[reset_mask] = body_pos[reset_mask, right_foot_idx, :2]
    left_drift = torch.sum(torch.square(body_pos[:, left_foot_idx, :2] - env._foot_anchor_l), dim=1)
    right_drift = torch.sum(torch.square(body_pos[:, right_foot_idx, :2] - env._foot_anchor_r), dim=1)
    return -(left_drift + right_drift)


def lateral_velocity_penalty(env, asset_cfg=None, max_lateral: float = 0.3):
    """Penalise lateral (Y) and backward (X) velocity above threshold in robot local frame.
    Humanoids should move forward — penalise crabbing and reversing above 0.3m/s.
    """
    from isaaclab.utils.math import quat_apply_inverse
    asset = env.scene["robot"]
    quat = asset.data.root_quat_w
    lin_vel_world = asset.data.root_lin_vel_w[:, :3]
    lin_vel_local = quat_apply_inverse(quat, lin_vel_world)
    # Penalise lateral velocity exceeding threshold
    lateral_excess = torch.abs(lin_vel_local[:, 1]) - max_lateral
    # Penalise backward velocity exceeding threshold
    backward_excess = torch.clamp(-lin_vel_local[:, 0], min=0.0) - max_lateral
    return -torch.clamp(lateral_excess, min=0.0) - torch.clamp(backward_excess, min=0.0)


def penalty_body_yaw_local(env, asset_cfg=None):
    """Penalise yaw in robot LOCAL frame — spinning around own vertical axis.
    Never part of a legitimate getup, always a shortcut.
    """
    from isaaclab.utils.math import quat_apply_inverse
    asset = env.scene["robot"]
    ang_vel_world = asset.data.root_ang_vel_w
    quat = asset.data.root_quat_w
    ang_vel_local = quat_apply_inverse(quat, ang_vel_world)
    yaw_vel = torch.square(ang_vel_local[:, 2])
    return -yaw_vel


def hip_too_low_penalty(env, asset_cfg=None):
    """Only penalise hips below 0.1m — stops lying completely flat."""
    asset = env.scene["robot"]
    body_pos = asset.data.body_pos_w
    left_hip_idx = asset.find_bodies("Left_Hip_Pitch")[0][0]
    right_hip_idx = asset.find_bodies("Right_Hip_Pitch")[0][0]
    hip_h = (body_pos[:, left_hip_idx, 2] + body_pos[:, right_hip_idx, 2]) / 2.0
    deficit = torch.clamp(0.1 - hip_h, min=0.0)
    return -(deficit ** 2) * 100.0


def head_velocity_up_grounded(env, asset_cfg=None):
    """Reward upward head velocity ONLY when at least one foot is on the ground."""
    asset = env.scene["robot"]
    body_pos = asset.data.body_pos_w
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]
    left_h = body_pos[:, left_foot_idx, 2]
    right_h = body_pos[:, right_foot_idx, 2]
    foot_grounded = ((left_h < 0.05) | (right_h < 0.05)).float()
    head_vel_z = asset.data.root_lin_vel_w[:, 2]
    return torch.clamp(head_vel_z, min=0.0) * foot_grounded


def feet_off_ground_penalty(env, asset_cfg=None):
    """Penalise both feet being off the ground."""
    asset = env.scene["robot"]
    body_pos = asset.data.body_pos_w
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]
    left_h = body_pos[:, left_foot_idx, 2]
    right_h = body_pos[:, right_foot_idx, 2]
    left_penalty = torch.clamp(left_h - 0.05, min=0.0)
    right_penalty = torch.clamp(right_h - 0.05, min=0.0)
    return -(left_penalty + right_penalty)


def straight_arms_penalty(env, asset_cfg=None):
    """Penalise bent elbows."""
    asset = env.scene["robot"]
    joint_pos = asset.data.joint_pos
    left_elbow_idx = asset.find_joints("Left_Elbow_Pitch")[0][0]
    right_elbow_idx = asset.find_joints("Right_Elbow_Pitch")[0][0]
    left_bend = torch.square(joint_pos[:, left_elbow_idx])
    right_bend = torch.square(joint_pos[:, right_elbow_idx])
    return -(left_bend + right_bend)


def arm_pushoff_reward(env, asset_cfg=None):
    """Reward arms pushing off ground when hands are low."""
    asset = env.scene["robot"]
    body_pos = asset.data.body_pos_w
    joint_vel = asset.data.joint_vel
    left_hand_idx = asset.find_bodies("left_hand_link")[0][0]
    right_hand_idx = asset.find_bodies("right_hand_link")[0][0]
    left_hand_h = body_pos[:, left_hand_idx, 2]
    right_hand_h = body_pos[:, right_hand_idx, 2]
    left_grounded = (left_hand_h < 0.2).float()
    right_grounded = (right_hand_h < 0.2).float()
    left_shoulder_idx = asset.find_joints("ALeft_Shoulder_Pitch")[0][0]
    right_shoulder_idx = asset.find_joints("ARight_Shoulder_Pitch")[0][0]
    left_push = torch.abs(joint_vel[:, left_shoulder_idx]) * left_grounded
    right_push = torch.abs(joint_vel[:, right_shoulder_idx]) * right_grounded
    return left_push + right_push
