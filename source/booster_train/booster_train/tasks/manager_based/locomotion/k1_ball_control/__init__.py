import gymnasium as gym
from .ball_control_env_cfg import K1BallControlEnvCfg, K1BallControlEnvCfg_PLAY

gym.register(
    id="Booster-K1-BallControl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1BallControlEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:K1BallControlPPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-BallControl-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1BallControlEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:K1BallControlPPORunnerCfg",
    },
)
