import gymnasium as gym
from .getup_env_cfg import K1GetupEnvCfg, K1GetupEnvCfg_PLAY

gym.register(
    id="Booster-K1-Getup-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1GetupEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:K1GetupPPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Getup-v1-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1GetupEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:K1GetupPPORunnerCfg",
    },
)
