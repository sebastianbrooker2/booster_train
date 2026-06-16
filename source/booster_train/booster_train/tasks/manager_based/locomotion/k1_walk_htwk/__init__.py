import gymnasium as gym
from .walk_htwk_env_cfg import K1WalkHTWKEnvCfg, K1WalkHTWKEnvCfg_PLAY

gym.register(
    id="Booster-K1-WalkHTWK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1WalkHTWKEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:K1WalkHTWKPPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-WalkHTWK-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1WalkHTWKEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:K1WalkHTWKPPORunnerCfg",
    },
)

from .script_env_cfg import K1WalkHTWKScriptEnvCfg, K1WalkHTWKScriptFastEnvCfg

gym.register(
    id="Booster-K1-WalkHTWK-Script",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1WalkHTWKScriptEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:K1WalkHTWKPPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-WalkHTWK-Script-Fast",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": K1WalkHTWKScriptFastEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:K1WalkHTWKPPORunnerCfg",
    },
)
