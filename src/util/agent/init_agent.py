import stable_baselines3
from src.util.config.definitions import PolicyConfig
from src.util.path_builder import PolicyPaths


def init_agent(sb3_cfg: PolicyConfig, path_builder: PolicyPaths, scale_obs: bool = False, device: str = "cpu"):
    algo_cls = getattr(stable_baselines3, str(sb3_cfg.name))
    save_path = path_builder.sb3_run_path()
    agent = algo_cls.load(save_path, device=device)

    policy_kwargs = dict(agent.policy_kwargs)
    policy_kwargs["normalize_images"] = scale_obs

    agent = algo_cls.load(
        save_path,
        custom_objects=dict(
            policy_kwargs=policy_kwargs,
        ),
    )
    return agent
