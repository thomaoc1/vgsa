from dataclasses import asdict, dataclass

import ale_py
import gymnasium as gym
import hydra
from attacks_on_drl.runner import AttackRunner
from omegaconf import DictConfig

from src.run_attack.util import init_attacker
from src.util.agent import init_agent
from src.util.config.definitions import AttackerConfig, EnvConfig, PolicyConfig
from src.util.config.paths import CONFIG_PATH
from src.util.logger.config_summary import run_attack_summary
from src.util.logger.wandb_logger import WandbLogger
from src.util.path_builder import PolicyPaths, PredictionModelPaths
from src.util.sb3_env import init_env
from src.util.set_global_seed import set_global_seed
from src.victim.enc_actor_critic_victim import EncActorCriticVictim
from src.victim.enc_dqn_victim import EncDQNVictim

gym.register_envs(ale_py)
SEED = 101

ATTACK_SUMMARY_METRICS = {
    "ep_rew": ["mean"],
    "n_attacks": ["mean"],
}


@dataclass
class RunAttackConfig:
    gym_env: EnvConfig
    policy: PolicyConfig
    attacker: AttackerConfig
    episode_max_frames: int | float = float("+inf")
    n_episodes: int = 100


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="run_attack")
def main(cfg: DictConfig):
    set_global_seed(101)
    run_attack_cfg = RunAttackConfig(**cfg)  # pyright: ignore[reportCallIssue]
    run_attack_cfg.gym_env = EnvConfig(**cfg.gym_env)
    run_attack_cfg.policy = PolicyConfig(**cfg.policy)
    run_attack_cfg.attacker = AttackerConfig(**cfg.attacker)

    prediction_model_path_builder = PredictionModelPaths(
        run_attack_cfg.policy.name,
        run_attack_cfg.gym_env.name,
        agent_seed=run_attack_cfg.policy.seed,
        encoded=run_attack_cfg.attacker.is_encoded,
    )

    policy_path_builder = PolicyPaths(
        run_attack_cfg.policy.name,
        run_attack_cfg.gym_env.name,
        seed=run_attack_cfg.policy.seed,
    )

    assert run_attack_cfg.gym_env.n_envs == 1, "Attack only works on single env"

    atari_wrapper_args = {
        "terminal_on_life_loss": run_attack_cfg.gym_env.name == "BreakoutNoFrameskip-v4",
        "clip_reward": False,
    }
    env = init_env(
        run_attack_cfg.gym_env,
        atari_wrapper_args=atari_wrapper_args,
        is_ram_env=run_attack_cfg.attacker.uses_ram,
        seed=SEED,
    )

    agent = init_agent(sb3_cfg=run_attack_cfg.policy, path_builder=policy_path_builder)
    if run_attack_cfg.policy.name.lower() == "dqn":
        victim = EncDQNVictim(agent)
    else:
        victim = EncActorCriticVictim(agent)

    attacker = init_attacker(
        attacker_cfg=run_attack_cfg.attacker,
        victim=victim,
        env=env,
        env_n_actions=run_attack_cfg.gym_env.n_actions,
        prediction_model_path_builder=prediction_model_path_builder,
    )

    logger = WandbLogger(
        experiment_group="run_attack",
        config=run_attack_summary(asdict(run_attack_cfg)),
    )

    runner = AttackRunner(
        env,
        attacker,
        victim,
        episode_max_frames=run_attack_cfg.episode_max_frames,
    )

    try:
        result = runner.run(run_attack_cfg.n_episodes)
        logger.log(asdict(result))
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
