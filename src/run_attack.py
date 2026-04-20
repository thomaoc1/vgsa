from dataclasses import dataclass

import ale_py
import gymnasium as gym
import hydra
import torch
from attacks_on_drl.attacker import StrategicallyTimedAttacker
from attacks_on_drl.attacker.common.base_attacker import BaseAttacker
from attacks_on_drl.attacker.critical_point_attack import CriticalPointAttack
from attacks_on_drl.attacker.critical_point_attack.divergence import AtariDivergenceFunction
from attacks_on_drl.runner import AttackRunner
from attacks_on_drl.victim.common.base_victim import BaseVictim
from omegaconf import DictConfig

from src.attacker.rollout_helper.enc_obs_rollout_helper import EncObsRolloutHelper
from src.attacker.rollout_helper.obs_rollout_helper import ObsRolloutHelper
from src.attacker.rollout_helper.ram_rollout_helper import RamRolloutHelper
from src.attacker.vgsa import VGSAAttacker
from src.prediction_model.model.obs_prediction_model import ObsPredictionModel
from src.prediction_model.model.policy_enc_prediction_model import PolicyEncodingPredictionModel
from src.prediction_model.model.ram_prediction_model import RamPredictionModel
from src.util.agent import init_agent
from src.util.config.definitions import AttackerConfig, EnvConfig, PolicyConfig
from src.util.config.paths import CONFIG_PATH
from src.util.path_builder import PolicyPaths, PredictionModelPaths
from src.util.sb3_env import StackedAtariRamVecWrapper, init_env
from src.victim.enc_actor_critic_victim import EncActorCriticVictim
from src.victim.enc_dqn_victim import EncDQNVictim

gym.register_envs(ale_py)


@dataclass
class RunAttackConfig:
    gym_env: EnvConfig
    policy: PolicyConfig
    attacker: AttackerConfig
    debug_mode: bool = False
    episode_max_frames: int | float = float("+inf")
    n_episodes: int = 100


def init_vgsa(run_attack_cfg: RunAttackConfig, victim: BaseVictim, prediction_model_path_builder: PredictionModelPaths):
    attacker_cfg = run_attack_cfg.attacker
    n_actions = run_attack_cfg.gym_env.n_actions

    if attacker_cfg.is_encoded:
        obs_prediction_model = PolicyEncodingPredictionModel(n_actions)
        obs_prediction_model.load_state_dict(
            torch.load(prediction_model_path_builder.prediction_model_weights, map_location="cpu")
        )

        rollout_helper = EncObsRolloutHelper(
            enc_obs_prediction_model=obs_prediction_model,
            victim=victim,
            n_actions=n_actions,
            action_enum_len=attacker_cfg.rollout_helper.action_enum_len,
            baseline_obs_dist=attacker_cfg.rollout_helper.baseline_obs_dist,
        )
    else:
        obs_prediction_model = ObsPredictionModel(n_actions)
        obs_prediction_model.load_state_dict(
            torch.load(prediction_model_path_builder.prediction_model_weights, map_location="cpu")
        )

        rollout_helper = ObsRolloutHelper(
            obs_prediction_model=obs_prediction_model,
            victim=victim,
            n_actions=n_actions,
            action_enum_len=attacker_cfg.rollout_helper.action_enum_len,
            baseline_obs_dist=attacker_cfg.rollout_helper.baseline_obs_dist,
        )
    
    return VGSAAttacker(
        victim=victim,
        rollout_helper=rollout_helper,
        attack_threshold=attacker_cfg.attack_threshold,
    )


def init_cpa(
    run_attack_cfg: RunAttackConfig,
    env: StackedAtariRamVecWrapper,
    victim: BaseVictim,
    prediction_model_path_builder: PredictionModelPaths,
):
    attacker_cfg = run_attack_cfg.attacker
    n_actions = run_attack_cfg.gym_env.n_actions
    env_name = run_attack_cfg.gym_env.name

    obs_prediction_model = ObsPredictionModel(n_actions)
    obs_prediction_model.load_state_dict(
        torch.load(prediction_model_path_builder.prediction_model_weights, map_location="cpu")
    )

    ram_prediction_model = RamPredictionModel(n_actions)
    ram_prediction_model.load_state_dict(
        torch.load(prediction_model_path_builder.ram_prediction_model_weights, map_location="cpu")
    )

    rollout_helper = RamRolloutHelper(
        env=env,
        obs_prediction_model=obs_prediction_model,
        ram_prediction_model=ram_prediction_model,
        victim=victim,
        n_actions=n_actions,
        action_enum_len=attacker_cfg.rollout_helper.action_enum_len,
        baseline_obs_dist=attacker_cfg.rollout_helper.baseline_obs_dist,
    )
    divergence_function = AtariDivergenceFunction(env_name)

    return CriticalPointAttack(
        victim=victim,
        rollout_helper=rollout_helper,
        divergence_function=divergence_function,
        attack_threshold=attacker_cfg.attack_threshold,
    )


def init_attacker(
    env,
    run_attack_cfg: RunAttackConfig,
    prediction_model_path_builder: PredictionModelPaths,
    victim: BaseVictim,
) -> BaseAttacker:
    chosen_attacker = run_attack_cfg.attacker.name
    if chosen_attacker.lower() == "vgsa":
        attacker = init_vgsa(
            run_attack_cfg=run_attack_cfg, victim=victim, prediction_model_path_builder=prediction_model_path_builder
        )
    elif chosen_attacker.lower() == "cpa":
        attacker = init_cpa(
            run_attack_cfg=run_attack_cfg,
            env=env,
            victim=victim,
            prediction_model_path_builder=prediction_model_path_builder,
        )
    elif chosen_attacker.lower() == "sta":
        attacker = StrategicallyTimedAttacker(victim=victim, attack_threshold=run_attack_cfg.attacker.attack_threshold)
    else:
        raise ValueError(f"Unknown attacker: {chosen_attacker}")

    return attacker


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="run_attack")
def main(cfg: DictConfig):
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
        run_attack_cfg.gym_env, atari_wrapper_args=atari_wrapper_args, is_ram_env=run_attack_cfg.attacker.uses_ram
    )

    agent = init_agent(sb3_cfg=run_attack_cfg.policy, path_builder=policy_path_builder)
    if run_attack_cfg.policy.name.lower() == "dqn":
        victim = EncDQNVictim(agent)
    else:
        victim = EncActorCriticVictim(agent)

    attacker = init_attacker(env, run_attack_cfg, prediction_model_path_builder, victim)

    # Run Attack ehre
    runner = AttackRunner(env, attacker, victim)
    runner.run(100)


if __name__ == "__main__":
    main()
