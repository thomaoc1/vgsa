import torch
from attacks_on_drl.attacker import StrategicallyTimedAttacker
from attacks_on_drl.attacker.common import BaseAttacker
from attacks_on_drl.attacker.critical_point_attack import CriticalPointAttack
from attacks_on_drl.attacker.critical_point_attack.divergence import AtariDivergenceFunction
from attacks_on_drl.victim.common import BaseVictim
from stable_baselines3.common.vec_env import VecEnv

from src.attacker.rollout_helper.enc_obs_rollout_helper import EncObsRolloutHelper
from src.attacker.rollout_helper.obs_rollout_helper import ObsRolloutHelper
from src.attacker.rollout_helper.ram_rollout_helper import RamRolloutHelper
from src.attacker.vgsa import VGSAAttacker
from src.prediction_model.model import ObsPredictionModel, PolicyEncodingPredictionModel, RamPredictionModel
from src.util.config.definitions import AttackerConfig
from src.util.path_builder import PredictionModelPaths
from src.util.sb3_env import StackedAtariRamVecWrapper


def init_vgsa(
    victim: BaseVictim,
    attacker_cfg: AttackerConfig,
    n_actions: int,
    prediction_model_path_builder: PredictionModelPaths,
):

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
        is_encoded=attacker_cfg.is_encoded,
    )


def init_cpa(
    attacker_cfg: AttackerConfig,
    victim: BaseVictim,
    env: StackedAtariRamVecWrapper,
    n_actions: int,
    prediction_model_path_builder: PredictionModelPaths,
):

    env_name: str | None = env.get_attr("spec")[0].id
    if not env_name:
        raise ValueError("Could not get environment name from spec.id")
        
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
    attacker_cfg: AttackerConfig,
    victim: BaseVictim,
    env: VecEnv | StackedAtariRamVecWrapper,
    env_n_actions: int,
    prediction_model_path_builder: PredictionModelPaths,
) -> BaseAttacker:

    if attacker_cfg.name.lower() == "vgsa":
        attacker = init_vgsa(
            attacker_cfg=attacker_cfg,
            victim=victim,
            n_actions=env_n_actions,
            prediction_model_path_builder=prediction_model_path_builder,
        )
    elif attacker_cfg.name.lower() == "cpa":
        if isinstance(env, StackedAtariRamVecWrapper):
            attacker = init_cpa(
                attacker_cfg=attacker_cfg,
                victim=victim,
                env=env,
                n_actions=env_n_actions,
                prediction_model_path_builder=prediction_model_path_builder,
            )
        else:
            raise ValueError("For CPA, environment must be wrapped in StackedAtariRamVecWrapper.")
            
    elif attacker_cfg.name.lower() == "sta":
        attacker = StrategicallyTimedAttacker(victim=victim, attack_threshold=attacker_cfg.attack_threshold)
    else:
        raise ValueError(f"Unknown attacker: {attacker_cfg.name}")

    return attacker
