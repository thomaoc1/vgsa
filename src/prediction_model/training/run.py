from dataclasses import asdict, dataclass

import hydra
from omegaconf import DictConfig

from src.prediction_model.rollout_collection.transition_dataset import TransitionDataset
from src.util.config.definitions import EnvConfig, PolicyConfig, PredictionModelConfig
from src.util.config.paths import CONFIG_PATH
from src.util.path_builder import DatasetPaths, PredictionModelPaths
from src.util.set_global_seed import set_global_seed

from .util import init_loaders, init_prediction_model, init_prediction_model_trainer, split_dataset


@dataclass
class PredictionModelTrainerConfig:
    gym_env: EnvConfig
    next_state_pm: PredictionModelConfig
    policy: PolicyConfig
    trainer: str
    dataset_name: str
    victim_policy: bool = True
    lr: float = 1e-4
    lookahead_horizon: int = 1
    epochs: int = 100
    batch_size: int = 32
    load: bool = False


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_pm")
def main(cfg: DictConfig):
    pm_trainer_cfg = PredictionModelTrainerConfig(**cfg)  # pyright: ignore[reportCallIssue]
    pm_trainer_cfg.gym_env = EnvConfig(**cfg.gym_env)
    pm_trainer_cfg.next_state_pm = PredictionModelConfig(**cfg.next_state_pm)
    pm_trainer_cfg.policy = PolicyConfig(**cfg.policy)

    is_encoded = pm_trainer_cfg.next_state_pm.model_type == "PolicyEncodingPredictor"
    prediction_model_path_builder = PredictionModelPaths(
        pm_trainer_cfg.policy.name,
        pm_trainer_cfg.gym_env.name,
        seed=pm_trainer_cfg.next_state_pm.seed,
        agent_seed=pm_trainer_cfg.policy.seed,
        run_name=pm_trainer_cfg.next_state_pm.unique_name,
        encoded=is_encoded,
    )

    set_global_seed(pm_trainer_cfg.next_state_pm.seed)

    dataset_path_builder = DatasetPaths(
        pm_trainer_cfg.policy.name,
        pm_trainer_cfg.gym_env.name,
        encoded=is_encoded,
        agent_seed=pm_trainer_cfg.policy.seed,
    )

    is_ram = pm_trainer_cfg.next_state_pm.model_type == "RAMPredictionModelTrainer"
    dataset_file_path = dataset_path_builder.ram_train_file if is_ram else dataset_path_builder.train_file

    dataset = TransitionDataset.load(pm_trainer_cfg.gym_env.n_actions, dataset_file_path)

    prediction_model = init_prediction_model(
        prediction_model_path_builder,
        pm_trainer_cfg.next_state_pm,
        n_actions=pm_trainer_cfg.gym_env.n_actions,
        load=pm_trainer_cfg.load,
    )

    trainer = init_prediction_model_trainer(
        prediction_model_path_builder,
        prediction_model,
        asdict(pm_trainer_cfg),
        load=pm_trainer_cfg.load,
    )

    trainer.train(
        *init_loaders(
            *split_dataset(dataset),
            pm_trainer_cfg.lookahead_horizon,
            pm_trainer_cfg.batch_size,
        ),
        epochs=pm_trainer_cfg.epochs,
        lr=pm_trainer_cfg.lr,
    )


if __name__ == "__main__":
    main()
