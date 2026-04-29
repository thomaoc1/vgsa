from typing import Any

import torch

import src.prediction_model.model as model_module
from src.prediction_model.training.trainers.common import PredictionModelTrainer
from src.util.config.definitions import PredictionModelConfig
from src.util.logger.wandb_logger import WandbLogger
from src.util.path_builder import PredictionModelPaths


def prediction_model_training_summary(cfg: dict[str, Any]) -> dict:
    """Extract fields that determine reproducibility of a prediction model training run.

    Drops: policy agent_parameters (victim training details), dataset_name (derivative),
    gym_env geometry, model_kwargs (usually empty), suffix/unique_name (labels).
    """
    return {
        "env": cfg["gym_env"]["name"],
        "policy": cfg["policy"]["name"],
        "policy_seed": cfg["policy"]["seed"],
        "model_type": cfg["next_state_pm"]["model_type"],
        "model_seed": cfg["next_state_pm"]["seed"],
        "trainer": cfg["trainer"],
        "lr": cfg["lr"],
        "lookahead_horizon": cfg["lookahead_horizon"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "teacher_forcing": not cfg["load"],
    }


def init_prediction_model(
    path_builder: PredictionModelPaths,
    model_cfg: PredictionModelConfig,
    n_actions: int,
    load: bool,
) -> torch.nn.Module:
    prediction_model = getattr(model_module, model_cfg.model_type)(n_actions=n_actions)

    if load:
        prediction_model.load_state_dict(
            torch.load(
                path_builder.model_weights(),
                weights_only=True,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
        )

    return prediction_model


def init_prediction_model_trainer(
    prediction_model_path_builder: PredictionModelPaths,
    prediction_model: torch.nn.Module,
    trainer_cfg,
    load: bool,
) -> PredictionModelTrainer:
    import src.prediction_model.training.trainers as trainers

    logger = WandbLogger(
        experiment_group="train_prediction_model",
        config=prediction_model_training_summary(trainer_cfg),
        summary_metrics={"train_loss": ["min"], "val_loss": ["min"]},
    )

    return getattr(trainers, trainer_cfg["trainer"])(
        model=prediction_model,
        num_actions=trainer_cfg["gym_env"]["n_actions"],
        prediction_model_path_builder=prediction_model_path_builder,
        teacher_forcing=not load,
        logger=logger,
    )
