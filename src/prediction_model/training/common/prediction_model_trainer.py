import math
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from src.util.logger.common.protocol import LoggerProtocol
from src.util.path_builder import PredictionModelPaths


class PredictionModelTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        num_actions: int,
        prediction_model_path_builder: PredictionModelPaths,
        teacher_forcing: bool = True,
        logger: LoggerProtocol | None = None,
    ):
        self.prediction_model_path_builder = prediction_model_path_builder
        self.n_actions = num_actions
        self.teacher_forcing = teacher_forcing
        self.save_name_prefix = "prediction_model"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.epochs: int | None = None
        self.logger = logger

        self.train_losses = []
        self.validation_losses = []

    def _init_base_filename(self, run_name: str):
        return f"{self.save_name_prefix}{f'_{run_name}' if run_name else ''}"

    def _get_results(self) -> tuple[list, list]:
        return self.train_losses, self.validation_losses

    def _log(self, step_result: dict) -> None:
        if self.logger is not None:
            self.logger.log(step_result)

    def save(self):
        print(f"Saving model to: {self.prediction_model_path_builder.model_weights()}")
        torch.save(self.model.state_dict(), self.prediction_model_path_builder.model_weights())

    def teacher_forcing_schedule(self, epoch: int, total_epochs: int):
        if self.teacher_forcing:
            return 0.5 * (1 + math.cos(min(epoch, total_epochs // 2) / (total_epochs // 2) * math.pi))
        else:
            return 0

    @abstractmethod
    def train(self, train_loader: DataLoader, val_loader: DataLoader, **kwargs) -> tuple[list, list]:
        pass
