from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.util.path_builder import PredictionModelPaths

from .common import FrameCycler
from .common.prediction_model_trainer import PredictionModelTrainer



class ObsPredictionModelTrainer(PredictionModelTrainer):
    def __init__(
        self,
        model,
        num_actions,
        prediction_model_path_builder: PredictionModelPaths,
        teacher_forcing: bool = True,
    ):
        super().__init__(model, num_actions, prediction_model_path_builder, teacher_forcing)
        self._frame_cycler = FrameCycler()

    def _run_epoch(
        self, loader: DataLoader, teacher_forcing_prob: float, optimiser: Optional[Optimizer] = None
    ) -> float:
        total_epoch_loss = 0.0
        for states, actions, next_states in loader:
            _, K = actions.shape
            k_losses = torch.zeros(K, device=self.device)
            k_loss_components = [0] * K

            states, actions, next_states = (
                states.to(self.device) / 255.0,
                actions.to(self.device).long(),
                next_states.to(self.device) / 255.0,
            )

            one_hot_actions = F.one_hot(actions, num_classes=self.n_actions).float()

            if optimiser:
                optimiser.zero_grad()

            for predictive_step in range(K):
                self._frame_cycler.save_current_state(states)
                current_one_hot_action, target_next_state = (
                    one_hot_actions[:, predictive_step],
                    next_states[:, predictive_step, -1].unsqueeze(1),
                )
                predicted_next_state = self.model(states, current_one_hot_action).unsqueeze(1)
                recon_mse = F.mse_loss(predicted_next_state, target_next_state, reduction="none")
                recon_mse = recon_mse.view(target_next_state.size(0), -1).sum(dim=1).mean()
                k_loss_components[predictive_step] = recon_mse.item()

                if not optimiser or torch.rand(1).item() > teacher_forcing_prob:
                    states = predicted_next_state
                else:
                    states = self._frame_cycler.cycle_frames(next_states[:, predictive_step, -1])

            loss = k_losses.mean()
            if optimiser:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1)
                optimiser.step()

            total_epoch_loss += sum(k_loss_components) / K
            
        return total_epoch_loss / len(loader)

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float = 1e-4
    ) -> Tuple[List, List]:
        """
        Main function for training and evaluating the model.
        """
        optimiser = Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

        lr_scheduler = ReduceLROnPlateau(optimiser, patience=5, threshold=1e-3)

        print(f"Training for {self.epochs} epochs with LR: {lr} and TF: {self.teacher_forcing}")
        for epoch in range(1, self.epochs + 1):
            teacher_forcing_prob = self.teacher_forcing_schedule(epoch, total_epochs=epochs)
            self.model.train()
            training_loss = self._run_epoch(train_loader, teacher_forcing_prob, optimiser=optimiser)
            self.train_losses.append(training_loss)

            self.model.eval()
            with torch.no_grad():
                validation_loss = self._run_epoch(val_loader, teacher_forcing_prob)

            self.validation_losses.append(validation_loss)
            lr_scheduler.step(validation_loss)

            print(
                f"Epoch {epoch} (LR: {lr_scheduler.get_last_lr()[-1]}, TF: {teacher_forcing_prob:.2f}):"
                f" \n\tTraining: {training_loss} \n\tValidation: {validation_loss}"
            )

        self.save()
        return self._get_results()
