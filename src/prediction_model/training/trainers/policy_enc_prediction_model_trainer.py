import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .common import PredictionModelTrainer


class PolicyEncodingPredictionModelTrainer(PredictionModelTrainer):
    def _run_epoch(self, loader: DataLoader, teacher_forcing_prob: float, optimiser: Optimizer | None = None):
        total_loss = 0

        for encoded_state, actions, encoded_next_states in loader:
            _, K = actions.shape
            k_losses = torch.zeros(K, device=self.device)

            encoded_state, one_hot_actions, encoded_next_states = (
                encoded_state.to(self.device),
                F.one_hot(actions.long(), num_classes=self.n_actions).float().to(self.device),
                encoded_next_states.to(self.device),
            )

            if optimiser:
                optimiser.zero_grad()

            for predictive_step in range(K):
                (current_one_hot_action,) = one_hot_actions[:, predictive_step]
                (current_target_encoded_next_state,) = encoded_next_states[:, predictive_step]

                predicted_next_encoding = self.model(encoded_state, current_one_hot_action)

                recon_mse = F.mse_loss(predicted_next_encoding, current_target_encoded_next_state, reduction="none")
                loss = recon_mse.view(encoded_state.size(0), -1).sum(dim=1).mean()
                k_losses[predictive_step] = loss

                if not optimiser or torch.rand(1).item() > teacher_forcing_prob:
                    encoded_state = predicted_next_encoding
                else:
                    encoded_state = current_target_encoded_next_state

            loss = k_losses.mean()
            total_loss += loss.item()

            if optimiser:
                loss.backward()
                optimiser.step()

        n_batches = len(loader)
        return total_loss / n_batches

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 30, lr: float = 1e-4) -> None:
        optimiser = Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=5)
        self.epochs = epochs

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            teacher_forcing_prob = self.teacher_forcing_schedule(epoch, total_epochs=epochs)
            train_loss = self._run_epoch(train_loader, teacher_forcing_prob, optimiser=optimiser)

            self.model.eval()
            with torch.no_grad():
                validation_loss = self._run_epoch(val_loader, teacher_forcing_prob=0)

            scheduler.step(validation_loss)

            current_lr = optimiser.param_groups[0]["lr"]
            print(
                f"Epoch {epoch} (LR: {current_lr:.6f}, TF: {teacher_forcing_prob}):"
                f"\n\tTraining: {train_loss:.3f}"
                f"\n\tValidation: {validation_loss:.3f}"
            )
            self._log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": validation_loss,
                    "lr": current_lr,
                    "teacher_forcing": teacher_forcing_prob,
                }
            )

        self.save()
