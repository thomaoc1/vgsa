import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .common import PredictionModelTrainer


class RAMPredictionModelTrainer(PredictionModelTrainer):
    def _iteration(
        self,
        loader: DataLoader,
        teacher_forcing_prob: float,
        optimizer: Optimizer | None = None,
    ):
        epoch_loss = 0.0
        for current_state, action, next_state, _, _ in loader:
            current_state, action_seq, next_state = (
                current_state.to(self.device).float(),
                action.to(self.device),
                next_state.to(self.device),
            )

            action_seq = F.one_hot(action_seq, num_classes=self.n_actions).float()

            _, K, _ = action_seq.shape
            state_stack = current_state

            predictions = []

            hidden = None
            for t in range(K):
                action_t = action_seq[:, t, :]
                pred_state, hidden = self.model(state_stack / 255.0, action_t, hidden)
                predictions.append(pred_state)

                if not optimizer or torch.rand(1).item() < teacher_forcing_prob:
                    next_true_state = next_state[:, t, -1].float()
                else:
                    next_true_state = pred_state.argmax(dim=-1).float()

                state_stack = next_true_state.unsqueeze(1)

            predictions = torch.stack(predictions, dim=1)
            n_classes = predictions.size(-1)
            logits_flat = predictions.reshape(-1, n_classes)
            target_flat = next_state[:, :, -1, :].reshape(-1)
            loss = F.cross_entropy(logits_flat, target_flat, reduction="mean")

            epoch_loss += loss.item()

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

        return epoch_loss / len(loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100, lr: float = 0.03) -> None:
        optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(epochs):
            teacher_forcing_prob = self.teacher_forcing_schedule(epoch, total_epochs=epochs)
            self.model.train()
            train_loss = self._iteration(train_loader, teacher_forcing_prob, optimizer)

            self.model.eval()
            val_loss = self._iteration(val_loader, teacher_forcing_prob=0)

            print(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f},"
                f" Teacher forcing prob = {teacher_forcing_prob:.4f}"
            )
            self._log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "teacher_forcing": teacher_forcing_prob,
                }
            )
            self.save()

    def save(self):
        print(f"Saving model to: {self.prediction_model_path_builder.ram_model_weights()}")
        torch.save(self.model.state_dict(), self.prediction_model_path_builder.ram_model_weights())
