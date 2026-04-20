
import torch
import torch.nn.functional as F
from attacks_on_drl.victim.common import BaseVictim
from torch.utils.data import DataLoader


class GlobalPerturbationGenerator:
    def __init__(
        self,
        policy: BaseVictim,
        n_actions: int,
        epsilon: float = 0.03,
        alpha: float = 1e-5,
        lr: float = 0.05,
        epochs: int = 10,
        device: str = "cuda",
    ):
        self.policy = policy
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.device = device

    def _train_single_mask(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        action_index: int,
    ) -> torch.Tensor:
        example_batch = next(iter(train_loader))
        state_example = example_batch.to(self.device)[0]
        delta = torch.zeros_like(state_example, requires_grad=True, device=self.device, dtype=torch.float32)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        print(f"Training global mask for action {action_index}...")

        for epoch in range(self.epochs):
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device) / 255.0
                perturbed = batch + delta.clamp(-self.epsilon, self.epsilon)

                logits = self.policy.get_action_logits(perturbed)
                log_probs = F.log_softmax(logits, dim=1)
                loss = -(log_probs[:, action_index].mean() + self.alpha * delta.norm(2))

                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    delta.clamp_(-self.epsilon, self.epsilon)

            val_batch = next(iter(val_loader)).to(self.device) / 255.0
            val_pert = val_batch + delta.clamp(-self.epsilon, self.epsilon)
            val_logits = self.policy.get_action_logits(val_pert)
            val_log_probs = F.log_softmax(val_logits, dim=1)
            val_loss = -(val_log_probs[:, action_index].mean()).item()
            print(
                f"action={action_index} epoch={epoch} "
                f"train_loss={train_loss / len(train_loader):.5f} val_loss={val_loss:.5f}"
            )

        return delta.detach()
    
    
    def generate(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[int, torch.Tensor]:
        masks = {}
        for a in range(self.n_actions):
            masks[a] = self._train_single_mask(train_loader, val_loader, a)
        return masks