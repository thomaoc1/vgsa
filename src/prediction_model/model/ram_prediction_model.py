import torch
import torch.nn as nn


class RamPredictionModel(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.n_values = 256
        self.state_dim = 128

        hidden_dim = 256
        self._lstm = nn.LSTM(self.state_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)

        self._fc = nn.Sequential(
            nn.Linear(hidden_dim + n_actions, 512), nn.ReLU(), nn.Linear(512, self.state_dim * self.n_values)
        )

    def forward(self, state, action, hidden=None):
        _, hidden = self._lstm(state, hidden)
        last_hidden = hidden[0][-1]
        x = torch.concat([last_hidden, action], dim=-1)
        logits = self._fc(x)
        logits = logits.view(-1, self.state_dim, self.n_values)
        return logits, hidden
