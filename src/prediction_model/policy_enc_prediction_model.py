import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyEncodingPredictionModel(nn.Module):
    def __init__(self, n_actions: int, latent_dim: int = 512, hidden_dim: int = 512):
        super().__init__()

        self.action_embedding = nn.Linear(n_actions, hidden_dim, bias=False)

        self.fc1 = nn.Linear(latent_dim + hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent, action):
        a_emb = self.action_embedding(action)
        x = torch.cat([latent, a_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_latent = self.fc3(x)
        return next_latent
