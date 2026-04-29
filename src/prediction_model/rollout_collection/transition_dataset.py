import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    def __init__(self, n_actions: int, episodes: list | None = None, sample_map: list | None = None):
        assert (episodes and sample_map) or not (episodes or sample_map), "Both or neither must be provided (XOR)"

        self.sample_map = [] if not sample_map else sample_map
        self.episodes = [] if not episodes else episodes

        self.n_actions = n_actions
        self.lookahead_horizon = 1

    def add_episode(
        self,
        states: list[np.ndarray],
        actions: list[int],
    ):
        episode = {
            "states": torch.stack([torch.from_numpy(state) for state in states]),
            "actions": torch.tensor(actions),
        }
        self.episodes.append(episode)

        T = len(states)
        episode_idx = len(self.episodes) - 1
        self.sample_map.extend(
            (episode_idx, t)
            for t in range(T - 1)  # Cannot take action in terminal state
        )

    def __len__(self):
        return len(self.sample_map)

    def _pad_tensor(self, padee):
        pad_length = self.lookahead_horizon - len(padee)
        repeated_value = padee[-1].unsqueeze(0)
        padded_padee = torch.concat([padee, repeated_value.repeat_interleave(pad_length, dim=0)])
        return padded_padee

    def __getitem__(self, idx):
        episode_idx, t = self.sample_map[idx]
        episode = self.episodes[episode_idx]

        current_state = episode["states"][t]
        next_states = episode["states"][t + 1 : t + 1 + self.lookahead_horizon]
        actions = episode["actions"][t : t + self.lookahead_horizon]

        if len(next_states) < self.lookahead_horizon:
            next_states = self._pad_tensor(next_states)
            actions = self._pad_tensor(actions)

        return current_state, actions, next_states

    def subset(self, indices: list[int]) -> "TransitionDataset":
        subset_sample_map = [self.sample_map[i] for i in indices]
        subset = TransitionDataset(self.n_actions, episodes=self.episodes, sample_map=subset_sample_map)
        subset.lookahead_horizon = self.lookahead_horizon
        return subset

    def save(self, path):
        torch.save(
            {
                "episodes": self.episodes,
                "sample_map": self.sample_map,
            },
            path,
        )

    def get_stacked_states(self):
        return torch.stack([state for episode in self.episodes for state in episode["states"]])

    @staticmethod
    def load(n_actions: int, path: str):
        data = torch.load(path, weights_only=True)
        return TransitionDataset(n_actions, **data)
