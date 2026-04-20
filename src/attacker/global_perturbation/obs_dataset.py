from typing import Optional
from torch.utils.data import Dataset
import random

from src.prediction_model.data_collection.sas_dataset import TransitionDataset


class ObservationDataset(Dataset):
    def __init__(self, sas_dataset: TransitionDataset, dataset_size: Optional[int] = None):
        episodes = []
        sample_map = []

        for ep_idx, ep in enumerate(sas_dataset.episodes):
            episodes.append({"states": ep["states"]})
            T = len(ep["states"])
            sample_map.extend((ep_idx, t) for t in range(T))

        if dataset_size is not None and dataset_size < len(sample_map):
            sample_map = random.sample(sample_map, dataset_size)

        self.episodes = episodes
        self.sample_map = sample_map

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx):
        ep_idx, t = self.sample_map[idx]
        return self.episodes[ep_idx]["states"][t]