import torch
from torch.utils.data import DataLoader

from src.prediction_model.rollout_collection.transition_dataset import TransitionDataset


def split_dataset(dataset: TransitionDataset) -> tuple[TransitionDataset, TransitionDataset]:
    n_total = len(dataset)
    n_val = int(n_total * 0.2)
    indices = torch.randperm(n_total).tolist()
    train_dataset = dataset.subset(indices[: n_total - n_val])
    val_dataset = dataset.subset(indices[n_total - n_val :])
    return train_dataset, val_dataset


def init_loaders(
    train_set: TransitionDataset,
    val_set: TransitionDataset,
    lookahead_horizon: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_set.lookahead_horizon = lookahead_horizon
    val_set.lookahead_horizon = lookahead_horizon

    cuda = torch.cuda.is_available()
    shared = {"batch_size": batch_size, "pin_memory": cuda, "num_workers": 4 if cuda else 0}

    return (
        DataLoader(dataset=train_set, **shared, shuffle=True),
        DataLoader(dataset=val_set, **shared, shuffle=False),
    )
