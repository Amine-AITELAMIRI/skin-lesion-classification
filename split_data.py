"""Fixed, class-stratified validation split made only from the training folder."""

import random
from pathlib import Path

import torch
from torch.utils.data import Subset

from dataset import MelonaDataset


SPLIT_SEED = 42
VAL_FRACTION = 0.20


def set_seed(seed=SPLIT_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_indices(data_dir, seed=SPLIT_SEED, val_fraction=VAL_FRACTION):
    if Path(data_dir).name == "test":
        raise ValueError("Validation must come from the train folder")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    dataset = MelonaDataset(data_dir)
    train_indices, val_indices = [], []
    for label in range(len(dataset.classes)):
        indices = [i for i, (_, y) in enumerate(dataset.samples) if y == label]
        random.Random(seed + label).shuffle(indices)
        n_val = round(len(indices) * val_fraction)
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    train_indices.sort()
    val_indices.sort()
    assert not set(train_indices).intersection(val_indices)
    assert len(train_indices) + len(val_indices) == len(dataset)
    return train_indices, val_indices


def make_train_val(data_dir, train_transform, val_transform, seed=SPLIT_SEED):
    train_indices, val_indices = split_indices(data_dir, seed=seed)
    train_dataset = MelonaDataset(data_dir, transform=train_transform)
    val_dataset = MelonaDataset(data_dir, transform=val_transform)
    if train_dataset.samples != val_dataset.samples:
        raise RuntimeError("Training and validation sample order differs")
    return Subset(train_dataset, train_indices), Subset(val_dataset, val_indices)


def class_names(subset):
    return subset.dataset.classes
