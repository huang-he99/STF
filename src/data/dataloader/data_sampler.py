from typing import Iterator, Optional, Sized
from torch.utils.data import Sampler
import numpy as np
import math
import torch


class EpochBasedSampler(Sampler):
    def __init__(self, dataset, is_shuffle, seed):
        self.dataset = dataset
        self.is_shuffle = is_shuffle
        self.seed = seed
        self.epoch = 0
        self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        if self.is_shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g).tolist()
        else:
            indices = torch.arange(self.total_size).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
