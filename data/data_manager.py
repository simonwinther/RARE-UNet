import logging
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, random_split
import os
from data.datasets import PreprocessedDataset
from typing import Tuple, Type
import numpy as np
import random
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(
        self,
        dataset_class: Type[PreprocessedDataset],
        cfg: DictConfig,
        seed: int,
        tr_split_ratios: Tuple[float, float] = (0.80, 0.05),
    ):
        """
        Initialize the DataManager with a dataset and configuration.
        """
        self.dataset_class = dataset_class
        self.cfg = cfg
        self.train_ratio, self.val_ratio = tr_split_ratios
        self.seed = seed
        self.dataset_subset_size = cfg.get("dataset_subset_size", None)
        self.train_dataset, self.val_dataset = self._create_datasets()

    def _create_datasets(self):
        base = self.cfg.dataset.base_path

        images_path = f"{base}{self.cfg.dataset.images_subdir}"
        masks_path  = f"{base}{self.cfg.dataset.labels_subdir}"

        image_files = sorted(os.listdir(images_path))
        mask_files  = sorted(os.listdir(masks_path))
        N = len(image_files)
        assert N == len(mask_files), f"Found {N} images but {len(mask_files)} masks"

        # shuffle
        import random
        random.seed(self.seed)
        perms = list(range(N))
        random.shuffle(perms)

        # optionally take a subset for HPO
        if self.dataset_subset_size is not None:
            perms = perms[: self.dataset_subset_size]

        # build shuffled file lists
        image_files = [image_files[i] for i in perms]
        mask_files  = [mask_files[i]  for i in perms]
        M = len(image_files)

        # compute how many go to val/train
        n_val   = int(self.val_ratio   * M)
        n_train = M - n_val

        # slice
        train_imgs = image_files[:n_train]
        train_msks = mask_files[:n_train]
        val_imgs   = image_files[n_train:]
        val_msks   = mask_files[n_train:]

        # instantiate
        train_ds = self.dataset_class(
            self.cfg, phase="train", image_files=train_imgs, mask_files=train_msks
        )
        val_ds   = self.dataset_class(
            self.cfg, phase="val",   image_files=val_imgs,   mask_files=val_msks
        )
        return train_ds, val_ds

    def _create_dataloaders(self, distributed: bool = False):
        """
        Create DataLoaders. If distributed=True and torch.distributed is initialized,
        wraps the datasets in DistributedSampler so each rank sees its own shard.
        """
        if distributed and torch.distributed.is_initialized():
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            val_sampler   = DistributedSampler(self.val_dataset, shuffle=False)

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.training.batch_size,
                sampler=train_sampler,
                num_workers=self.cfg.training.num_workers,
                pin_memory=self.cfg.training.pin_memory,
                drop_last=self.cfg.architecture.drop_last,
                persistent_workers=self.cfg.training.persistent_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.training.batch_size,
                sampler=val_sampler,
                num_workers=self.cfg.training.num_workers,
                pin_memory=self.cfg.training.pin_memory,
                drop_last=False,
                persistent_workers=self.cfg.training.persistent_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.training.batch_size,
                shuffle=True,
                num_workers=self.cfg.training.num_workers,
                pin_memory=self.cfg.training.pin_memory,
                drop_last=self.cfg.architecture.drop_last,
                persistent_workers=self.cfg.training.persistent_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.training.batch_size,
                shuffle=False,
                num_workers=self.cfg.training.num_workers,
                pin_memory=self.cfg.training.pin_memory,
                drop_last=False,
                persistent_workers=self.cfg.training.persistent_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )

        return train_loader, val_loader

    def get_dataloaders(self, distributed: bool = False):
         """Return the train, val, and test DataLoaders."""
         return self._create_dataloaders(distributed)