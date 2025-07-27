import logging
import os
from typing import Optional
import numpy as np
from omegaconf import DictConfig

import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torch
import numpy as np
import torchio as tio
from pathlib import Path

logger = logging.getLogger(__name__)

class PreprocessedDataset(Dataset):
    """
    Loads .pt volumes & masks saved by preprocess_data.py.
    Accepts optional image_files/mask_files lists for compatibility
    with DataManager’s splitting logic.
    """

    def __init__(
        self,
        cfg: DictConfig,
        phase: str,
        image_files: Optional[list] = None,
        mask_files:  Optional[list] = None,
    ):
        super().__init__()
        base       = Path(cfg.dataset.base_path)
        
        images_dir = base / cfg.dataset.images_subdir
        masks_dir  = base / cfg.dataset.labels_subdir

        # If DataManager passed a subset of file names, use those;
        # otherwise glob everything under the preproc folder.
        if image_files is not None and mask_files is not None:
            self.img_files  = [images_dir / f for f in image_files]
            self.mask_files = [masks_dir  / f for f in mask_files]
        else:
            self.img_files  = sorted(images_dir.glob("*.pt"))
            self.mask_files = sorted(masks_dir.glob("*.pt"))

        assert len(self.img_files) == len(self.mask_files), \
            f"Found {len(self.img_files)} images but {len(self.mask_files)} masks"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = torch.load(self.img_files[idx])    # (C, W, H, D)
        msk = torch.load(self.mask_files[idx])   # (1, W, H, D)
        msk = msk.squeeze(0).long()              # (W, H, D)
        return img, msk
