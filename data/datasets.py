import logging
import os
from typing import Optional
import numpy as np
from omegaconf import DictConfig

from .data_augmentation import AugmentationUtils
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
        #HARDCODED: 
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
        # print(f"Loaded {self.img_files[idx].name} and {self.mask_files[idx].name}")
        # print(f"Image shape: {img.shape}, Mask shape: {msk.shape}")
        msk = msk.squeeze(0).long()              # (W, H, D)
        return img, msk


class NotPreprocessedDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        phase: str,
        image_files: Optional[list] = None,
        mask_files: Optional[list] = None,
        images_path: Optional[str] = None,
        masks_path: Optional[str] = None,
    ):
        """
        Initialize the Medical Decathlon dataset.

        Sets image and mask paths, loads file names, and selects
        data augmentation transforms based on the phase (train/val/test).
        """
        self.cfg = cfg

        self.phase = phase
        base = cfg.dataset.base_path
        self.images_path = (
            images_path
            if images_path is not None
            else os.path.join(base, cfg.dataset.images_subdir)
        )
        self.masks_path = (
            masks_path
            if masks_path is not None
            else os.path.join(base, cfg.dataset.labels_subdir)
        )
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(f"Images path does not exist: {self.images_path}")
        
        if not os.path.exists(self.masks_path):
            raise FileNotFoundError(f"Labels path does not exist: {self.masks_path}")

        if image_files is None or mask_files is None:
            self.image_files = sorted(os.listdir(self.images_path))
            self.mask_files = sorted(os.listdir(self.masks_path))
        else:
            self.image_files = image_files
            self.mask_files = mask_files

        self.target_shape = self.cfg.dataset.target_shape
        self.num_classes = self.cfg.dataset.num_classes

        if self.phase == "train":
            self.transform = AugmentationUtils.get_train_transforms(
                cfg, self.target_shape
            )
        elif self.phase == "val":
            self.transform = AugmentationUtils.get_validation_transforms(
                self.target_shape
            )
        elif self.phase == "test":
            self.transform = AugmentationUtils.get_test_transforms()

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.image_files)

    def load_img_and_gts(self, idx):
        """
        Load the image and corresponding ground truth mask at the given index.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Image and mask arrays.
        """
        image_path = os.path.join(self.images_path, self.image_files[idx])

        image = nib.as_closest_canonical(nib.load(image_path)).get_fdata()  # [W, H, D]

        mask_path = os.path.join(self.masks_path, self.mask_files[idx])
        mask = nib.as_closest_canonical(nib.load(mask_path)).get_fdata()  # [W, H, D]
        return image, mask

    def __matmul__(self, idx):
        """
        Allows syntax like: dataset @ idx
        Returns the image filename at the given index.
        """
        return self.__get_image_name(idx)

    def __get_image_name(self, idx):
        """
        Returns the name of the image file at the given index.
        """
        return self.image_files[idx] if idx < len(self.image_files) else None

    def __getitem__(self, idx):
        """
        Load and preprocess the image and mask at the given index.

        Applies necessary shape adjustments and TorchIO transformations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and mask.
        """
        image_np, mask_np = self.load_img_and_gts(idx)

        image = torch.from_numpy(image_np.copy()).float()
        mask = torch.from_numpy(mask_np.copy()).long()

        if image.ndim == 3:  # Case: (W, H, D) -> (1, W, H, D)
            image = image.unsqueeze(0)
        elif image.ndim == 4:  # Case: (W, H, D, M) -> (M, W, H, D)
            image = image.permute(3, 0, 1, 2)
        else:
            raise ValueError(f"Image loaded with unsupported dimensions: {image.shape}")

        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        else:
            raise ValueError(f"Invalid mask shape: {mask.shape}")

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image), mask=tio.LabelMap(tensor=mask)
        )

        subject = self.transform(subject)
        image_out = subject.image.data
        mask_out = subject.mask.data.squeeze(0).long()
        return image_out, mask_out


class ModalitiesDataset(NotPreprocessedDataset):
    def __getitem__(self, idx):
        """
        Load and preprocess a multi-modal image and its mask.

        Assumes input image has shape (W, H, D, M) and mask is (W, H, D).
        Applies modality-aware preprocessing and TorchIO transforms.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed multi-modal image and mask.
        """
        image_np, mask_np = super().load_img_and_gts(idx)

        image = torch.from_numpy(image_np.copy()).float()  # Shape (W, H, D, M)
        mask = torch.from_numpy(mask_np.copy()).long()  # Shape (W, H, D)

        # Image: (W, H, D, M) -> (M, W, H, D)
        if image.ndim == 4:
            # image = image.permute(3, 2, 1, 0)
            image = image.permute(3, 0, 1, 2)
        else:
            # This dataset is specifically for multi-modality images expected to be 4D
            raise ValueError(f"ModalitiesDataset expected 4D image, got {image.shape}")

        # Mask: (W, H, D) -> (1, W, H, D)
        if mask.ndim == 3:
            # mask = mask.permute(2, 1, 0)
            mask = mask.unsqueeze(0)
        else:
            raise ValueError(f"ModalitiesDataset expected 3D mask, got {mask.shape}")

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image),
            mask=tio.LabelMap(tensor=mask),
        )
        subject = self.transform(subject)
        image_out = subject.image.data
        mask_out = subject.mask.data.squeeze(0).long()

        return image_out, mask_out

