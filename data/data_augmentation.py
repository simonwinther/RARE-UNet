import logging
import torch
import torchio as tio
import numpy as np
from hydra.utils import instantiate

logger = logging.getLogger(__name__)


class AugmentationUtils:

    @staticmethod
    def _get_preprocessing_list(
        target_shape, target_spacing=(1.0, 1.0, 1.0), rescale_percentiles=(0.5, 99.5)
    ):
        return [
            tio.ToCanonical(),
            tio.Resample(target_spacing),
            tio.CropOrPad(target_shape, padding_mode="constant"),
            tio.RescaleIntensity((0, 1), percentiles=rescale_percentiles),
        ]

    @staticmethod
    def get_train_transforms(
        cfg,
        target_shape,
        target_spacing=(1.0, 1.0, 1.0),
        rescale_percentiles=(0.5, 99.5),
    ):
        preprocessing_list = AugmentationUtils._get_preprocessing_list(
            target_shape=target_shape,
            target_spacing=target_spacing,
            rescale_percentiles=rescale_percentiles,
        )

        if cfg.training.data_augmentation.enabled:
            augmentations = instantiate(cfg.training.data_augmentation.transforms)
            logger.info(
                f"Using data augmentation: {cfg.training.data_augmentation.transforms}"
            )
        else:
            logger.info("Data augmentation is disabled.")
            augmentations = []

        all_transforms_list = preprocessing_list + augmentations

        return tio.Compose(all_transforms_list)

    @staticmethod
    def get_validation_transforms(target_shape, rescale_percentiles=(0.5, 99.5)):
        transforms = [
            tio.RescaleIntensity((0, 1), percentiles=rescale_percentiles),
            tio.CropOrPad(target_shape, padding_mode="constant"),
        ]
        return tio.Compose(transforms)

    @staticmethod
    def get_test_transforms(rescale_percentiles=(0.5, 99.5)):
        transforms = [
            tio.RescaleIntensity((0, 1), percentiles=rescale_percentiles),
        ]
        return tio.Compose(transforms)