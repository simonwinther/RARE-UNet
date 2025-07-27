#!/usr/bin/env python
"""
One-off preprocessing script for Medical Decathlon volumes.
Converts raw .nii.gz images and masks into preprocessed PyTorch tensors (.pt),
applying orientation, resampling, cropping/padding, and intensity normalization.
"""
import argparse
from pathlib import Path
import logging

import torch
import torchio as tio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_target_shape(s: str):
    """
    Convert a comma-separated string "X,Y,Z" into a tuple of ints (X, Y, Z).
    """
    return tuple(int(x) for x in s.split(","))


def main(
    raw_images: Path,
    raw_masks: Path,
    out_dir: Path,
    target_shape: tuple,
    target_spacing: tuple,
):
    """
    Preprocess all volumes in `raw_images` and corresponding masks in `raw_masks`.
    Writes out two subfolders under `out_dir`: "images/" and "masks/", each containing
    one .pt file per case, named by the case ID (filename stem).
    """
    # Create output directories
    imgs_out = out_dir / "images"
    masks_out = out_dir / "masks"
    imgs_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    # Define the heavy preprocessing pipeline
    preprocess = tio.Compose([
        tio.ToCanonical(), 
        tio.Resample(
            target_spacing,
            image_interpolation="linear",
            label_interpolation="nearest",
        ),
        tio.CropOrPad(target_shape, padding_mode="constant"),
        tio.RescaleIntensity((0, 1), percentiles=(0.5, 99.5)),
    ])

    img_paths = sorted(raw_images.glob("*.nii.gz"))
    logger.info(f"Found {len(img_paths)} volumes to preprocess...")

    for img_path in img_paths:
        case_id = img_path.stem
        mask_path = raw_masks / img_path.name

        # Load as TorchIO Subject
        subject = tio.Subject(
            image=tio.ScalarImage(str(img_path)),
            mask=tio.LabelMap(str(mask_path)),
        )

        # Apply preprocessing
        subject = preprocess(subject)

        # Extract preprocessed tensors
        img_t = subject.image.data   # shape (1, D, H, W)
        msk_t = subject.mask.data    # shape (1, D, H, W)

        # Save as .pt
        torch.save(img_t,   imgs_out / f"{case_id}.pt")
        torch.save(msk_t, masks_out / f"{case_id}.pt")

    logger.info(f"Preprocessing complete — saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess raw .nii.gz volumes to .pt tensors"
    )
    parser.add_argument(
        "--raw-images",
        type=Path,
        required=True,
        help="Folder containing raw .nii.gz images",
    )
    parser.add_argument(
        "--raw-masks",
        type=Path,
        required=True,
        help="Folder containing raw .nii.gz masks",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for preprocessed .pt files",
    )
    parser.add_argument(
        "--target-shape",
        type=str,
        required=True,
        help="Spatial shape for resampling, e.g. '256,256,128'",
    )
    parser.add_argument(
        "--target-spacing",
        type=str,
        default="1.0,1.0,1.0",
        help="Voxel spacing for resampling, e.g. '1.0,1.0,1.0'",
    )

    args = parser.parse_args()
    tgt_shape = parse_target_shape(args.target_shape)
    tgt_spacing = tuple(float(x) for x in args.target_spacing.split(","))

    main(
        args.raw_images,
        args.raw_masks,
        args.out_dir,
        tgt_shape,
        tgt_spacing,
    )
