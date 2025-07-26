from typing import Optional
from networkx import union
import numpy as np
from sympy import N
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from typing import Optional



# Zero convention for metric tensors - avoiding division by zero, without smoothing
def zero_convention(metric_tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(1.0, device=metric_tensor.device) if metric_tensor == 0 else torch.tensor(0.0, device=metric_tensor.device)

# metrics.py:
# Accuracy

# ----- Accuracy -----
def accuracy_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute pixel‐wise accuracy for segmentation:
    (number_of_correct_pixels) / (number_of_valid_pixels).
    
    preds: (B, W, H, D) or any shape of integer‐encoded class predictions.
    masks: same shape, containing integer labels (0..C-1), or `ignore_index`.
    ignore_index: int label in `masks` to skip (e.g. 255). If None, count all pixels.
    """
    if preds.shape != masks.shape:
        raise ValueError("Predictions and masks must have the same shape.")

    # Flatten everything to 1D:
    preds_flat = preds.reshape(-1)
    masks_flat = masks.reshape(-1)

    # If ignore_index is specified, keep only the “valid” pixels
    if ignore_index is not None:
        valid_mask = (masks_flat != ignore_index)
        preds_flat = preds_flat[valid_mask]
        masks_flat = masks_flat[valid_mask]

    # Now compute correct / total_valid
    total_valid = masks_flat.numel()
    if total_valid == 0:
        # no valid pixels ⇒ return 0.0 (or you could choose 1.0, depending on convention)
        return torch.tensor(0.0, device=preds.device)

    correct = (preds_flat == masks_flat).sum().float()
    return correct / total_valid

# ----- End Accuracy -----

# ----- IoU (Intersection over Union) -----
def compute_iou_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 0.0, # 1e-6,
) -> torch.Tensor:
    """Compute IoU score for a single class."""
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()
    intersection = (pred_c * mask_c).sum()
    union = pred_c.sum() + mask_c.sum() - intersection
    if union == 0:
        return zero_convention(intersection)
    return (intersection + smooth) / (union + smooth)


def iou_score_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 0.0, # 1e-6,
    ignore_index: Optional[int] = None,
) -> list[torch.Tensor]:
    """Return list of IoU scores for each non-ignored class."""
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        score = compute_iou_score(preds, masks, c, smooth)
        scores.append(score)
    return scores


def iou_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 0.0, # 1e-6,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Return average IoU score across non-ignored classes."""
    iou_scores = iou_score_classes(preds, masks, num_classes, smooth, ignore_index)
    if not iou_scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(iou_scores).mean()


# ----- End IoU (Intersection over Union) -----


# ----- Dice coefficient -----
def compute_dice_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 0.0, # 1e-6,  # Remove unused `ignore_index`
) -> torch.Tensor:
    """Compute Dice score for a single class."""
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()
    intersection = (pred_c * mask_c).sum()
    sum_pred = pred_c.sum()
    sum_mask = mask_c.sum()
    union = sum_pred + sum_mask
    if union == 0:
        return zero_convention(intersection)
    return (2 * intersection + smooth) / (sum_pred + sum_mask + smooth)


def _compute_dice_scores(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float,
    ignore_index: Optional[int],
) -> list[torch.Tensor]:
    """Helper to compute Dice scores for all classes (excluding `ignore_index`)."""
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue  # Skip ignored class
        score = compute_dice_score(preds, masks, c, smooth)
        scores.append(score)
    return scores


def dice_coefficient(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 0.0, # 1e-6,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Return average Dice coefficient across non-ignored classes."""
    dice_scores = _compute_dice_scores(preds, masks, num_classes, smooth, ignore_index)
    # ASSERT shapes equal
    assert preds.shape == masks.shape, "Shapes are not equal you fucked up..."

    if not dice_scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(dice_scores).mean()


#TODO: This function is currently duplicated from _compute_dice_scores.
# Consider refactoring to avoid redundancy and ensure consistency.
def dice_coefficient_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 0.0, # 1e-6,
    ignore_index: Optional[int] = None,
) -> list[torch.Tensor]:
    """Return list of Dice scores for each non-ignored class."""
    return _compute_dice_scores(preds, masks, num_classes, smooth, ignore_index)


# ----- End Dice coefficient -----


# ----- Precision -----
def precision_score_class(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 0.0, # 1e-6,
) -> torch.Tensor:
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()
    tp = (pred_c * mask_c).sum()
    fp = (pred_c * (1 - mask_c)).sum()
    if tp + fp == 0:
        return zero_convention(tp)
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision


def precision_score_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 0.0, # 1e-6,
    ignore_index: Optional[int] = None,
) -> list[torch.Tensor]:
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        score = precision_score_class(preds, masks, c, smooth)
        scores.append(score)
    return scores


def precision_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 0.0, # 1e-6,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    scores = precision_score_classes(preds, masks, num_classes, smooth, ignore_index)
    if not scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(scores).mean()


# ----- End Precision -----


# ----- Recall -----
def recall_score_class(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 0.0, # 1e-6,
) -> torch.Tensor:
    pred_c = (preds == c).float()
    mask_c = (masks == c).float()
    tp = (pred_c * mask_c).sum()
    fn = ((1 - pred_c) * mask_c).sum()
    if tp + fn == 0:
        return zero_convention(tp)
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall


def recall_score_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 0.0, # 1e-6,
    ignore_index: Optional[int] = None,
) -> list[torch.Tensor]:
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        score = recall_score_class(preds, masks, c, smooth)
        scores.append(score)
    return scores


def recall_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 0.0, # 1e-6,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    scores = recall_score_classes(preds, masks, num_classes, smooth, ignore_index)
    if not scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(scores).mean()


# ----- End Recall -----


# ----- F1 -----
def f1_score_class(
    preds: torch.Tensor,
    masks: torch.Tensor,
    c: int,
    smooth: float = 0.0, # 1e-6,
) -> torch.Tensor:
    precision = precision_score_class(preds, masks, c, smooth)
    recall = recall_score_class(preds, masks, c, smooth)
    if precision + recall == 0:
        return zero_convention(precision + recall)
    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)
    return f1


def f1_score_classes(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 0.0, # 1e-6,
    ignore_index: Optional[int] = None,
) -> list[torch.Tensor]:
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        score = f1_score_class(preds, masks, c, smooth)
        scores.append(score)
    return scores


def f1_score(
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    smooth: float = 0.0, # 1e-6,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    scores = f1_score_classes(preds, masks, num_classes, smooth, ignore_index)
    if not scores:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(scores).mean()


# ----- End F1 -----


def safe_mean(tensor_list):
    return torch.stack(tensor_list).mean().item() if tensor_list else 0.0
