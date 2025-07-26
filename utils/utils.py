import string
from networkx import omega
from numpy import divide
from typing import Dict, Optional, Tuple
from datetime import datetime
import torch
import yaml
import json
from omegaconf import OmegaConf
import re
import numpy as np, random
import torch.nn.functional as F
import warnings
import functools
from omegaconf import DictConfig

def setup_environment(cfg):
    """
    Seeds RNGs, then applies either cudnn.benchmark or full determinism
    based on cfg.training.optimization.mode.
    """
    # Seed
    seed = cfg.seed
    setup_seed(seed) 

    # Decide on benchmark vs deterministic
    mode = cfg.optimization.mode
    if mode == "deterministic":
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    elif mode == "benchmark":
        torch.backends.cudnn.benchmark = True


def setup_seed(seed: int):
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


def resize_masks_to(maps: torch.Tensor, gt_maps: torch.Tensor) -> torch.Tensor:
    """
    Resizes ground truth masks to match the spatial dimensions of the given feature maps.

    Args:
        maps (torch.Tensor): Feature maps with shape `(B, C, D, H, W)`.
        gt_maps (torch.Tensor): Ground truth masks with shape `(B, D, H, W)` or `(B, 1, D, H, W)`.

    Returns:
        torch.Tensor: Resized masks with shape `(B, D_new, H_new, W_new)`.
    """
    if maps.ndim != 5:
        raise ValueError(f"Expected maps to have 5 dimensions, got {maps.ndim}.")
    
    if gt_maps.ndim not in (4, 5):
        raise ValueError(f"Expected gt_maps to have 4 or 5 dimensions, got {gt_maps.ndim}.")
    
    if gt_maps.ndim == 4:
        gt_maps = gt_maps.unsqueeze(1)  
        
    resized = F.interpolate(gt_maps.float(), size=maps.shape[2:], mode="nearest")
    return resized.squeeze(1).long()

