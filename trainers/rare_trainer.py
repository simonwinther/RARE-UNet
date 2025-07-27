from typing import Any, Dict, List
from numpy import resize
from sympy import N
import torch
import torch.nn.functional as F

from utils.metric_collecter import Agg
from trainers.trainer import BaseTrainer

from utils.utils import resize_masks_to
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DistributedDataParallel

class RARETrainer(BaseTrainer):
    def __init__(self, cfg, model, *args, **kwargs):
        if isinstance(model, DistributedDataParallel):
            net = model.module
        else:
            net = model
        self.cons_loss_start_epoch = cfg.architecture.get("cons_loss_start_epoch", 0)
        n_levels = net.n_ms_levels
        assert isinstance(n_levels, int), "n_ms_levels must be int"
        assert n_levels > 0, "n_ms_levels must be greater than 0"
        assert n_levels < cfg.architecture.depth, "n_ms_levels must be less than or equal to depth"
        self.n_ms_levels = n_levels
        self.weights = {
            'segmentation': [1 / cfg.architecture.depth] * cfg.architecture.depth,
            'consistency': [1 / (n_levels)] * n_levels
        }
        super().__init__(cfg, model, *args, **kwargs)
        
    def _compute_loss(self, outputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        segs, cons_pairs = outputs
        return self._compute_losses(segs, cons_pairs, masks)

    def _compute_losses(self, segs, cons_pairs, targets):
        seg_loss = self._compute_segmentation_loss(segs, targets)
        cons_loss = self._compute_consistency_loss(cons_pairs)
        λ_cons = min(1.0, (self.current_epoch + 1e-6) / (self.cons_loss_start_epoch + 1e-6))
        return seg_loss + λ_cons * cons_loss

    def _compute_segmentation_loss(self, segs, targets):
        seg_loss = torch.tensor(0.0, device=self.device)
        for w, pred in zip(self.weights['segmentation'], segs):
            gt = resize_masks_to(pred, targets)
            seg_loss += w * self.criterion(pred, gt)
        return seg_loss

    def _compute_consistency_loss(self, cons_pairs):
        losses = torch.stack([
            w * F.mse_loss(ms, enc.detach())
            for w, (ms, enc) in zip(self.weights['consistency'], cons_pairs)
        ], dim=0) 
        return losses.mean()

    def _compute_per_scale_consistency_losses(self, cons_pairs) -> Dict[str, float]:
        """
        Returns a dict of per-scale consistency losses using the same naming pattern.
        """
        return {
            f"consistency_loss_scale{i}": F.mse_loss(
                ms_feats, enc_feats.detach()
            ).item()
            for i, (ms_feats, enc_feats) in enumerate(cons_pairs, start=1)
        }

    def _setup_metric_collector_rules(self):
        # Full res (scale0)
        super()._setup_metric_collector_rules()
        for metric in self.metrics:
            # Per scale
            for i in range(1, self.n_ms_levels + 1):
                self.mc.set_rule(f"{metric}_scale{i}", Agg.MEAN)
                self.mc.set_rule(f"{metric}_per_class_scale{i}", Agg.LIST_MEAN)

            # Multiscale combined (excludes fullres)
            self.mc.set_rule(f"{metric}_multiscale_avg", Agg.MEAN)
            self.mc.set_rule(f"{metric}_per_class_multiscale_avg", Agg.LIST_MEAN)

            # combined across all scales (includes fullres)
            self.mc.set_rule(f"{metric}_all_scales_avg", Agg.MEAN)
            self.mc.set_rule(f"{metric}_per_class_all_scales_avg", Agg.LIST_MEAN)

    def _convert_fullres_key_to_scale_variant(self, key: str, scale_idx: int) -> str:
        if scale_idx != 0:
            if key.endswith("_per_class_fullres"):
                return key.replace("_per_class_fullres", f"_per_class_scale{scale_idx}")
            elif key.endswith("_fullres"):
                return key.replace("_fullres", f"_scale{scale_idx}")
        return key

    def _compute_multiscale_averages(self, combined: Dict[str, Any]):
        """
        Compute average across all scales excluding fullres, hence the name multiscale average.
        """
        for metric in self.metrics:
            keys = []
            cls_keys = []
            for i in range(1, self.n_ms_levels + 1):
                keys.append(f"{metric}_scale{i}")
                cls_keys.append(f"{metric}_per_class_scale{i}")

            vals = [combined[k] for k in keys]
            combined[f"{metric}_multiscale_avg"] = sum(vals) / len(vals)

            cls_lists = [combined[k] for k in cls_keys]
            num_cls = len(cls_lists[0])
            combined[f"{metric}_per_class_multiscale_avg"] = [
                sum(cls_list[c] for cls_list in cls_lists) / len(cls_lists)
                for c in range(num_cls)
            ]

    def _compute_all_scales_avg(self, combined: Dict[str, Any]):
        """Compute averages across all scales (multiscales + fullres)."""
        for metric in self.metrics:
            all_scales_avg = combined.get(f"{metric}_multiscale_avg", 0.0)
            fullres = combined.get(f"{metric}_fullres", 0.0)
            all_scales_avg *= self.n_ms_levels
            all_scales_avg += fullres
            all_scales_avg /= self.n_ms_levels + 1

            per_class_multiscale_avg = combined.get(
                f"{metric}_per_class_multiscale_avg", [0.0] * self.num_classes
            )
            per_class_fullres = combined.get(
                f"{metric}_per_class_fullres", [0.0] * self.num_classes
            )
            all_scales_avg_per_class = [
                (ms * self.n_ms_levels + f) / (self.n_ms_levels + 1)
                for ms, f in zip(per_class_multiscale_avg, per_class_fullres)
            ]
            combined[f"{metric}_all_scales_avg"] = all_scales_avg
            combined[f"{metric}_per_class_all_scales_avg"] = all_scales_avg_per_class

    def _compute_metrics(self, outputs, masks) -> Dict[str, Any]:
        outputs, cons = outputs
        per_scale_metrics: List[Dict[str, Any]] = []

        for i, seg in enumerate(outputs):
            gt = resize_masks_to(seg, masks)
            raw_metrics = super()._compute_metrics(seg, gt)

            renamed = {}
            for k, v in raw_metrics.items():
                renamed[self._convert_fullres_key_to_scale_variant(k, i)] = v
            per_scale_metrics.append(renamed)

        combined = {k: v for d in per_scale_metrics for k, v in d.items()}
        self._compute_multiscale_averages(combined)
        self._compute_all_scales_avg(combined)
        combined.update(self._compute_per_scale_consistency_losses(cons))
        return combined
