from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Optional, Sequence
from .metrics import zero_convention

# LOSS_FUNCTIONS = {}  # TODO moe into a dedicated module :)

# @deprecated("No longer used -- we use hydra.utils.instantiate instead")
# def register_loss_function(cls):
#     """Decorator to register loss functions by class name (lowercased)."""
#     LOSS_FUNCTIONS[cls.__name__.lower()] = cls
#     return cls


# @deprecated("No longer used -- we use hydra.utils.instantiate instead")
# def get_loss_fn(arch_cfg):
#     model_cfg = arch_cfg.model

#     assert "loss" in model_cfg, f"Loss not defined in the config."
#     assert "name" in model_cfg.loss, f"Loss name not defined in the config."
#     assert "params" in model_cfg.loss, f"Loss params not defined in the config."

#     loss_name = model_cfg.loss.name.lower()
#     loss_params = OmegaConf.to_container(model_cfg.loss.params, resolve=True)
#     assert isinstance(
#         loss_params, dict
#     ), f"Expected a dict for loss params, got {type(loss_params)}"

#     loss_params = {str(k): v for k, v in loss_params.items()}

#     assert (
#         loss_name in LOSS_FUNCTIONS
#     ), f"Loss function '{loss_name}' not found in registered loss functions."
#     loss_class = LOSS_FUNCTIONS[loss_name]

#     return loss_class(**loss_params)


def compute_dice(prob, target, smooth=0.0, dims=(0, 1, 2, 3)): # 1e-6
    """
    Compute the Dice score for a single class.
    """
    intersection = torch.sum(prob * target, dim=dims)
    union = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)
    if union == 0:
        return zero_convention(intersection)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice


def soft_dice_loss(output, target, num_classes, smooth=0.0, ignore_index=None): # 1e-6
    """
    Compute the soft Dice loss in a differentiable way using softmax probabilities.
    """
    # Convert target to one-hot encoding with shape (B, C, D, H, W)
    target_one_hot = (
        F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    )

    probs = torch.softmax(output, dim=1)  # (B, C, D, H, W)

    dice_scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        # Compute dice score for class c using the helper function
        dice_score = compute_dice(
            probs[:, c, ...], target_one_hot[:, c, ...], smooth=smooth
        )
        dice_scores.append(dice_score)

    # Average dice score across classes and convert it to a loss
    mean_dice = torch.stack(dice_scores).mean()
    return 1 - mean_dice


class CombinedLoss(nn.Module):
    """
    Combined loss function for multi-class segmentation.
    The loss is a linear combination of the cross-entropy loss and the Dice loss.
    """

    def __init__(self, alpha: float, ignore_index: Optional[int] = None):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.ignore_index = ignore_index

    def soft_dice_loss(self, output, target):
        num_classes = output.shape[1]
        return soft_dice_loss(
            output, target, num_classes, ignore_index=self.ignore_index
        )

    def forward(self, pred, target):
        return self.alpha * self.ce(pred, target) + (
            1 - self.alpha
        ) * self.soft_dice_loss(pred, target)


class FocalDiceLoss(nn.Module):
    """
    Credits: https://github.com/usagisukisuki/Adaptive_t-vMF_Dice_loss/blob/main/SegLoss/focal_diceloss.py
    """

    def __init__(self, n_classes, beta=2.0):  # TODO change to cfg.num_classes
        super(FocalDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.beta = 1.0 / beta

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _fdice_loss(self, score, target):
        target = target.float()
        smooth = 0.0 # 1.0
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        yz_sum = y_sum + z_sum
        if yz_sum == 0:
            return 1 - zero_convention(intersect)
        loss = (2 * intersect + smooth) / (yz_sum + smooth)

        loss = 1 - loss**self.beta

        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )

        loss = 0.0
        for i in range(self.n_classes):
            dice = self._fdice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]

        return loss / self.n_classes
