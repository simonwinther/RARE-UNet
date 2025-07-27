import torch
from inference import RAREPredictor
import nibabel as nib
from utils.metrics import dice_coefficient

path = "/home/si-hj/delete_dir_for_release_github/RARE-UNet/trained_models/rare_unet/Hippocampus/2025-07-27_22-15-46_deleteme"

inference = RAREPredictor(
    model_dir_path=path,
)

image_path = '/home/si-hj/delete_dir_for_release_github/HC_PREPROC/images/hippocampus_017.nii.pt'
gt_mask_path = '/home/si-hj/delete_dir_for_release_github/HC_PREPROC/masks/hippocampus_017.nii.pt'

pred_numpy = inference.predict('/home/si-hj/delete_dir_for_release_github/HC_PREPROC/images/hippocampus_017.nii.pt')

ground_truth_tensor = torch.load(gt_mask_path, map_location='cpu').squeeze().long()
pred_tensor = torch.from_numpy(pred_numpy).long()

num_classes = 3 
dice_val = dice_coefficient(pred_tensor, ground_truth_tensor, num_classes, ignore_index=0)

print(f"Prediction (NumPy array) shape: {pred_numpy.shape}")
print(f"Prediction (Tensor) shape: {pred_tensor.shape}")
print(f"Ground Truth (Tensor) shape: {ground_truth_tensor.shape}")
print("-" * 30)
print(f"Dice Coefficient: {dice_val.item()}")
