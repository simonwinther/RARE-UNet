import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from skimage.transform import resize
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from models.rare_unet import RAREUNet

class RAREPredictor:
    """
    A predictor for RARE-UNet that uses the Hydra configuration from training
    and pathlib.Path for robust file system interactions.
    """
    def __init__(self, model_dir_path: str, device: str = 'cpu'):
        """
        Initializes the RAREPredictor.

        Args:
            model_dir_path (str): Path to the directory containing model files
                                  (e.g., 'trained_models/2025-07-23_...').
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        
        model_dir = Path(model_dir_path)
        config_path = model_dir / "config.yaml"
        model_path = model_dir / "best_model.pth"

        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
        
        if not config_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Model directory '{model_dir}' must contain 'config.yaml' and 'best_model.pth'."
            )

        self.cfg: DictConfig = OmegaConf.load(config_path)
        self.model = self._load_model(model_path)
        self.last_prediction = None

    def _load_model(self, model_path: Path) -> torch.nn.Module:
        """
        Instantiates the model using the Hydra config and loads the saved weights.
        """
        try:
            print("Instantiating model using configuration...")
            model = instantiate(self.cfg.architecture, cfg=self.cfg, mode="inference")

            print(f"Loading weights from {model_path}...")
            # torch.load works directly with Path objects
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            model.to(self.device)
            model.eval()
            
            print("Model loaded successfully.")
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading the model: {e}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses the image using settings from the config."""
        target_shape = self.cfg.dataset.target_shape
        resized_image = resize(image, tuple(target_shape), anti_aliasing=True, preserve_range=True)
        
        mean, std = np.mean(resized_image), np.std(resized_image)
        normalized_image = (resized_image - mean) / std if std > 0 else resized_image
        return normalized_image

    def _postprocess(self, model_output: torch.Tensor, original_shape: tuple) -> np.ndarray:
        """Post-processes model output to a segmentation mask."""
        if self.cfg.dataset.num_classes > 1:
            mask = torch.argmax(model_output, dim=1).squeeze().cpu().numpy()
        else:
            mask = (torch.sigmoid(model_output) > 0.5).squeeze().cpu().numpy()
        
        resized_mask = resize(mask, original_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        return resized_mask

    def predict(self, image_path: str) -> np.ndarray:
        """
        Runs inference on a 3D medical image.
        """
        img_file = Path(image_path)
        print(f"Loading image: {img_file}")
        # nibabel works directly with Path objects
        img_nifti = nib.load(img_file)
        img_data = img_nifti.get_fdata()
        original_shape = img_data.shape

        print("Preprocessing image...")
        preprocessed_img = self._preprocess(img_data)

        print("Running model inference...")
        with torch.no_grad():
            input_tensor = torch.from_numpy(preprocessed_img).unsqueeze(0).unsqueeze(0).float().to(self.device)
            output = self.model(input_tensor)

        print("Postprocessing output...")
        prediction_mask = self._postprocess(output, original_shape)
        
        self.last_prediction = (img_data, prediction_mask)
        print("Prediction complete.")
        return prediction_mask

    def visualize(self, save_path: str = "visualization.png"):
        """Visualizes the last prediction result."""
        if not self.last_prediction:
            print("No prediction available. Run `predict()` first.")
            return

        image, mask = self.last_prediction
        slice_idx = np.argmax(np.sum(mask, axis=(0, 1)))
        img_slice = np.rot90(image[:, :, slice_idx])
        mask_slice = np.rot90(mask[:, :, slice_idx])
        
        colors = self.cfg.dataset.get("colors", ['#FF0000', '#00FF00', '#0000FF'])
        labels = self.cfg.dataset.get("label_names", [f"Class {i+1}" for i in range(len(colors))])

        cmap_colors = ['#00000000'] + colors
        class_cmap = ListedColormap(cmap_colors)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img_slice, cmap="gray")
        masked_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
        ax.imshow(masked_overlay, cmap=class_cmap, alpha=0.5, vmin=0, vmax=len(cmap_colors) - 1)

        legend_handles = [mpatches.Patch(color=c, label=n, alpha=0.5) for n, c in zip(labels, colors)]
        ax.legend(handles=legend_handles, loc='upper right')
        ax.set_title(f"RARE-UNet Segmentation (Slice {slice_idx})")
        ax.axis('off')

        output_file = Path(save_path)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
        plt.show()