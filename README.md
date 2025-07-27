# RARE-UNet: Resolution-Aligned Routing Entry for Adaptive Medical Image Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![LNCS 2025](https://img.shields.io/badge/LNCS-2025-blue)](https://www.springer.com/series/558)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange)](https://pytorch.org/)

📄 This paper has been accepted by LNCS 2025 — (https://arxiv.org/abs/XXXX.XXXXX)

## Abstract

Accurate segmentation of brain MRI is critical for clinical assessment but faces challenges from variable resolution and quality across imaging protocols. We introduce RARE-UNet, a resolution-adaptive extension of the 3D UNet, designed for efficient segmentation of brain MRI with diverse voxel spacing and anisotropic resolutions. By incorporating multi-scale gateway blocks (MSBs), RARE-UNet routes inputs to encoder depths based on their resolution, preserving image fidelity and reducing computational overhead. The model employs resolution-specific segmentation heads and a scale consistency loss to align features across scales, achieving robust performance with minimal pre-processing. Evaluated on hippocampus and brain tumor segmentation tasks, RARE-UNet outperforms state-of-the-art models like nnU-Net across multiple resolutions, demonstrating superior accuracy and efficiency. This framework supports real-world clinical workflows by adapting to varying imaging conditions.

📢 Accepted at LNCS 2025

## Architecture Overview

![Architecture](figures/architecture.pdf)
*Figure 1: RARE-UNet architecture with multi-scale gateway blocks for resolution-adaptive input routing*

## Sample Results

![Sample Segmentation](figures/sample_segmentation.png)
*Figure 2: Qualitative comparison of hippocampus and brain tumor segmentation results across resolutions*

## Key Features

- **Resolution-Adaptive Processing**: Routes inputs to appropriate encoder depths based on resolution, avoiding resampling artifacts.
- **Multi-Scale Gateway Blocks**: Aligns features across scales using MSE consistency loss for robust segmentation.
- **Efficient Inference**: Activates only relevant encoder layers, reducing computational cost for low-resolution inputs.
- **Robust Performance**: Outperforms nnU-Net and standard UNets across diverse resolutions.
- **Clinical Applicability**: Handles multi-center dataset variability, supporting real-world MRI workflows.
- **Scalable Design**: Adjustable architecture depth for varying computational resources.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/simonwinther/RARE-UNet
cd RARE-UNet

# Install dependencies
pip install -e .
```

### Inference

```python
import torch
from inference import RAREPredictor
from utils.metrics import dice_coefficient

# Initialize the model
model = RAREPredictor(model_dir_path="trained_models/rare_unet/Hippocampus/2025-07-27_22-15-46")

# Run inference on a brain MRI image
pred_numpy = model.predict("data/images/hippocampus_017.nii.pt")

# Load ground truth and compute Dice coefficient
ground_truth_tensor = torch.load("data/masks/hippocampus_017.nii.pt").squeeze().long()
pred_tensor = torch.from_numpy(pred_numpy).long()
dice_val = dice_coefficient(pred_tensor, ground_truth_tensor, num_classes=3, ignore_index=0)

print(f"Dice Coefficient: {dice_val.item()}")
```

### Training

To train the model, use the provided training scripts. Example for distributed training:

```bash
bash train_ddp.sh
```

For local training:

```bash
bash train_local.sh
```

Configuration files in `config/` allow customization of architecture, dataset, and training settings.

## Project Structure

```
RARE-UNet/
├── config/                     # Configuration files
│   ├── architecture/           # Model architecture configurations
│   │   ├── rare_unet.yaml     # RARE-UNet architecture settings
│   │   └── unet.yaml          # Baseline UNet settings
│   ├── dataset/               # Dataset configurations
│   │   ├── example.yaml       # Example dataset config
│   │   └── Task01_BrainTumour.yaml  # Brain tumor dataset config
│   ├── training/              # Training configurations
│   │   ├── default.yaml       # Default training settings
│   │   ├── Task01_BrainTumour.yaml  # Brain tumor training config
│   │   └── Task04_Hippocampus.yaml  # Hippocampus training config
│   └── base.yaml              # Base configuration
├── data/                      # Data handling and preprocessing
│   ├── data_manager.py       # Dataset management utilities
│   ├── datasets.py           # Dataset loading and processing
│   └── preprocess_data.py    # Preprocessing utilities to convert to .nii.pt
├── models/                    # Model implementations
│   ├── rare_unet.py          # RARE-UNet model with multi-scale blocks
│   └── unet.py               # Baseline 3D UNet model
├── trainers/                  # Training utilities
│   ├── early_stopping.py     # Early stopping implementation
│   ├── rare_trainer.py       # RARE-UNet training logic
│   └── trainer.py            # General training utilities
├── utils/                     # Utility functions
│   ├── checkpoint_handler.py # Model checkpoint management
│   ├── logging.py            # Logging utilities
│   ├── losses.py             # Custom loss functions (MSE consistency + Dice)
│   ├── metric_collecter.py   # Metric collection utilities
│   ├── metrics.py            # Evaluation metrics
│   ├── table.py              # Result table generation
│   ├── utils.py              # General utilities
│   ├── wandb_logger.py       # Weights & Biases logging
│   └── weight_strategies.py  # Weight initialization strategies
├── example.py                # Example usage script
├── inference.py              # Inference pipeline
├── train.py                  # Main training script
├── train_ddp.sh              # Distributed training script
├── train_local.sh            # Local training script
├── README.md                 # Project documentation
└── setup.py                  # Package installation
```

## Data Structure

The project expects data in the following format, with images and masks preprocessed into `.nii.pt` format using `data/preprocess_data.py`:

```
your_dataset/
├── images/
│   ├── hippocampus_001.nii.pt
│   ├── ...
│   └── hippocampus_394.nii.pt
├── masks/
│   ├── hippocampus_001.nii.pt
│   ├── ...
│   └── hippocampus_394.nii.pt
```

Configure datasets using YAML files in `config/dataset/`, such as `Task01_BrainTumour.yaml` or `Task04_Hippocampus.yaml`.

## Models

### RARE-UNet Model (`models/rare_unet.py`)
- 3D UNet backbone with multi-scale gateway blocks.
- Resolution-adaptive routing for variable-resolution inputs.
- Shared bottleneck and resolution-specific segmentation heads.

### Baseline UNet Model (`models/unet.py`)
- Standard 3D UNet implementation for comparison.
- Used as a baseline in experiments.

## Acknowledgments

We thank the following projects and teams for their foundational work:

- **nnU-Net Team** for the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework
- **PyTorch Team** for the [PyTorch](https://pytorch.org/) deep learning library
- **Medical Imaging Community** for providing benchmark datasets for hippocampus and brain tumor segmentation

This work builds upon these foundations to advance resolution-adaptive segmentation in brain MRI.

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@article{simon2025rare,
  title={RARE-UNet: Resolution-Adaptive UNet for Brain MRI Segmentation},
  author={Simon, Hjalte and others},
  journal={Lecture Notes in Computer Science (LNCS)},
  year={2025}
}
```

## Contact

For questions and collaborations, please contact: [{zlp616, fhz806}@alumni.ku.dk]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
