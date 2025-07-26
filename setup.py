from setuptools import setup, find_packages

setup(
    name="rare-unet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "tabulate==0.9.0",
        "numpy==1.26.4",
        "wandb==0.19.9",
        "torchio==0.20.5",
        "torchinfo==1.8.0",
        "tqdm==4.67.1",
        "matplotlib==3.10.0",
        "nibabel==5.3.2",
        "omegaconf==2.3.0",
        "hydra-core==1.3.2",
    ],
    extras_require={
        "gpu": ["pytorch-cuda"]
    },
    python_requires=">=3.10", # Updated to your Python 3.10.9
)