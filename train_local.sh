#!/usr/bin/env bash
# train_local.sh


export CUDA_VISIBLE_DEVICES=3

# Default values
DATASET="Task01_BrainTumour"
ARCH="rare_unet"

# Allow user to override via arguments
if [ ! -z "$1" ]; then
  DATASET="$1"
fi

if [ ! -z "$2" ]; then
  ARCH="$2"
fi

python train.py \
  +dataset=$DATASET \
  +architecture=$ARCH \
  training.early_stopper.criterion=dice_multiscale_avg \
  gpu.mode=single \
  gpu.devices="[0]" \
  training.learning_rate=2e-3 
  # wandb.log=true
