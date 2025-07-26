#!/usr/bin/env bash
# train_ddp.sh


export CUDA_VISIBLE_DEVICES=0,1,2

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

# python -m torch.distributed.run \
#   --nproc_per_node=3 \
#   train.py \
#   +dataset=$DATASET \
#   +architecture=$ARCH \
#   training.early_stopper.criterion=dice_multiscale_avg \
#   gpu.mode=multi \
#   gpu.devices="[0,1,2]" \
#   training.learning_rate=2e-3 \
#   wandb.log=true \
#   wandb.name=resume_from_checkpoint

python -m torch.distributed.run \
  --nproc_per_node=3 \
  train.py \
  +dataset=$DATASET \
  +architecture=$ARCH \
  training.early_stopper.criterion=dice_multiscale_avg \
  gpu.mode=multi \
  gpu.devices="[0,1,2]" \
  training.learning_rate=2e-3 \
  wandb.log=true \
  wandb.name=test_resume_new_hopefully_resumed \
  +resume_checkpoint=/home/si-hj/Desktop_Simon_Cleanup/medsegnet/trained_models/rare_unet/Task01_BrainTumour/2025-07-23_00-10-52_test_resume/best_model.pth
