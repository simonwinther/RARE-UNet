# Standard Library
# For core Python functionalities, like interacting with the operating system,
# file paths, and system-level operations.
import os
import traceback
import logging
from pathlib import Path
from sys import stderr


# External Libraries
# For all third-party packages installed from external sources (e.g., via pip).
import torch
import hydra
import numpy as np
import torch.distributed as dist
from omegaconf import DictConfig
from torchinfo import summary
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.utils import instantiate as inst
from trainers.rare_trainer import RARETrainer

# Local Application Imports
# For all internal modules and packages specific to this project.
from data.data_manager import DataManager
from data.datasets import PreprocessedDataset
from utils.checkpoint_handler import CheckpointHandler
from utils.utils import setup_environment
from utils.logging import setup_logging
from utils.wandb_logger import get_wandb_logger

@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # ─── parse torchrun’s ───────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) 

    # ─── basic setup ─────────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    setup_environment(cfg)
    ckpt_handler = CheckpointHandler(cfg)
    setup_logging(cfg.get("logging", {}), ckpt_handler.exp_dir)
    logger = logging.getLogger(__name__)  

    if torch.cuda.device_count() == 0:
        logger.error("No CUDA devices found. Please check your GPU setup.")
        exit(1)

    # ─── choose single‐ vs multi‐GPU based on our config ────────────────────
    if cfg.gpu.mode == "multi":
        is_distributed = True
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=cfg.gpu.backend,
            init_method=cfg.gpu.env,
            world_size=len(cfg.gpu.devices),
            rank=local_rank,
        )
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
    else:
        is_distributed = False
        rank = 0
        first_gpu = cfg.gpu.devices[0]
        torch.cuda.set_device(first_gpu)
        device = torch.device(f"cuda:{first_gpu}")

    is_main = (rank == 0)
    try:
        model = inst(cfg.architecture.path, cfg)        
        optimizer = inst(cfg.training.optimizer.params, params=model.parameters())
        criterion = inst(cfg.training.loss.params)
        lr_scheduler = inst(cfg.training.scheduler.params, optimizer=optimizer)
    except Exception:
        logger.error("Failed to instantiate network. See full traceback below.")
        logger.error(traceback.format_exc()) 
    
        if is_distributed:
            dist.destroy_process_group()
        exit(1)        
    
    wandb_resume_id = None
    resume_path = cfg.get("resume_checkpoint", None)
    if resume_path and is_main:
        logger.info(f"Resume requested. Checking for wandb ID in: {resume_path}")
        wandb_resume_id = ckpt_handler.get_resume_id_from_checkpoint(resume_path)

    model = model.to(device)
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    
    wandb_logger = None
    if is_main:
        model_summary = summary(
            model,
            input_size=(
                cfg.training.batch_size,
                cfg.dataset.in_channels,
                *cfg.dataset.target_shape
            ),
            col_names=tuple(cfg.architecture.summary.col_names),
            verbose=cfg.architecture.summary.verbose,
        )
        wandb_logger = get_wandb_logger(
            cfg=cfg, 
            model=model, 
            model_summary=model_summary,
            resume_id=wandb_resume_id
        )

    # ─── create your DataLoaders------------------------------------───────
    data_manager = DataManager(PreprocessedDataset, cfg, seed, tr_split_ratios=(0.80, 0.05))
    train_dataloader, val_dataloader = data_manager.get_dataloaders(distributed=is_distributed)
    
    trainer = RARETrainer(
        cfg,
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        ckpt_handler,
        wandb_logger,
        is_distributed, 
        local_rank, 
        is_main,
        resume_path,
    )

    final_status_code = 0
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        final_status_code = 1
    finally:
        if is_main and wandb_logger:
            wandb_logger.finalize(exit_code=final_status_code)
        logger.info("Training completed.")
    
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()


