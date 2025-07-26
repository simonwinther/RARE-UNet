# wandb_logger.py
from typing import Any, Optional
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import os  # Added for path handling

import logging

logger = logging.getLogger(__name__)

class WandBLogger:
    """Handles all WandB interactions in one place"""

    def __init__(
        self,
        config: DictConfig,
        model: Optional[torch.nn.Module] = None,
        model_summary: Optional[Any] = None,
        resume_id: Optional[str] = None,
    ):
        """
        Initializes Weights & Biases logging.
        # ... (Args documentation remains the same) ...
        """
        wandb_ctx = config.get("wandb", {})
        
        resolved_config = OmegaConf.to_container(config, resolve=True)
        _project = wandb_ctx.wandb_project
        _name =  wandb_ctx.name
        _group = wandb_ctx.group
        _tags =  wandb_ctx.tags

        init_kwargs = {
            "project": _project,
            "name": _name,
            "config": resolved_config,
            "group": _group,
            "tags": _tags,
        }

        if resume_id:
            logger.info(f"Resuming WandB run with ID: {resume_id}")
            init_kwargs["id"] = resume_id
            init_kwargs["resume"] = "allow" 
            
        try:
            self.run = wandb.init(**init_kwargs)
            self.run_id = self.run.id
        except Exception as e:
            print(f"Error initializing WandB: {e}")
            self.run = None  

        if self.run and model is not None:
            self.run.define_metric("epoch")
            self.run.define_metric("*", step_metric="epoch")

            # Log gradients and parameters
            wandb.watch(model, log="all", log_freq=100) 

            if model_summary is not None:
               total_params       = model_summary.total_params
               trainable_params   = model_summary.trainable_params
               non_trainable_params = total_params - trainable_params
               self.run.config.update({
                   "total_params": total_params,
                   "trainable_params": trainable_params,
                   "non_trainable_params": non_trainable_params
            }, allow_val_change=True)

    def log_metrics(self, metrics: dict, step: int | None = None, commit: bool = False):
        """Log metrics with optional step parameter."""
        if self.run:
            self.run.log(metrics, step=step, commit=commit)

    def finalize(self, exit_code: int | None = None):
        """Finishes the WandB run."""
        if self.run:
            self.run.finish(exit_code=exit_code)
            print("WandB run finished.")

    @property
    def is_active(self) -> bool:
        """Check if the WandB run is active."""
        return self.run is not None


def get_wandb_logger(
    cfg: DictConfig, 
    model: Optional[torch.nn.Module] = None, 
    model_summary: Optional[Any] = None,
    resume_id: Optional[str] = None
) -> WandBLogger | None:
    """
    Initializes and returns a WandBLogger instance based on config.
    Returns None if wandb is disabled in the config.
    """
    if cfg.wandb.get("log", False): 
        return WandBLogger(
            config=cfg, 
            model=model, 
            model_summary=model_summary,
            resume_id=resume_id
        )
    else:
        logger.info("Wandb is disabled. No logger will be created.")
        return None
