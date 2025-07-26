# utils/checkpoint_handler.py
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict
import torch
import logging
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

class CheckpointHandler:
    def __init__(self, cfg: DictConfig):
        # Basic setup for the CheckpointHandler
        self.cfg        = cfg
        self.model_name = cfg.architecture.name or ""
        self.task_name  = cfg.dataset.name or ""
        self.wandb_name = cfg.get('wandb', {}).get('name', 'default') 
        self.timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logger     = logging.getLogger(f"{self.task_name}_{self.timestamp}")
        
        missing_required_config = not self.model_name or not self.task_name
        if missing_required_config:
            raise ValueError(
                "Configuration must contain 'architecture.name' and 'dataset.name'."
            )
            
        # build out directories
        self.base_exp_dir = Path("trained_models") / self.model_name / self.task_name
        self.exp_dir      = self.base_exp_dir / f"{self.timestamp}_{self.wandb_name}"
        self.config_path  = self.exp_dir / "config.yaml"
        self._paths = {
            "best": self.exp_dir / "best_model.pth",
            "last": self.exp_dir / "last_model.pth",
        }
     
        # create experiment directory
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # save config.yaml
        try: 
            OmegaConf.save(self.cfg, self.config_path)
        except Exception as e:
            self.logger.error(f"Failed saving config: {e}")
            raise 
        
        self.logger.info(f"CheckpointHandler: {self.model_name} on {self.task_name}")
        self.logger.info(f"Experiment dir: {self.exp_dir}")

    def save(
        self,
        kind: str,
        model: torch.nn.Module | DDP,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        best_metric_val: Optional[float] = None,
        wandb_run_id: Optional[str] = None,
    ) -> Path:
        """
        Save a checkpoint of `kind` ("best" or "last").
        This checkpoint includes the epoch, model state, optimizer state,
        and there are some optional parameters described below:
        
        - lr_scheduler   : If provided, its state will also be saved, 
                            If no lr scheduler is used, None will be passed.
        - best_metric_val: If this is a "best" checkpoint, this should be the
                            best metric value achieved so far. If this is a "last"
                            checkpoint, this can be None. Also, if early stopping 
                            is disabled, this will always be None.
        - wandb_run_id   : If you are using Weights & Biases, this is the run ID of the current run.
                            This is optional, and if wandb is disabled, None will be passed.
        """
        
        if kind not in self._paths:
            raise ValueError(f"Unknown checkpoint kind {kind!r}, choose from {list(self._paths)}")

        net = model.module if isinstance(model, DDP) else model


        ckpt: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }


        if lr_scheduler:
            ckpt["scheduler_state_dict"] = lr_scheduler.state_dict()

        if best_metric_val is not None:
            ckpt["best_metric"] = best_metric_val
            
        if wandb_run_id:
            ckpt["wandb_run_id"] = wandb_run_id

        path = self._paths[kind]
        torch.save(ckpt, path)
        self.logger.debug(f"Saved {kind.upper()} checkpoint at epoch={epoch} to {path}")
        return path

    def load(
        self,
        path_str: str,
        model: torch.nn.Module | DDP,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Loads state into model/optimizer/scheduler from a specific path
        and returns the full checkpoint dictionary.
        """
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file {path} does not exist.")

        self.logger.info(f"Loading checkpoint and restoring state from: {path}")
        ckpt = torch.load(path, map_location=map_location)

        net = model.module if isinstance(model, DDP) else model
        net.load_state_dict(ckpt["model_state_dict"])
        
        if optimizer and ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.logger.debug(f"Successfully restored state from checkpoint at {path}")
        return ckpt
    
    def get_resume_id_from_checkpoint(self, path_str: str) -> Optional[str]:
        """
        Extracts the wandb run ID from a checkpoint file if it exists.
        Returns None if no run ID is found.
        """
        path = Path(path_str)
        if not path.exists():
            self.logger.warning(f"Checkpoint file {path} does not exist.")
            return None

        ckpt = torch.load(path, map_location="cpu")
        return ckpt.get("wandb_run_id", None)
    
    def path(self, kind: str) -> Optional[Path]:
        """Return the Path for 'best' or 'last', or None if it doesn’t exist."""
        p = self._paths.get(kind)
        return p if p and p.exists() else None