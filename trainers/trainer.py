# Standard Library
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Libraries
import torch
from omegaconf import DictConfig
from utils.table import print_train_val_table

# PyTorch
from torch.distributed import barrier
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.distributed as dist
import torch.nn.functional as F
from torch import is_distributed
from torch.cuda.amp import autocast, GradScaler

# Project-Specific Modules
from data.datasets import MedicalDecathlonDataset
from trainers.early_stopping import EarlyStopping
from utils import metrics  # Assuming your metrics live here
from utils.checkpoint_handler import CheckpointHandler
from utils.metric_collecter import Agg, MetricCollector
from utils.metrics import dice_coefficient_classes, safe_mean
from utils.wandb_logger import WandBLogger
# from .trainer_registry import register_trainer

# Initialize logging
logger = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        ckpt_handler: CheckpointHandler,
        wandb_logger: Optional[WandBLogger],
        is_distributed: bool = False,
        rank: int = 0,
        is_main: bool = True,
        resume_path: Optional[str] = None
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        self.device = device
        self.ckpt_handler = ckpt_handler
        self.wandb_logger = wandb_logger
        self.is_distributed = is_distributed
        self.rank = rank
        self.is_main = is_main
        self.resume_path = resume_path

        self.logger = logging.getLogger(__name__)
        self.num_epochs = cfg.training.num_epochs
        self.num_classes = cfg.dataset.num_classes
        self.class_labels = {i: f"class_{i}" for i in range(self.num_classes)}
        self.ignore_index = cfg.dataset.get("ignore_index", 0)
        self.max_norm = self.cfg.training.get("grad_clip_norm", 1.0)
        
        # Setup early stopping if configured
        self.early_stopper = None
        early_cfg = cfg.training.get("early_stopper", None)
        if early_cfg:
            self.early_stopper = EarlyStopping(
                patience=early_cfg.get("patience", 15),
                delta=early_cfg.get("delta", 0.0),
                criterion=early_cfg.get("criterion", "loss"),
                verbose=early_cfg.get("verbose", True),
            )
        if not self.early_stopper:
            self.logger.info("No early stopping configured.")
        
        
        self.improved = False
        self.start_epoch = 0
        self.best_metric = None

        if resume_path:
            self.logger.info(f"Resuming from checkpoint: {resume_path}")

            checkpoint = self.ckpt_handler.load(
                resume_path, 
                model=self.model, 
                optimizer=self.optimizer, 
                scheduler=self.lr_scheduler
            )
            
            self.start_epoch = checkpoint.get("epoch", 0)
            self.best_metric = checkpoint.get("best_metric", None)
            self.logger.info(f"Resumed from epoch {self.start_epoch}")
            
            if self.best_metric is not None:
                self.logger.info(f"Best metric from checkpoint: {self.best_metric:.4f}")
                if self.early_stopper:
                    self.early_stopper.best_score = self.best_metric
                    self.logger.info(f"Early stopping restored with best score: {self.early_stopper.best_score:.4f}")

        # Setup metric collector
        self.metrics = ["dice", "iou", "precision", "recall", "f1"]
        self.mc = MetricCollector()
        self._setup_metric_collector_rules()

        self.use_amp = cfg.get("use_amp", False) and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.GradScaler(device.type)
            self.logger.info("Employing mixed precision training with torch.cuda.amp.GradScaler.")
        self.logger.info(f"BaseTrainer initialized for training over {self.num_epochs} epochs.")

    def _setup_metric_collector_rules(self):
        """Sets up the basic rules for the metric collector. Can be overridden."""
        self.mc.set_rule("loss", Agg.MEAN)
        for metric in self.metrics:
            self.mc.set_rule(f"{metric}_fullres", Agg.MEAN)
            self.mc.set_rule(f"{metric}_per_class_fullres", Agg.LIST_MEAN)
        self.logger.debug("Base metric collector rules set.")

    def train(self):
        start_time = time.time()
        disable = not self.is_main # Only show bar on main process

        for epoch in trange(
            self.start_epoch,
            self.num_epochs, 
            desc="Epoch [Training Progress]",
            disable=disable,
            position=0,
            leave=True,
            dynamic_ncols=True
        ):
            self.on_epoch_start(epoch)
            
            epoch_start_time = time.time()
            train_dict = self.train_one_epoch(epoch)
            train_dict["epoch_time"] = time.time() - epoch_start_time

            epoch_start_time = time.time()
            val_dict = self.validate(epoch)
            val_dict["epoch_time"] = time.time() - epoch_start_time

            if self.is_distributed:
                self.logger.debug(f"[R{self.rank}] Dice (Tr/Val): {train_dict['dice_fullres']:.4f}/{val_dict['dice_fullres']:.4f}")
                train_dict = self.mc.reduce_across_ranks(train_dict, self.device)
                val_dict = self.mc.reduce_across_ranks(val_dict, self.device)
                self.logger.debug(f"[R{self.rank}] Reduced Dice (Tr/Val): {train_dict['dice_fullres']:.4f}/{val_dict['dice_fullres']:.4f}")

            # ----------- DELETE ME DEBUG -----------
            if self.is_main:
                train_dice = train_dict.get("dice_fullres", 0.0)
                val_dice = val_dict.get("dice_fullres", 0.0)
                epoch_time = train_dict['epoch_time']
                
                # tqdm.write is safe to use with active tqdm bars
                tqdm.write(f"--- Epoch {epoch+1} Summary ---")
                tqdm.write(f"    Epoch Time: {epoch_time:.2f} seconds")
                tqdm.write(f"    Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")
                tqdm.write(f"----------------------")
            # ----------- DELETE ME DEBUG -----------
            
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    metric_to_reduce_lr = val_dict["dice_fullres"]
                    self.lr_scheduler.step(metric_to_reduce_lr)
                else:
                    self.lr_scheduler.step()
            
            train_dict["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            if self.wandb_logger:
                self.log_to_wandb(train_dict, step=epoch, prefix="train", commit=False)
                self.log_to_wandb(val_dict, step=epoch, prefix="val", commit=True)

            if self.cfg.logging.get("ppmat", False):
                table_str = print_train_val_table(train_dict, val_dict)
                tqdm.write(f"\nEpoch {epoch + 1} Results:\n{table_str}")


            stop_early = False
            if self.early_stopper:
                crit_name = self.early_stopper.criterion
                assert crit_name in val_dict, f"Early stopping criterion '{crit_name}' not found in validation metrics."

                if self.is_main:
                    stop_early, improved = self.early_stopper(
                        val_dict["loss"], 
                        val_dict[crit_name]
                    )
                    self.improved = improved
                    self.best_metric = self.early_stopper.best_score

                if self.is_distributed:
                    stop_signal = torch.tensor(1.0 if stop_early else 0.0, device=self.device)
                    dist.broadcast(stop_signal, src=0)
                    if stop_signal.item() == 1.0:
                        stop_early = True
                        
            self.on_epoch_end(epoch)
            if stop_early:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1} by rank {self.rank}.")
                break
                
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.2f} minutes.")

    def train_one_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.train()
        self.current_epoch = epoch
        self.mc.reset()

        for images, masks in tqdm(
            self.train_dataloader, 
            desc=f"Epoch {epoch+1} [R{self.rank} | Train]", 
            # position=self.rank + 1,
            position=1, # delete me and uncomment position
            disable=not self.is_main, #delete me and uncomment position
            dynamic_ncols=True,
            leave=False
        ):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self._compute_loss(outputs, masks)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(
                    self.optimizer
                )  # Unscale gradients before clipping
                clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_norm
                )  # Or whatever parameters you clip
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
            batch_metrics = self._compute_metrics(outputs, masks)
            batch_metrics["loss"] = loss.item()
            self.mc.update(batch_metrics)
        return self.mc.aggregate()

    def validate(self, epoch: int) -> Dict[str, Any]:
        self.model.eval()
        self.mc.reset()
        with torch.no_grad():
            for images, masks in tqdm(
                self.val_dataloader, 
                desc=f"Epoch {epoch+1} [R{self.rank} | Val]", 
                # position=self.rank + 1,
                position=1, # delete me and uncomment position
                disable=not self.is_main, #delete me and uncomment position
                dynamic_ncols=True,
                leave=False,
            ):
                images, masks = images.to(self.device), masks.to(self.device)

                with autocast(self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, masks)

                batch_metrics = self._compute_metrics(outputs, masks)
                batch_metrics["loss"] = loss.item()
                self.mc.update(batch_metrics)
        return self.mc.aggregate()

    def _compute_loss(self, outputs, masks) -> torch.Tensor:
        if isinstance(outputs, (list, tuple)):
            raise TypeError(
                "Backbone-UNet: Output should be a single tensor, not a list or tuple."
            )
        return self.criterion(outputs, masks)

    def log_to_wandb(
        self, metrics: Dict[str, Any], step: int, prefix: str, commit: bool
    ):
        if not self.is_main:
            # ensures only main process logs to wandb
            return   
        if self.wandb_logger is None or not self.wandb_logger.is_active:
            return

        log_dict: Dict[str, Any] = {"epoch": step}

        for key, value in metrics.items():
            full_key = f"{prefix}/{key}"

            if isinstance(value, (float, int)):
                log_dict[full_key] = value
            elif isinstance(value, list) and all(
                isinstance(v, (float, int)) for v in value
            ):
                # We always assume that a list of metric means is per-class
                log_dict[full_key] = {f"class_{i}": v for i, v in enumerate(value)}

        self.wandb_logger.log_metrics(log_dict, step=step, commit=commit)

    def _compute_metrics(
        self, outputs: torch.Tensor, masks: torch.Tensor
    ) -> Dict[str, Any]:
        if isinstance(outputs, (list, tuple)):
            raise TypeError(
                "Backbone-UNet: Output should be a single tensor, not a list or tuple."
            )
        preds = torch.argmax(outputs, dim=1)

        dice_scores_cls = metrics.dice_coefficient_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        iou_scores_cls = metrics.iou_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        precision_scores_cls = metrics.precision_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        recall_scores_cls = metrics.recall_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )
        f1_scores_cls = metrics.f1_score_classes(
            preds, masks, self.num_classes, ignore_index=self.ignore_index
        )

        return {
            "dice_fullres": safe_mean(dice_scores_cls),
            "iou_fullres": safe_mean(iou_scores_cls),
            "precision_fullres": safe_mean(precision_scores_cls),
            "recall_fullres": safe_mean(recall_scores_cls),
            "f1_fullres": safe_mean(f1_scores_cls),
            "dice_per_class_fullres": [x.item() for x in dice_scores_cls],
            "iou_per_class_fullres": [x.item() for x in iou_scores_cls],
            "precision_per_class_fullres": [x.item() for x in precision_scores_cls],
            "recall_per_class_fullres": [x.item() for x in recall_scores_cls],
            "f1_per_class_fullres": [x.item() for x in f1_scores_cls],
        }
    
    def on_epoch_start(self, epoch: int):
        """
        Called at the start of each epoch. Can be overridden for custom behavior.
        """
        self.improved = False
        if self.is_distributed:
            sampler = getattr(self.train_dataloader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            else: 
                self.logger.warning(
                    f"Sampler {type(sampler)} does not support set_epoch(). "
                    "This may lead to non-shuffled data in distributed training."
                )
    
    def on_epoch_end(self, epoch: int):
        """
        Called at the end of each epoch. Can be overridden for custom behavior.
        """
        # sync all ranks
        if self.is_distributed:
            torch.distributed.barrier()

        wandb_id_to_save = None
        
        if self.wandb_logger and self.wandb_logger.is_active: 
            wandb_id_to_save = self.wandb_logger.run_id
        
        if self.is_main:
            last_path = self.ckpt_handler.save(
                kind="last",
                model=self.model,
                epoch=epoch + 1,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                best_metric_val=self.best_metric,
                wandb_run_id=wandb_id_to_save
            )
            self.logger.info(f"Saved LAST checkpoint: {last_path}")

            if getattr(self, "improved", False):
                best_path = self.ckpt_handler.save(
                    kind="best",
                    model=self.model,
                    epoch=epoch + 1,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    best_metric_val=self.best_metric,
                    wandb_run_id=wandb_id_to_save
                )
                self.logger.info(f"Saved BEST checkpoint: {best_path}")