# utils/callbacks.py (or wherever you prefer)
import logging


# utils/callbacks.py
class EarlyStopping:
    """
    Monitors a single validation metric and stops training when it
    stops improving. Supports minimizing (loss) or maximizing (all others).
    """
    def __init__(self, patience: int, delta: float, criterion: str, verbose: bool):
        """
        Args:
          patience (int): epochs to wait after last improvement
          delta (float): minimum change to count as an improvement
          criterion (str): name of the metric you’ll pass in 
                           (e.g. "loss" or "dice_multiscale_avg")
          verbose (bool): whether to log progress
        """
        self.patience   = patience
        self.delta      = delta
        self.criterion  = criterion
        self.verbose    = verbose
        self.counter    = 0
        self.early_stop = False

        # determine whether lower is better (loss) or higher is better (all other metrics)
        if criterion == "loss":
            self.mode       = "min"
            self.best_score = float("inf")
        else:
            self.mode       = "max"
            self.best_score = -float("inf")

        self.logger = logging.getLogger(__name__)

    def _is_improved(self, score: float) -> bool:
        """Return True if `score` is an improvement over best_score."""
        if self.mode == "min":
            return score < self.best_score - self.delta
        else:  # mode == "max"
            return score > self.best_score + self.delta

    def __call__(self, val_loss: float, val_metric: float):
        """
        Check for improvement and update internal state.

        Args:
          val_loss (float): the latest validation loss
          val_metric (float): the latest value of the monitored metric
                              (e.g. dice_multiscale_avg)

        Returns:
          (early_stop: bool, improved: bool)
        """
        # pick the score to compare
        score = val_loss if self.mode == "min" else val_metric

        if self._is_improved(score):
            if self.verbose:
                self.logger.info(
                    f"{self.criterion} improved "
                    f"({self.best_score:.4f} -> {score:.4f})"
                )
            self.best_score = score
            self.counter    = 0
            improved        = True
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(
                    f"No improvement in {self.criterion} "
                    f"({self.counter}/{self.patience})"
                )
            improved = False
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.logger.info("Early stopping triggered")

        return self.early_stop, improved
