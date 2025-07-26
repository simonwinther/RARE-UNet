from typing import Optional
from tabulate import tabulate
import numpy as np

def print_table(data, headers=("Property", "Value"), tablefmt="pretty"):
    """
    Prints a dictionary or list of (key, value) pairs in tabular format.

    Args:
        data: A dictionary {key: value} or a list of [key, value].
        headers (tuple): Column headers.
        tablefmt (str): A valid tabulate table format.
    """
    # If data is a dictionary, convert to list of [key, value].
    if isinstance(data, dict):
        data = [[k, v] for k, v in data.items()]
    # If data is already a list of pairs, we assume it's fine.
    return tabulate(data, headers=headers, tablefmt=tablefmt)


def print_norm_image_stats(normalized_image):
    stats = {
        "Shape": normalized_image.shape,
        "Min pixel value": np.min(normalized_image),
        "Max pixel value": np.max(normalized_image),
        "Mean pixel value": np.mean(normalized_image),
        "Std. Deviation": np.std(normalized_image),
        "Variance": np.var(normalized_image)
    }
    return print_table(stats)


def print_train_val_table(
    train_metrics: dict,
    val_metrics: dict,
    metrics_order: Optional[list] = None,
    headers: tuple = ("Metric", "Train", "Validation"),
    floatfmt: str = ".4f",
    tablefmt: str = "pretty",
):
    # keys = metrics_order or list(set(train_metrics) | set(val_metrics))
    # the following is uglier, but preserves order of metrics.summary()
    train_keys = list(train_metrics.keys())
    val_keys = [k for k in val_metrics.keys() if k not in train_keys]
    keys = metrics_order or (train_keys + val_keys)
    
    rows = []

    def _fmt(val):
        if isinstance(val, int):
            return str(val)
        elif isinstance(val, float):
            return f"{val:{floatfmt}}"
        elif isinstance(val, list):
            if all(isinstance(v, int) for v in val):
                return "[" + ", ".join(str(v) for v in val) + "]"
            elif all(isinstance(v, float) for v in val):
                return "[" + ", ".join(f"{v:{floatfmt}}" for v in val) + "]"
            elif all(isinstance(v, list) for v in val):
                return "[" + ", ".join("[" + ", ".join(f"{x:{floatfmt}}" for x in v) + "]" for v in val) + "]"
            else:
                return str(val)
        elif val is None:
            return "nan"
        else:
            return str(val)

    for m in keys:
        t = train_metrics.get(m)
        v = val_metrics.get(m)
        rows.append([m, _fmt(t), _fmt(v)])
    return print_table(rows, headers=headers, tablefmt=tablefmt)
 