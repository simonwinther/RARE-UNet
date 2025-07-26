# utils/metric_collector.py

import logging
from enum import Enum
from collections import defaultdict
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch

import torch.distributed as dist


class Agg(Enum):
    MEAN = "mean"
    SUM = "sum"
    LAST = "last"
    LIST_MEAN = "list_mean"
    RAW = "raw"


def _agg_raw(vals: List[Any]) -> str:
    return ",".join(map(str, vals))


def _agg_mean(vals: List[float]) -> Union[float, None]:
    return float(np.mean(vals)) if vals else None


def _agg_sum(vals: List[float]) -> Union[float, None]:
    return float(np.sum(vals)) if vals else None


def _agg_last(vals: List[Any]) -> Any:
    return vals[-1] if vals else None


def _agg_list_mean(vals: List[Any]) -> Any:
    """
    Compute the mean over a list of scalars or a list of equal-length lists (or higher-d arrays):
      - 1D list of numbers -> float mean
      - ND list            -> elementwise mean over the first axis
    """
    arr = np.array(vals, dtype=float)
    if arr.ndim == 1:
        return float(np.nanmean(arr)) if vals else None
    # for 2D, 3D, etc: mean across axis=0, return nested lists
    return np.nanmean(arr, axis=0).tolist()


# map each Agg to its function
_AGG_FN: Dict[Agg, Callable[[List[Any]], Any]] = {
    Agg.RAW: _agg_raw,
    Agg.MEAN: _agg_mean,
    Agg.SUM: _agg_sum,
    Agg.LAST: _agg_last,
    Agg.LIST_MEAN: _agg_list_mean,
}


class MetricCollector:
    """
    Collect per-batch scalar lists and collapse them at the end.
    """

    def __init__(self):
        self._data: Dict[str, List[Any]] = defaultdict(list)
        self._rules: Dict[str, Agg] = {}
        self._processed = 0
        self._skipped = 0

    def reset(self) -> None:
        self._data.clear()
        self._processed = 0
        self._skipped = 0

    def set_rule(self, name: str, agg: Union[Agg, str]) -> None:
        if isinstance(agg, str):
            agg = Agg(agg)
        self._rules[name] = agg

    def update(self, batch_metrics: Dict[str, Any]) -> None:
        """
        batch_metrics must contain only Python numbers or nested lists thereof.
        """
        for k, v in batch_metrics.items():
            self._processed += 1
            self._data[k].append(v)

    def skip(self) -> None:
        self._skipped += 1

    def aggregate(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, vals in self._data.items():
            rule = self._rules.get(name, Agg.MEAN)
            fn = _AGG_FN[rule]
            out[name] = fn(vals) if vals else None

        out["num_processed"] = self._processed
        out["num_skipped"] = self._skipped
        return out

    def reduce_across_ranks(self, metrics: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Reduce scalar and list metrics across all DDP ranks using all_reduce.
        Assumes that each rank has already computed per-rank values.
        """
        if not dist.is_initialized():
            return metrics  # not in DDP

        world_size = dist.get_world_size()
        reduced = {}

        for key, val in metrics.items():
            # Only reduce numeric values
            if isinstance(val, (float, int)):
                t = torch.tensor(val, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                reduced[key] = t.item() / world_size

            elif isinstance(val, list) and all(isinstance(v, (float, int)) for v in val):
                t = torch.tensor(val, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                reduced[key] = (t / world_size).tolist()

            else:
                reduced[key] = val  # skip strings, dicts, None, etc.
        return reduced
