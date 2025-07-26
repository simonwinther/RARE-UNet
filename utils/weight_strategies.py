from typing import List, Callable, Dict


def _compute_power_of_two_weights(depth: int) -> List[float]:
    """Computes weights decreasing by powers of two for each level."""
    weights = [
        pow(2.0, -float(index)) for index in range(depth)
    ]  # Start from 0 for pow(2,0)=1
    # Normalize weights
    total = sum(weights)
    if total == 0:  # Avoid division by zero if depth is 0 or less (shouldn't happen)
        return []
    return [w / total for w in weights]


def _compute_linear_weights(depth: int) -> List[float]:
    """Computes equal weights for each level."""
    if depth <= 0:
        return []
    weight = 1.0 / float(depth)
    return [weight] * depth


WEIGHT_STRATEGIES: Dict[str, Callable[[int], List[float]]] = {
    "power_of_two": _compute_power_of_two_weights,
    "equal": _compute_linear_weights,
}


def compute_weights_depth(depth: int, strategy: str = "power_of_two") -> List[float]:
    """
    Get the weights for deep supervision outputs based on the depth and chosen strategy.

    Args:
        depth (int): Number of output levels.
            strategy (str): The weighting strategy to use ('power_of_two', 'linear', 'equal').
                            Defaults to 'power_of_two'.

    Returns:
        list: Normalized weights for each output level.
    """
    strategy_fn = WEIGHT_STRATEGIES.get(strategy.lower())
    if strategy_fn is None:
        raise ValueError(
            f"Unknown weighting strategy: '{strategy}'. "
            f"Available strategies: {list(WEIGHT_STRATEGIES.keys())}"
        )
    return strategy_fn(depth)
