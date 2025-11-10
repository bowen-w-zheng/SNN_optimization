"""
Intensification and feasibility screening logic from the paper.
"""

import jax.numpy as jnp
from typing import Dict, NamedTuple
import chex


class FeasibilityThresholds(NamedTuple):
    """Thresholds for feasibility screening."""

    min_fr: float = 1.0  # Minimum population firing rate (sp/s)
    max_ff: float = 5.0  # Maximum Fano factor
    min_dsh: float = 1.0  # Minimum dimensionality of shared variance


class IntensificationConfig(NamedTuple):
    """Configuration for intensification."""

    max_reps: int = 5  # Maximum repetitions (R)
    sd_threshold: float = 0.15  # SD threshold for early stopping
    potential_threshold: float = 1.0  # SD multiplier for "potentially optimal"


def check_feasibility(
    stats: Dict,
    thresholds: FeasibilityThresholds = FeasibilityThresholds(),
) -> tuple[bool, str]:
    """
    Check if simulation statistics indicate a feasible parameter set.

    Args:
        stats: Dictionary of statistics (fr, ff, dsh, etc.)
        thresholds: Feasibility thresholds

    Returns:
        (is_feasible, reason)
        is_feasible: True if feasible
        reason: String explaining why infeasible (empty if feasible)
    """
    # Check firing rate
    if "fr" in stats and stats["fr"] < thresholds.min_fr:
        return False, f"fr too low: {stats['fr']:.2f} < {thresholds.min_fr}"

    # Check Fano factor
    if "ff" in stats and stats["ff"] > thresholds.max_ff:
        return False, f"ff too high: {stats['ff']:.2f} > {thresholds.max_ff}"

    # Check dimensionality of shared variance
    if "dsh" in stats and stats["dsh"] < thresholds.min_dsh:
        return False, f"dsh too low: {stats['dsh']} < {thresholds.min_dsh}"

    return True, ""


def check_stability(
    fr_timeseries: chex.Array,
    window_size: int = 10,
    stability_threshold: float = 5.0,
) -> bool:
    """
    Check if network activity has stabilized.

    Uses a simple change-point detector: checks if mean FR changes
    beyond K×SD within the simulation.

    Args:
        fr_timeseries: Firing rate over time bins (n_bins,)
        window_size: Number of bins for rolling window
        stability_threshold: Threshold in SD units for change detection

    Returns:
        True if stable, False if unstable
    """
    if len(fr_timeseries) < 2 * window_size:
        return True  # Too short to judge

    # Compute rolling mean
    n_bins = len(fr_timeseries)

    early_window = fr_timeseries[:window_size]
    late_window = fr_timeseries[-window_size:]

    mean_early = jnp.mean(early_window)
    mean_late = jnp.mean(late_window)

    # Global SD
    global_sd = jnp.std(fr_timeseries) + 1e-6

    # Check if change exceeds threshold
    change = jnp.abs(mean_late - mean_early)
    is_stable = change < stability_threshold * global_sd

    return bool(is_stable)


def is_potentially_optimal(
    cost: float,
    incumbent_mean: float,
    incumbent_sd: float,
    config: IntensificationConfig = IntensificationConfig(),
) -> bool:
    """
    Check if a candidate is "potentially optimal" (paper's intensification logic).

    A candidate is potentially optimal if its first-repetition cost is within
    K×SD of the incumbent's mean cost.

    Args:
        cost: Cost from first repetition
        incumbent_mean: Mean cost of current incumbent
        incumbent_sd: SD of incumbent's costs across repetitions
        config: Intensification configuration

    Returns:
        True if potentially optimal
    """
    threshold = incumbent_mean + config.potential_threshold * incumbent_sd
    return cost <= threshold


def should_stop_early(
    costs: chex.Array, config: IntensificationConfig = IntensificationConfig()
) -> bool:
    """
    Check if intensification should stop early.

    Stops if SD of observed costs falls below threshold.

    Args:
        costs: Array of costs from repetitions so far (k,)
        config: Intensification configuration

    Returns:
        True if should stop
    """
    if len(costs) < 2:
        return False

    sd = jnp.std(costs)
    return sd < config.sd_threshold


def intensification_loop(
    evaluate_fn: callable,
    theta: chex.Array,
    incumbent_mean: float,
    incumbent_sd: float,
    config: IntensificationConfig = IntensificationConfig(),
) -> tuple[chex.Array, float, float, bool]:
    """
    Run intensification loop for a candidate parameter set.

    Args:
        evaluate_fn: Function that evaluates theta and returns (cost, stats)
        theta: Parameter set to evaluate
        incumbent_mean: Mean cost of current incumbent
        incumbent_sd: SD of incumbent
        config: Intensification configuration

    Returns:
        (costs, mean_cost, sd_cost, is_new_incumbent)
        costs: Array of costs from all repetitions
        mean_cost: Mean cost
        sd_cost: SD of costs
        is_new_incumbent: True if this becomes new incumbent
    """
    costs = []

    # First repetition
    cost_0, stats_0 = evaluate_fn(theta, rep=0)
    costs.append(cost_0)

    # Check if potentially optimal
    if not is_potentially_optimal(cost_0, incumbent_mean, incumbent_sd, config):
        # Not worth intensifying
        return jnp.array(costs), float(cost_0), 0.0, False

    # Additional repetitions
    for rep in range(1, config.max_reps):
        cost_rep, _ = evaluate_fn(theta, rep=rep)
        costs.append(cost_rep)

        # Check early stopping
        if should_stop_early(jnp.array(costs), config):
            break

    # Compute statistics
    costs_array = jnp.array(costs)
    mean_cost = float(jnp.mean(costs_array))
    sd_cost = float(jnp.std(costs_array))

    # Check if new incumbent
    is_new_incumbent = mean_cost < incumbent_mean

    return costs_array, mean_cost, sd_cost, is_new_incumbent


class ShortRunCheck(NamedTuple):
    """Result from short feasibility run."""

    is_feasible: bool
    reason: str
    mean_fr_e: float
    mean_fr_i: float


def run_feasibility_check(
    simulate_fn: callable,
    theta: chex.Array,
    thresholds: FeasibilityThresholds = FeasibilityThresholds(),
) -> ShortRunCheck:
    """
    Run short simulation for feasibility checking.

    Args:
        simulate_fn: Function that simulates network and returns (fr_e, fr_i, stats)
        theta: Parameter set
        thresholds: Feasibility thresholds

    Returns:
        ShortRunCheck result
    """
    # Run short simulation (e.g., 10 seconds)
    fr_e, fr_i, stats = simulate_fn(theta)

    # Compute population firing rate
    mean_fr = float(jnp.mean(jnp.concatenate([fr_e, fr_i])))
    stats["fr"] = mean_fr

    # Check feasibility
    is_feasible, reason = check_feasibility(stats, thresholds)

    return ShortRunCheck(
        is_feasible=is_feasible,
        reason=reason,
        mean_fr_e=float(jnp.mean(fr_e)),
        mean_fr_i=float(jnp.mean(fr_i)),
    )
