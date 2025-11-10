"""
Cost function for SNOPS optimization (paper equation 7).
"""

import jax.numpy as jnp
from typing import Dict, NamedTuple
import chex

from snops_jax.stats.single_pair import (
    compute_fr,
    compute_ff,
    compute_rsc,
    fisher_z_transform,
)
from snops_jax.stats.fa_jax import compute_shared_variance_stats


class CostConfig(NamedTuple):
    """Configuration for cost function."""

    # Which statistics to include
    use_fr: bool = True
    use_ff: bool = True
    use_rsc: bool = True
    use_pct_sh: bool = True
    use_dsh: bool = True
    use_es: bool = True  # Eigenspectrum

    # Weights for each statistic
    w_fr: float = 1.0
    w_ff: float = 1.0
    w_rsc: float = 1.0
    w_pct_sh: float = 1.0
    w_dsh: float = 1.0
    w_es: float = 1.0

    # Parameters
    bin_size: float = 200.0  # ms
    min_fr: float = 0.5  # sp/s
    dsh_threshold: float = 0.95  # For FA dimensionality
    n_es: int = 50  # Number of eigenspectrum values to compare


class TargetStatistics(NamedTuple):
    """Target statistics from recordings."""

    # Means
    fr_mean: float
    ff_mean: float
    rsc_z_mean: float  # Fisher z-transformed
    pct_sh_mean: float
    dsh_mean: float
    es_mean: chex.Array  # (n_es,) eigenspectrum

    # Variances (for normalization)
    fr_var: float
    ff_var: float
    rsc_z_var: float
    pct_sh_var: float
    dsh_var: float
    es_var: float  # Variance of sum of squared differences


def compute_cost(
    spike_counts: chex.Array,
    target: TargetStatistics,
    config: CostConfig,
    rng_key: chex.PRNGKey = None,
) -> tuple[float, dict]:
    """
    Compute cost function c_S(θ) from paper equation 7.

    Cost = (1 / Σw_j) * Σ w_j * d(s_j^true, s_j(θ)) / v_j^true

    where d(·,·) is squared difference, v_j^true is variance across sessions.

    Args:
        spike_counts: (n_neurons, n_bins) spike counts from simulation
        target: Target statistics from recordings
        config: Cost configuration
        rng_key: Random key for FA

    Returns:
        (cost, statistics_dict)
        cost: Scalar cost value
        statistics_dict: Dictionary of computed statistics
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Compute statistics from simulation
    stats = {}
    cost_terms = []
    weights = []

    # 1. Firing rate
    if config.use_fr:
        fr_sim = compute_fr(spike_counts, config.bin_size)
        stats["fr"] = fr_sim

        diff_sq = (fr_sim - target.fr_mean) ** 2
        normalized = diff_sq / (target.fr_var + 1e-10)
        cost_terms.append(normalized)
        weights.append(config.w_fr)

    # 2. Fano factor
    if config.use_ff:
        ff_sim = compute_ff(spike_counts, config.min_fr, config.bin_size)
        stats["ff"] = ff_sim

        diff_sq = (ff_sim - target.ff_mean) ** 2
        normalized = diff_sq / (target.ff_var + 1e-10)
        cost_terms.append(normalized)
        weights.append(config.w_ff)

    # 3. Spike count correlation (Fisher z)
    if config.use_rsc:
        rsc_sim = compute_rsc(spike_counts, config.min_fr, config.bin_size)
        rsc_z_sim = fisher_z_transform(rsc_sim)
        stats["rsc"] = rsc_sim
        stats["rsc_z"] = rsc_z_sim

        diff_sq = (rsc_z_sim - target.rsc_z_mean) ** 2
        normalized = diff_sq / (target.rsc_z_var + 1e-10)
        cost_terms.append(normalized)
        weights.append(config.w_rsc)

    # 4. Shared variance statistics (via FA)
    if config.use_pct_sh or config.use_dsh or config.use_es:
        fa_stats = compute_shared_variance_stats(
            spike_counts, n_factors=None, dsh_threshold=config.dsh_threshold, rng_key=rng_key
        )

        stats.update(fa_stats)

        # Percent shared variance
        if config.use_pct_sh:
            pct_sh_sim = fa_stats["pct_sh"]

            diff_sq = (pct_sh_sim - target.pct_sh_mean) ** 2
            normalized = diff_sq / (target.pct_sh_var + 1e-10)
            cost_terms.append(normalized)
            weights.append(config.w_pct_sh)

        # Dimensionality of shared variance
        if config.use_dsh:
            dsh_sim = fa_stats["dsh"]

            diff_sq = (dsh_sim - target.dsh_mean) ** 2
            normalized = diff_sq / (target.dsh_var + 1e-10)
            cost_terms.append(normalized)
            weights.append(config.w_dsh)

        # Eigenspectrum
        if config.use_es:
            es_sim = fa_stats["eigenspectrum"][: config.n_es]

            # Pad if necessary
            if es_sim.shape[0] < config.n_es:
                es_sim = jnp.pad(es_sim, (0, config.n_es - es_sim.shape[0]))

            es_target = target.es_mean[: config.n_es]

            # Sum of squared differences
            diff_sq_sum = jnp.sum((es_sim - es_target) ** 2)
            normalized = diff_sq_sum / (target.es_var + 1e-10)
            cost_terms.append(normalized)
            weights.append(config.w_es)

    # Compute weighted cost
    cost_terms = jnp.array(cost_terms)
    weights = jnp.array(weights)

    total_weight = jnp.sum(weights)
    cost = jnp.sum(weights * cost_terms) / (total_weight + 1e-10)

    return float(cost), stats


def compute_target_statistics(
    spike_counts_sessions: list[chex.Array],
    config: CostConfig,
    rng_key: chex.PRNGKey = None,
) -> TargetStatistics:
    """
    Compute target statistics from multiple recording sessions.

    Args:
        spike_counts_sessions: List of spike count matrices, one per session
        config: Cost configuration
        rng_key: Random key

    Returns:
        TargetStatistics with means and variances
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    n_sessions = len(spike_counts_sessions)
    keys = jax.random.split(rng_key, n_sessions)

    # Collect statistics from all sessions
    fr_vals = []
    ff_vals = []
    rsc_z_vals = []
    pct_sh_vals = []
    dsh_vals = []
    es_vals = []

    for i, counts in enumerate(spike_counts_sessions):
        # Single/pairwise stats
        fr = compute_fr(counts, config.bin_size)
        ff = compute_ff(counts, config.min_fr, config.bin_size)
        rsc = compute_rsc(counts, config.min_fr, config.bin_size)
        rsc_z = fisher_z_transform(rsc)

        fr_vals.append(fr)
        ff_vals.append(ff)
        rsc_z_vals.append(rsc_z)

        # FA stats
        fa_stats = compute_shared_variance_stats(
            counts, dsh_threshold=config.dsh_threshold, rng_key=keys[i]
        )

        pct_sh_vals.append(fa_stats["pct_sh"])
        dsh_vals.append(fa_stats["dsh"])

        # Eigenspectrum (padded to config.n_es)
        es = fa_stats["eigenspectrum"][: config.n_es]
        if es.shape[0] < config.n_es:
            es = jnp.pad(es, (0, config.n_es - es.shape[0]))
        es_vals.append(es)

    # Compute means and variances
    fr_array = jnp.array(fr_vals)
    ff_array = jnp.array(ff_vals)
    rsc_z_array = jnp.array(rsc_z_vals)
    pct_sh_array = jnp.array(pct_sh_vals)
    dsh_array = jnp.array(dsh_vals)
    es_array = jnp.stack(es_vals)  # (n_sessions, n_es)

    # For eigenspectrum, compute variance of the summed squared differences
    # This is the variance across sessions of the scalar: sum((es - mean(es))^2)
    es_mean = jnp.mean(es_array, axis=0)
    es_diffs = es_array - es_mean[None, :]
    es_sq_diffs_per_session = jnp.sum(es_diffs**2, axis=1)  # (n_sessions,)
    es_var = jnp.var(es_sq_diffs_per_session)

    return TargetStatistics(
        fr_mean=float(jnp.mean(fr_array)),
        ff_mean=float(jnp.mean(ff_array)),
        rsc_z_mean=float(jnp.mean(rsc_z_array)),
        pct_sh_mean=float(jnp.mean(pct_sh_array)),
        dsh_mean=float(jnp.mean(dsh_array)),
        es_mean=es_mean,
        fr_var=float(jnp.var(fr_array)),
        ff_var=float(jnp.var(ff_array)),
        rsc_z_var=float(jnp.var(rsc_z_array)),
        pct_sh_var=float(jnp.var(pct_sh_array)),
        dsh_var=float(jnp.var(dsh_array)),
        es_var=float(es_var),
    )
