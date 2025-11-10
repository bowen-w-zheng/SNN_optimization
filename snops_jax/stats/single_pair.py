"""
Single-neuron and pairwise statistics.

Implements fr (firing rate), ff (Fano factor), and rsc (spike count correlation).
"""

import jax
import jax.numpy as jnp
import chex


def compute_fr(spike_counts: chex.Array, bin_size: float = 200.0) -> float:
    """
    Compute mean firing rate across all neurons and bins.

    Args:
        spike_counts: (n_neurons, n_bins) spike counts
        bin_size: Bin size (ms)

    Returns:
        Mean firing rate (sp/s)
    """
    bin_size_s = bin_size / 1000.0  # Convert to seconds
    mean_count = jnp.mean(spike_counts)
    fr = mean_count / bin_size_s
    return fr


def compute_ff(spike_counts: chex.Array, min_fr: float = 0.5, bin_size: float = 200.0) -> float:
    """
    Compute Fano factor (variance/mean) averaged across neurons.

    Excludes neurons with fr < min_fr.

    Args:
        spike_counts: (n_neurons, n_bins) spike counts
        min_fr: Minimum firing rate threshold (sp/s)
        bin_size: Bin size (ms)

    Returns:
        Mean Fano factor
    """
    # Compute firing rate for each neuron
    n_bins = spike_counts.shape[1]
    total_time_s = (n_bins * bin_size) / 1000.0
    fr_per_neuron = jnp.sum(spike_counts, axis=1) / total_time_s

    # Filter neurons
    valid = fr_per_neuron >= min_fr
    counts_valid = spike_counts[valid]

    # Compute variance and mean across time for each neuron
    var_per_neuron = jnp.var(counts_valid, axis=1)
    mean_per_neuron = jnp.mean(counts_valid, axis=1)

    # Fano factor per neuron
    # Avoid division by zero
    ff_per_neuron = jnp.where(
        mean_per_neuron > 1e-6, var_per_neuron / mean_per_neuron, 0.0
    )

    # Average across neurons
    ff = jnp.mean(ff_per_neuron)

    return ff


def compute_rsc(
    spike_counts: chex.Array, min_fr: float = 0.5, bin_size: float = 200.0
) -> float:
    """
    Compute mean spike count correlation (Pearson correlation across time).

    Averages correlation coefficients across all neuron pairs.
    Excludes neurons with fr < min_fr.

    Args:
        spike_counts: (n_neurons, n_bins) spike counts
        min_fr: Minimum firing rate threshold (sp/s)
        bin_size: Bin size (ms)

    Returns:
        Mean Pearson correlation coefficient
    """
    # Filter low-firing neurons
    n_bins = spike_counts.shape[1]
    total_time_s = (n_bins * bin_size) / 1000.0
    fr_per_neuron = jnp.sum(spike_counts, axis=1) / total_time_s

    valid = fr_per_neuron >= min_fr
    counts_valid = spike_counts[valid]

    n_valid = counts_valid.shape[0]

    if n_valid < 2:
        return 0.0

    # Fast correlation computation using efficient JAX operations
    # Standardize (z-score) each neuron's activity
    mean = jnp.mean(counts_valid, axis=1, keepdims=True)
    std = jnp.std(counts_valid, axis=1, keepdims=True) + 1e-8
    z_scored = (counts_valid - mean) / std

    # Correlation matrix = (Z @ Z.T) / n_bins
    # Use efficient BLAS matrix multiplication
    n_bins = counts_valid.shape[1]
    corr_matrix = jnp.dot(z_scored, z_scored.T) / n_bins

    # Extract upper triangle efficiently (excluding diagonal)
    # Sum upper triangle and divide by number of pairs
    n_pairs = (n_valid * (n_valid - 1)) // 2
    upper_sum = (jnp.sum(corr_matrix) - jnp.trace(corr_matrix)) / 2.0
    mean_corr = upper_sum / (n_pairs + 1e-10)

    return mean_corr


def fisher_z_transform(r: float) -> float:
    """
    Apply Fisher z-transformation to correlation coefficient.

    z = 0.5 * ln((1 + r) / (1 - r))

    Args:
        r: Correlation coefficient

    Returns:
        Fisher z-transformed value
    """
    # Clip to avoid numerical issues
    r = jnp.clip(r, -0.9999, 0.9999)
    z = 0.5 * jnp.log((1 + r) / (1 - r))
    return z


def compute_statistics_summary(
    spike_counts: chex.Array,
    bin_size: float = 200.0,
    min_fr: float = 0.5,
) -> dict:
    """
    Compute all single-neuron and pairwise statistics.

    Optimized with fast JAX operations (no JIT due to dynamic shapes).

    Args:
        spike_counts: (n_neurons, n_bins)
        bin_size: Bin size (ms)
        min_fr: Minimum firing rate (sp/s)

    Returns:
        Dictionary with fr, ff, rsc, rsc_z
    """
    fr = compute_fr(spike_counts, bin_size)
    ff = compute_ff(spike_counts, min_fr, bin_size)
    rsc = compute_rsc(spike_counts, min_fr, bin_size)
    rsc_z = fisher_z_transform(rsc)

    return {"fr": float(fr), "ff": float(ff), "rsc": float(rsc), "rsc_z": float(rsc_z)}
