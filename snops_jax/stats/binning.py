"""
Spike binning and neuron subsampling utilities.
"""

import jax
import jax.numpy as jnp
import chex


def bin_spikes(
    spike_times: chex.Array,
    spike_neurons: chex.Array,
    n_neurons: int,
    bin_size: float,
    duration: float,
    burn_in: float = 0.0,
) -> chex.Array:
    """
    Bin spike times into count matrix.

    Args:
        spike_times: Times of spikes (n_spikes,) [ms]
        spike_neurons: Neuron indices for each spike (n_spikes,)
        n_neurons: Total number of neurons
        bin_size: Bin size (ms)
        duration: Total duration (ms)
        burn_in: Burn-in period to discard (ms)

    Returns:
        Spike count matrix (n_neurons, n_bins)
    """
    # Filter spikes after burn-in
    valid = spike_times >= burn_in
    spike_times_filt = spike_times[valid] - burn_in
    spike_neurons_filt = spike_neurons[valid]

    # Compute bin indices
    n_bins = int((duration - burn_in) / bin_size)
    bin_indices = jnp.floor(spike_times_filt / bin_size).astype(jnp.int32)

    # Filter spikes that fall within valid bins
    valid_bins = (bin_indices >= 0) & (bin_indices < n_bins)
    bin_indices = bin_indices[valid_bins]
    spike_neurons_filt = spike_neurons_filt[valid_bins]

    # Create count matrix using segment_sum equivalent
    # Create flattened index: neuron * n_bins + bin
    flat_idx = spike_neurons_filt * n_bins + bin_indices
    counts_flat = jnp.zeros(n_neurons * n_bins, dtype=jnp.float32)
    counts_flat = counts_flat.at[flat_idx].add(1.0)

    # Reshape to (n_neurons, n_bins)
    counts = counts_flat.reshape(n_neurons, n_bins)

    return counts


def subsample_neurons(
    spike_counts: chex.Array,
    n_sample: int,
    rng_key: chex.PRNGKey,
    min_fr: float = 0.5,
    bin_size: float = 200.0,
) -> tuple[chex.Array, chex.Array]:
    """
    Subsample neurons for statistics computation.

    Excludes low-firing neurons and randomly samples n_sample neurons.

    Args:
        spike_counts: Spike count matrix (n_neurons, n_bins)
        n_sample: Number of neurons to sample
        rng_key: JAX random key
        min_fr: Minimum firing rate threshold (sp/s)
        bin_size: Bin size (ms)

    Returns:
        (sampled_counts, sampled_indices)
        sampled_counts: (n_sample, n_bins)
        sampled_indices: (n_sample,) indices of sampled neurons
    """
    n_neurons, n_bins = spike_counts.shape

    # Compute mean firing rate for each neuron
    total_time_s = (n_bins * bin_size) / 1000.0
    fr = jnp.sum(spike_counts, axis=1) / total_time_s

    # Find neurons above threshold
    valid = fr >= min_fr
    valid_indices = jnp.where(valid, size=n_neurons, fill_value=-1)[0]
    valid_indices = valid_indices[valid_indices >= 0]  # Remove padding

    n_valid = jnp.sum(valid)

    # Sample from valid neurons
    n_to_sample = jnp.minimum(n_sample, n_valid)

    sampled_indices = jax.random.choice(
        rng_key, valid_indices, shape=(n_to_sample,), replace=False
    )

    sampled_counts = spike_counts[sampled_indices]

    return sampled_counts, sampled_indices


def apply_subsample_protocol(
    spike_counts: chex.Array,
    n_sample: int = 50,
    n_repeats: int = 10,
    rng_key: chex.PRNGKey = None,
    min_fr: float = 0.5,
    bin_size: float = 200.0,
) -> chex.Array:
    """
    Apply the paper's subsampling protocol: average over multiple random samples.

    Args:
        spike_counts: (n_neurons, n_bins)
        n_sample: Number of neurons per sample (default 50)
        n_repeats: Number of random samples (default 10)
        rng_key: Random key
        min_fr: Minimum firing rate (sp/s)
        bin_size: Bin size (ms)

    Returns:
        List of sampled spike count matrices, one per repeat
        Shape: (n_repeats, n_sample, n_bins)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    keys = jax.random.split(rng_key, n_repeats)

    def sample_once(key):
        counts, _ = subsample_neurons(spike_counts, n_sample, key, min_fr, bin_size)
        return counts

    # Stack samples
    samples = jax.vmap(sample_once)(keys)

    return samples
