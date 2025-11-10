"""Statistics computation for neural activity."""

from snops_jax.stats.binning import bin_spikes, subsample_neurons
from snops_jax.stats.single_pair import compute_fr, compute_ff, compute_rsc, fisher_z_transform
from snops_jax.stats.fa_jax import fit_fa_em, compute_shared_variance_stats
from snops_jax.stats.cost import compute_cost, CostConfig

__all__ = [
    "bin_spikes",
    "subsample_neurons",
    "compute_fr",
    "compute_ff",
    "compute_rsc",
    "fisher_z_transform",
    "fit_fa_em",
    "compute_shared_variance_stats",
    "compute_cost",
    "CostConfig",
]
