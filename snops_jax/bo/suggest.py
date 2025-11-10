"""
Candidate suggestion via acquisition function optimization.

Uses multi-start L-BFGS-B to maximize acquisition function.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Callable
import chex
from jaxopt import ScipyBoundedMinimize
from scipy.optimize import minimize
import numpy as np

from snops_jax.bo.gp import GPSurrogate
from snops_jax.bo.acquisition import compute_ei, compute_constrained_ei


def suggest_candidates(
    cost_gp: GPSurrogate,
    feasibility_gp: GPSurrogate,
    bounds: chex.Array,  # (n_dims, 2) [lower, upper] bounds
    f_best: float,
    n_candidates: int = 1,
    n_restarts: int = 50,
    use_feasibility: bool = True,
    rng_key: chex.PRNGKey = None,
) -> chex.Array:
    """
    Suggest new candidates by optimizing acquisition function.

    Uses multi-start L-BFGS-B to maximize (constrained) EI.

    Args:
        cost_gp: Cost GP surrogate
        feasibility_gp: Feasibility GP surrogate (optional)
        bounds: Parameter bounds (n_dims, 2)
        f_best: Best observed cost
        n_candidates: Number of candidates to return (batch size)
        n_restarts: Number of random restarts for optimization
        use_feasibility: Whether to use constrained EI
        rng_key: Random key

    Returns:
        Suggested candidates (n_candidates, n_dims)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    n_dims = bounds.shape[0]

    # Define acquisition function (negative for minimization)
    def neg_acquisition(x):
        x_2d = x.reshape(1, -1)
        if use_feasibility and feasibility_gp is not None:
            acq = compute_constrained_ei(x_2d, cost_gp, feasibility_gp, f_best)
        else:
            acq = compute_ei(x_2d, cost_gp, f_best)
        return -acq[0]  # Negative for minimization

    # Multi-start optimization
    best_candidates = []
    best_values = []

    keys = jax.random.split(rng_key, n_restarts)

    for i in range(n_restarts):
        # Random initialization within bounds
        x0 = jax.random.uniform(
            keys[i], shape=(n_dims,), minval=bounds[:, 0], maxval=bounds[:, 1]
        )

        # Optimize using scipy (easier for constrained optimization)
        result = minimize(
            lambda x: float(neg_acquisition(jnp.array(x))),
            x0=np.array(x0),
            method="L-BFGS-B",
            bounds=[(float(bounds[j, 0]), float(bounds[j, 1])) for j in range(n_dims)],
        )

        if result.success:
            best_candidates.append(result.x)
            best_values.append(result.fun)

    # Select top candidates
    if len(best_candidates) == 0:
        # Fallback: random sampling
        candidates = jax.random.uniform(
            rng_key, shape=(n_candidates, n_dims), minval=bounds[:, 0], maxval=bounds[:, 1]
        )
        return candidates

    # Sort by acquisition value
    indices = np.argsort(best_values)[:n_candidates]
    candidates = jnp.array([best_candidates[i] for i in indices])

    # If fewer candidates than requested, pad with random
    if candidates.shape[0] < n_candidates:
        n_extra = n_candidates - candidates.shape[0]
        extra = jax.random.uniform(
            rng_key, shape=(n_extra, n_dims), minval=bounds[:, 0], maxval=bounds[:, 1]
        )
        candidates = jnp.vstack([candidates, extra])

    return candidates


def latin_hypercube_sampling(
    bounds: chex.Array, n_samples: int, rng_key: chex.PRNGKey = None
) -> chex.Array:
    """
    Generate Latin Hypercube samples within bounds.

    Args:
        bounds: Parameter bounds (n_dims, 2)
        n_samples: Number of samples
        rng_key: Random key

    Returns:
        LHS samples (n_samples, n_dims)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    n_dims = bounds.shape[0]

    # Generate uniform samples in [0, 1]^d using LHS
    # For each dimension, divide [0,1] into n_samples bins and sample one point per bin
    samples = jnp.zeros((n_samples, n_dims))

    for d in range(n_dims):
        key_d = jax.random.fold_in(rng_key, d)

        # Permute bin indices
        perm = jax.random.permutation(key_d, n_samples)

        # Sample within each bin
        bin_width = 1.0 / n_samples
        uniform = jax.random.uniform(key_d, shape=(n_samples,))
        samples_d = (perm + uniform) * bin_width

        samples = samples.at[:, d].set(samples_d)

    # Scale to bounds
    samples_scaled = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])

    return samples_scaled


def initialize_bo(
    bounds: chex.Array,
    n_init: int = 50,
    rng_key: chex.PRNGKey = None,
) -> chex.Array:
    """
    Initialize BO with Latin Hypercube Sampling.

    Args:
        bounds: Parameter bounds (n_dims, 2)
        n_init: Number of initial samples (default 50, per paper)
        rng_key: Random key

    Returns:
        Initial parameter sets (n_init, n_dims)
    """
    return latin_hypercube_sampling(bounds, n_init, rng_key)


def trust_region_bounds(
    x_incumbent: chex.Array,
    bounds: chex.Array,
    trust_radius: float = 0.2,
) -> chex.Array:
    """
    Create trust region bounds around incumbent.

    Args:
        x_incumbent: Current incumbent (n_dims,)
        bounds: Global parameter bounds (n_dims, 2)
        trust_radius: Fraction of parameter range for trust region

    Returns:
        Trust region bounds (n_dims, 2)
    """
    n_dims = bounds.shape[0]
    param_ranges = bounds[:, 1] - bounds[:, 0]
    radius = trust_radius * param_ranges

    tr_lower = jnp.maximum(x_incumbent - radius, bounds[:, 0])
    tr_upper = jnp.minimum(x_incumbent + radius, bounds[:, 1])

    tr_bounds = jnp.stack([tr_lower, tr_upper], axis=1)

    return tr_bounds
