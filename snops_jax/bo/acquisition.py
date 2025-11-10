"""
Acquisition functions for Bayesian Optimization.

Implements Expected Improvement (EI) and Constrained EI.
"""

import jax
import jax.numpy as jnp
import chex
from scipy.stats import norm

from snops_jax.bo.gp import GPSurrogate, predict_gp, predict_feasibility_prob


def compute_ei(
    X_cand: chex.Array,
    cost_gp: GPSurrogate,
    f_best: float,
    xi: float = 0.01,
) -> chex.Array:
    """
    Compute Expected Improvement acquisition function.

    EI(x) = E[max(f_best - f(x), 0)]
          = (f_best - μ) Φ(Z) + σ φ(Z)
    where Z = (f_best - μ) / σ

    Args:
        X_cand: Candidate points (n_cand, n_dims)
        cost_gp: Cost GP surrogate
        f_best: Best observed cost (minimum)
        xi: Exploration parameter (default 0.01)

    Returns:
        EI values (n_cand,)
    """
    # Predict at candidates
    mean, var = predict_gp(cost_gp, X_cand)
    std = jnp.sqrt(var + 1e-10)

    # Compute improvement
    improvement = f_best - mean - xi

    # Z-score
    Z = improvement / std

    # EI formula
    ei = improvement * jax.scipy.stats.norm.cdf(Z) + std * jax.scipy.stats.norm.pdf(Z)

    # Set EI to 0 where std is very small
    ei = jnp.where(std > 1e-10, ei, 0.0)

    return ei


def compute_constrained_ei(
    X_cand: chex.Array,
    cost_gp: GPSurrogate,
    feasibility_gp: GPSurrogate,
    f_best: float,
    xi: float = 0.01,
    feasibility_threshold: float = 0.5,
) -> chex.Array:
    """
    Compute Constrained Expected Improvement (paper equation 12).

    CEI(x) = P(feasible | x) * EI(x)
           = Φ((μ_g - 0.5) / σ_g) * EI(x)

    where g(x) is the feasibility GP with threshold 0.5.

    Args:
        X_cand: Candidate points (n_cand, n_dims)
        cost_gp: Cost GP surrogate
        feasibility_gp: Feasibility GP surrogate
        f_best: Best observed cost
        xi: Exploration parameter
        feasibility_threshold: Threshold for feasibility (default 0.5)

    Returns:
        Constrained EI values (n_cand,)
    """
    # Compute EI
    ei = compute_ei(X_cand, cost_gp, f_best, xi)

    # Compute feasibility probability
    mean_g, var_g = predict_gp(feasibility_gp, X_cand)
    std_g = jnp.sqrt(var_g + 1e-10)

    # Paper uses: Φ((μ_g - threshold) / σ_g)
    prob_feasible = jax.scipy.stats.norm.cdf((mean_g - feasibility_threshold) / std_g)

    # Constrained EI
    cei = prob_feasible * ei

    return cei


def compute_ucb(
    X_cand: chex.Array,
    cost_gp: GPSurrogate,
    beta: float = 2.0,
) -> chex.Array:
    """
    Compute Upper Confidence Bound (alternative acquisition function).

    UCB(x) = μ(x) - β * σ(x)  (negative for minimization)

    Args:
        X_cand: Candidate points (n_cand, n_dims)
        cost_gp: Cost GP surrogate
        beta: Exploration parameter (higher = more exploration)

    Returns:
        UCB values (n_cand,) - higher is better
    """
    mean, var = predict_gp(cost_gp, X_cand)
    std = jnp.sqrt(var + 1e-10)

    # For minimization, we want lower mean and higher uncertainty
    # So we maximize -(mean - beta*std) = -mean + beta*std
    ucb = -mean + beta * std

    return ucb


def local_penalization(
    X_cand: chex.Array,
    X_pending: chex.Array,
    cost_gp: GPSurrogate,
    lipschitz: float = 10.0,
) -> chex.Array:
    """
    Apply local penalization to avoid selecting points near pending evaluations.

    Penalizes candidates within Lipschitz * std distance of pending points.

    Args:
        X_cand: Candidate points (n_cand, n_dims)
        X_pending: Pending evaluation points (n_pending, n_dims)
        cost_gp: Cost GP (for std estimates)
        lipschitz: Lipschitz constant estimate

    Returns:
        Penalty multipliers (n_cand,) - multiply with acquisition function
    """
    if X_pending.shape[0] == 0:
        return jnp.ones(X_cand.shape[0])

    # Predict std at candidates
    _, var = predict_gp(cost_gp, X_cand)
    std = jnp.sqrt(var + 1e-10)

    # Compute distances to pending points
    # dist[i, j] = ||X_cand[i] - X_pending[j]||
    diff = X_cand[:, None, :] - X_pending[None, :, :]  # (n_cand, n_pending, n_dims)
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # (n_cand, n_pending)

    # Minimum distance to any pending point
    min_dist = jnp.min(dist, axis=1)  # (n_cand,)

    # Penalization radius
    radius = lipschitz * std

    # Penalty: 0 if within radius, 1 otherwise (smooth version)
    penalty = jax.nn.sigmoid((min_dist - radius) / (0.1 * radius))

    return penalty


def batch_ei_fantasy(
    X_cand: chex.Array,
    cost_gp: GPSurrogate,
    f_best: float,
    batch_size: int,
    n_fantasies: int = 10,
    rng_key: chex.PRNGKey = None,
) -> chex.Array:
    """
    Compute batch EI using fantasy samples (q-EI approximation).

    Args:
        X_cand: Candidate points (n_cand, n_dims)
        cost_gp: Cost GP surrogate
        f_best: Best observed cost
        batch_size: Number of points to select
        n_fantasies: Number of fantasy samples
        rng_key: Random key

    Returns:
        Batch indices (batch_size,)
    """
    # For simplicity, use greedy selection with local penalization
    # (Full q-EI requires Monte Carlo integration)
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    selected = []
    X_pending = jnp.empty((0, X_cand.shape[1]))

    for _ in range(batch_size):
        # Compute EI with local penalization
        ei = compute_ei(X_cand, cost_gp, f_best)
        penalty = local_penalization(X_cand, X_pending, cost_gp)
        penalized_ei = ei * penalty

        # Select best
        best_idx = jnp.argmax(penalized_ei)
        selected.append(best_idx)

        # Add to pending
        X_pending = jnp.vstack([X_pending, X_cand[best_idx:best_idx+1]])

    return jnp.array(selected)
