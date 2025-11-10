"""
Gaussian Process surrogates for cost and feasibility.

Uses tinygp with ARD Matérn-5/2 kernel as specified in the paper.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable
import chex
from tinygp import kernels, GaussianProcess
from jaxopt import ScipyMinimize
import optax


class GPSurrogate(NamedTuple):
    """Trained GP surrogate."""

    kernel: kernels.Kernel
    X_train: chex.Array  # (n_train, n_dims)
    y_train: chex.Array  # (n_train,)
    y_mean: float  # Mean of training targets (for standardization)
    y_std: float  # Std of training targets
    noise: float  # Observation noise


def build_matern52_ard_kernel(n_dims: int, lengthscales: chex.Array, variance: float) -> kernels.Kernel:
    """
    Build ARD Matérn-5/2 kernel.

    Args:
        n_dims: Number of dimensions
        lengthscales: Length scales per dimension (n_dims,)
        variance: Kernel variance (amplitude^2)

    Returns:
        Kernel object
    """
    # Matérn-5/2 with ARD
    kernel = variance * kernels.Matern52(distance=kernels.distance.L2Distance(lengthscales))
    return kernel


def fit_gp(
    X_train: chex.Array,
    y_train: chex.Array,
    n_restarts: int = 10,
    rng_key: chex.PRNGKey = None,
) -> GPSurrogate:
    """
    Fit GP by optimizing kernel hyperparameters via marginal likelihood.

    Uses ARD Matérn-5/2 kernel and constant mean.

    Args:
        X_train: Training inputs (n_train, n_dims)
        y_train: Training targets (n_train,)
        n_restarts: Number of random restarts for optimization
        rng_key: JAX random key

    Returns:
        Fitted GPSurrogate
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    n_train, n_dims = X_train.shape

    # Standardize targets
    y_mean = jnp.mean(y_train)
    y_std = jnp.std(y_train) + 1e-6
    y_standardized = (y_train - y_mean) / y_std

    # Define negative log marginal likelihood
    def neg_log_marginal_likelihood(params):
        """Compute negative log marginal likelihood."""
        lengthscales = jnp.exp(params["log_lengthscales"])
        variance = jnp.exp(params["log_variance"])
        noise = jnp.exp(params["log_noise"])

        kernel = build_matern52_ard_kernel(n_dims, lengthscales, variance)
        gp = GaussianProcess(kernel, X_train, diag=noise)

        return -gp.log_probability(y_standardized)

    # Multi-restart optimization
    best_params = None
    best_nll = jnp.inf

    for i in range(n_restarts):
        # Initialize parameters
        key_i = jax.random.fold_in(rng_key, i)
        keys = jax.random.split(key_i, 3)

        init_params = {
            "log_lengthscales": jax.random.normal(keys[0], (n_dims,)) * 0.5,
            "log_variance": jax.random.normal(keys[1], ()) * 0.5,
            "log_noise": jnp.log(0.01),  # Small noise
        }

        # Optimize using L-BFGS-B
        solver = ScipyMinimize(fun=neg_log_marginal_likelihood, method="L-BFGS-B")
        result = solver.run(init_params)

        if result.state.fun_val < best_nll:
            best_nll = result.state.fun_val
            best_params = result.params

    # Extract best hyperparameters
    lengthscales = jnp.exp(best_params["log_lengthscales"])
    variance = jnp.exp(best_params["log_variance"])
    noise = jnp.exp(best_params["log_noise"])

    kernel = build_matern52_ard_kernel(n_dims, lengthscales, variance)

    return GPSurrogate(
        kernel=kernel,
        X_train=X_train,
        y_train=y_standardized,
        y_mean=y_mean,
        y_std=y_std,
        noise=noise,
    )


def predict_gp(surrogate: GPSurrogate, X_test: chex.Array) -> tuple[chex.Array, chex.Array]:
    """
    Predict mean and variance at test points.

    Args:
        surrogate: Fitted GP surrogate
        X_test: Test inputs (n_test, n_dims)

    Returns:
        (mean, variance) predictions (n_test,)
        Mean and variance are in original (unstandardized) scale.
    """
    # Build GP
    gp = GaussianProcess(surrogate.kernel, surrogate.X_train, diag=surrogate.noise)

    # Condition on training data
    cond_gp = gp.condition(surrogate.y_train, X_test)

    # Predict (in standardized space)
    mean_std = cond_gp.mean
    var_std = cond_gp.variance

    # Transform back to original scale
    mean = mean_std * surrogate.y_std + surrogate.y_mean
    variance = var_std * (surrogate.y_std**2)

    return mean, variance


def fit_feasibility_gp(
    X_train: chex.Array,
    feasible: chex.Array,  # Binary labels {0, 1}
    n_restarts: int = 5,
    rng_key: chex.PRNGKey = None,
) -> GPSurrogate:
    """
    Fit GP for feasibility classification.

    Uses GP regression on binary labels (simpler than full GP classification).

    Args:
        X_train: Training inputs (n_train, n_dims)
        feasible: Binary feasibility labels (n_train,) [0 or 1]
        n_restarts: Number of random restarts
        rng_key: Random key

    Returns:
        Fitted GP surrogate
    """
    # Use GP regression on {0, 1} labels
    # Predictions will be probabilities (after sigmoid)
    return fit_gp(X_train, feasible.astype(jnp.float32), n_restarts, rng_key)


def predict_feasibility_prob(surrogate: GPSurrogate, X_test: chex.Array) -> chex.Array:
    """
    Predict feasibility probability using GP.

    Uses probit approximation: P(feasible) ≈ Φ(μ / √(1 + σ²))

    Args:
        surrogate: Fitted feasibility GP
        X_test: Test inputs (n_test, n_dims)

    Returns:
        Feasibility probabilities (n_test,)
    """
    mean, var = predict_gp(surrogate, X_test)

    # Probit approximation for binary classification
    # (More accurate would be Laplace/EP, but this is simpler and often adequate)
    prob = jax.scipy.stats.norm.cdf(mean / jnp.sqrt(1.0 + var))

    return prob
