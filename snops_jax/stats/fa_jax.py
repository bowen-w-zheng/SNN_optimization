"""
Factor Analysis (FA) using Expectation-Maximization (EM) in JAX.

Computes shared variance statistics: %sh, dsh, eigenspectrum.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple
import chex
from sklearn.model_selection import KFold


class FAResult(NamedTuple):
    """Result from Factor Analysis."""

    L: chex.Array  # Loading matrix (n_neurons, n_factors)
    Psi: chex.Array  # Diagonal noise covariance (n_neurons,)
    n_factors: int  # Number of latent factors
    log_likelihood: float  # Final log-likelihood


def fit_fa_em(
    data: chex.Array,
    n_factors: int,
    max_iter: int = 50,  # Reduced from 100 for speed
    tol: float = 1e-3,  # Relaxed from 1e-4 for faster convergence
    rng_key: chex.PRNGKey = None,
) -> FAResult:
    """
    Fit Factor Analysis model using EM algorithm.

    Model: X = L * Z + ε
    where Z ~ N(0, I), ε ~ N(0, Ψ), Ψ is diagonal

    Args:
        data: Data matrix (n_samples, n_features)
        n_factors: Number of latent factors
        max_iter: Maximum EM iterations
        tol: Convergence tolerance on log-likelihood
        rng_key: Random key for initialization

    Returns:
        FAResult with loading matrix L and noise covariance Psi
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    n_samples, n_features = data.shape

    # Center data
    data_mean = jnp.mean(data, axis=0)
    data_centered = data - data_mean

    # Initialize parameters
    sample_cov = jnp.cov(data_centered.T)

    # Initialize L with random values scaled by covariance
    L = jax.random.normal(rng_key, (n_features, n_factors)) * 0.1

    # Initialize Psi as diagonal of sample covariance
    Psi = jnp.diag(sample_cov).copy()
    Psi = jnp.maximum(Psi, 0.01)  # Ensure positive

    prev_ll = -jnp.inf

    for iteration in range(max_iter):
        # E-step: Compute posterior over latent variables
        # Posterior: p(z|x) ~ N(m, V)
        # V = (I + L^T Psi^{-1} L)^{-1}
        # m = V L^T Psi^{-1} x

        Psi_inv = 1.0 / Psi  # Element-wise inverse (diagonal)
        Psi_inv_L = Psi_inv[:, None] * L  # Psi^{-1} L

        V_inv = jnp.eye(n_factors) + L.T @ Psi_inv_L  # (n_factors, n_factors)
        V = jnp.linalg.inv(V_inv)

        # Compute sufficient statistics
        # E[z] for each sample
        Ez = data_centered @ Psi_inv_L @ V  # (n_samples, n_factors)

        # E[zz^T] = V + E[z]E[z]^T
        Ezz = V[None, :, :] + Ez[:, :, None] * Ez[:, None, :]  # (n_samples, n_factors, n_factors)
        Ezz_sum = jnp.sum(Ezz, axis=0)  # (n_factors, n_factors)

        # M-step: Update parameters
        # L_new = (Σ x E[z]^T) (Σ E[zz^T])^{-1}
        xEz = data_centered.T @ Ez  # (n_features, n_factors)
        L_new = xEz @ jnp.linalg.inv(Ezz_sum)

        # Psi_new = diag(S - L_new (Σ x E[z]^T)^T / n_samples)
        S_diag = jnp.sum(data_centered**2, axis=0) / n_samples
        L_xEz_diag = jnp.sum(L_new * xEz, axis=1) / n_samples
        Psi_new = S_diag - L_xEz_diag
        Psi_new = jnp.maximum(Psi_new, 0.01)  # Ensure positive

        # Update parameters
        L = L_new
        Psi = Psi_new

        # Compute log-likelihood
        ll = _compute_log_likelihood(data_centered, L, Psi)

        # Check convergence
        if jnp.abs(ll - prev_ll) < tol:
            break

        prev_ll = ll

    return FAResult(L=L, Psi=Psi, n_factors=n_factors, log_likelihood=ll)


def _compute_log_likelihood(data: chex.Array, L: chex.Array, Psi: chex.Array) -> float:
    """
    Compute log-likelihood of FA model.

    p(x) ~ N(0, LL^T + Psi)

    Args:
        data: Centered data (n_samples, n_features)
        L: Loading matrix (n_features, n_factors)
        Psi: Diagonal noise (n_features,)

    Returns:
        Log-likelihood
    """
    n_samples, n_features = data.shape

    # Covariance matrix
    C = L @ L.T + jnp.diag(Psi)

    # Log-likelihood (avoiding explicit inverse)
    # ll = -0.5 * n_samples * (log|C| + tr(C^{-1} S))
    # where S = (1/n) X^T X

    sign, logdet = jnp.linalg.slogdet(C)
    C_inv = jnp.linalg.inv(C)

    S = (data.T @ data) / n_samples
    trace_term = jnp.trace(C_inv @ S)

    ll = -0.5 * n_samples * (logdet + trace_term + n_features * jnp.log(2 * jnp.pi))

    return ll


def select_n_factors_cv(
    data: chex.Array,
    max_factors: int = 20,
    n_folds: int = 5,
    rng_key: chex.PRNGKey = None,
) -> int:
    """
    Select number of factors using cross-validation.

    Args:
        data: Data matrix (n_samples, n_features)
        max_factors: Maximum number of factors to try
        n_folds: Number of CV folds
        rng_key: Random key

    Returns:
        Optimal number of factors
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    n_samples, n_features = data.shape
    max_factors = min(max_factors, n_features - 1)

    # Use sklearn KFold for simplicity (not JIT-compiled)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    cv_scores = []

    for n_factors in range(1, max_factors + 1):
        fold_lls = []

        for train_idx, test_idx in kf.split(data):
            train_data = data[train_idx]
            test_data = data[test_idx]

            # Fit on train
            key_fit = jax.random.fold_in(rng_key, n_factors)
            result = fit_fa_em(train_data, n_factors, rng_key=key_fit)

            # Evaluate on test
            test_mean = jnp.mean(test_data, axis=0)
            test_centered = test_data - test_mean
            ll_test = _compute_log_likelihood(test_centered, result.L, result.Psi)

            fold_lls.append(ll_test)

        mean_ll = jnp.mean(jnp.array(fold_lls))
        cv_scores.append(mean_ll)

    # Select number of factors with best CV score
    best_n = jnp.argmax(jnp.array(cv_scores)) + 1

    return int(best_n)


def compute_shared_variance_stats(
    spike_counts: chex.Array,
    n_factors: int = None,
    dsh_threshold: float = 0.95,
    use_cv: bool = False,
    rng_key: chex.PRNGKey = None,
) -> dict:
    """
    Compute shared variance statistics from Factor Analysis.

    Args:
        spike_counts: (n_neurons, n_bins)
        n_factors: Number of factors (if None, use heuristic or CV)
        dsh_threshold: Threshold for dimensionality of shared variance (default 0.95)
        use_cv: If True and n_factors is None, use CV to select (slow but accurate)
        rng_key: Random key

    Returns:
        Dictionary with:
        - pct_sh: Percent shared variance per neuron, then averaged
        - dsh: Dimensionality of shared variance
        - eigenspectrum: Eigenvalues of LL^T
        - L: Loading matrix
        - Psi: Noise covariance
        - n_factors: Number of factors used
    """
    # Move spike_counts to CPU to avoid GPU OOM during FA computation
    # FA is not performance-critical compared to simulation
    try:
        cpu_device = jax.devices('cpu')[0]
        spike_counts = jax.device_put(spike_counts, cpu_device)
    except:
        # Fallback if CPU device not available
        pass

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Transpose to (n_bins, n_neurons) for FA
    data = spike_counts.T
    n_bins, n_neurons = data.shape

    # Select number of factors if not specified
    if n_factors is None:
        if use_cv:
            # Use expensive but accurate CV (for production runs)
            max_factors = min(20, n_bins - 1, n_neurons - 1)
            n_factors = select_n_factors_cv(data, max_factors=max_factors, rng_key=rng_key)
        else:
            # Use fast heuristic (for demos and quick tests)
            # Heuristic: min(n_bins // 2, n_neurons // 10, 15)
            n_factors = min(n_bins // 2, n_neurons // 10, 15)
            n_factors = max(n_factors, 1)  # At least 1 factor

    # Fit FA
    result = fit_fa_em(data, n_factors, rng_key=rng_key)

    # Compute shared variance covariance
    shared_cov = result.L @ result.L.T  # (n_neurons, n_neurons)

    # Percent shared variance per neuron
    # %sh_j = ||L_j||^2 / (||L_j||^2 + Psi_j)
    shared_var_per_neuron = jnp.sum(result.L**2, axis=1)  # ||L_j||^2
    total_var_per_neuron = shared_var_per_neuron + result.Psi

    pct_sh_per_neuron = shared_var_per_neuron / total_var_per_neuron
    pct_sh = jnp.mean(pct_sh_per_neuron)  # Average across neurons

    # Dimensionality of shared variance (dsh)
    # Number of eigenvalues of LL^T that explain dsh_threshold of variance
    eigenvalues = jnp.linalg.eigvalsh(shared_cov)
    eigenvalues = jnp.sort(eigenvalues)[::-1]  # Descending order
    eigenvalues = jnp.maximum(eigenvalues, 0.0)  # Remove numerical negatives

    total_var = jnp.sum(eigenvalues)
    cumsum = jnp.cumsum(eigenvalues)
    dsh = jnp.searchsorted(cumsum, dsh_threshold * total_var) + 1

    return {
        "pct_sh": float(pct_sh),
        "dsh": int(dsh),
        "eigenspectrum": eigenvalues,
        "L": result.L,
        "Psi": result.Psi,
        "n_factors": n_factors,
    }
