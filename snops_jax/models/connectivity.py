"""
Network connectivity generators for CBN (Classical Balanced Network)
and SBN (Spatial Balanced Network).
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
import chex


class NetworkParams(NamedTuple):
    """Parameters for network connectivity (free parameters for optimization)."""

    # Synaptic strengths [mV] - these are the free parameters
    J_ee: float = 0.0  # E -> E strength, range [0, 150]
    J_ei: float = -50.0  # I -> E strength (inhibitory), range [-150, 0]
    J_ie: float = 50.0  # E -> I strength, range [0, 150]
    J_ii: float = -50.0  # I -> I strength (inhibitory), range [-150, 0]
    J_eF: float = 50.0  # Feedforward -> E strength, range [0, 150]
    J_iF: float = 50.0  # Feedforward -> I strength, range [0, 150]

    # For SBN: spatial connection widths [mm]
    sigma_e: float = 0.25  # E connection width, range [0, 0.25]
    sigma_i: float = 0.25  # I connection width, range [0, 0.25]
    sigma_F: float = 0.25  # Feedforward width, range [0, 0.25]

    def to_dict(self):
        """Convert to dictionary."""
        return self._asdict()


class Connectivity(NamedTuple):
    """Network connectivity stored as edge lists (COO format for GPU efficiency)."""

    # E -> E connections
    ee_src: chex.Array  # Source E neuron indices
    ee_dst: chex.Array  # Destination E neuron indices
    ee_weights: chex.Array  # Connection weights [mV]

    # I -> E connections (inhibitory)
    ie_src: chex.Array  # Source I neuron indices
    ie_dst: chex.Array  # Destination E neuron indices
    ie_weights: chex.Array  # Connection weights [mV] (negative)

    # E -> I connections
    ei_src: chex.Array
    ei_dst: chex.Array
    ei_weights: chex.Array

    # I -> I connections (inhibitory)
    ii_src: chex.Array
    ii_dst: chex.Array
    ii_weights: chex.Array

    # Feedforward connections
    fe_src: chex.Array  # Feedforward source indices
    fe_dst: chex.Array  # E neuron destination indices
    fe_weights: chex.Array

    fi_src: chex.Array
    fi_dst: chex.Array
    fi_weights: chex.Array


def build_cbn(
    n_e: int,
    n_i: int,
    n_ff: int,
    params: NetworkParams,
    p_ee: float = 0.2,
    p_ei: float = 0.5,
    p_ie: float = 0.5,
    p_ii: float = 0.5,
    p_fe: float = 0.5,
    p_fi: float = 0.5,
    rng_key: chex.PRNGKey = None,
) -> Connectivity:
    """
    Build Classical Balanced Network (CBN) with random connectivity.

    Args:
        n_e: Number of excitatory neurons
        n_i: Number of inhibitory neurons
        n_ff: Number of feedforward inputs
        params: Network parameters with synaptic strengths
        p_ee, p_ei, p_ie, p_ii: Connection probabilities
        p_fe, p_fi: Feedforward connection probabilities
        rng_key: JAX random key

    Returns:
        Connectivity object with edge lists
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    keys = jax.random.split(rng_key, 6)

    # E -> E connections
    ee_src, ee_dst, ee_weights = _random_connections(
        n_e, n_e, p_ee, params.J_ee, keys[0], allow_self=False
    )

    # I -> E connections
    ie_src, ie_dst, ie_weights = _random_connections(
        n_i, n_e, p_ie, params.J_ei, keys[1], allow_self=False
    )

    # E -> I connections
    ei_src, ei_dst, ei_weights = _random_connections(
        n_e, n_i, p_ei, params.J_ie, keys[2], allow_self=False
    )

    # I -> I connections
    ii_src, ii_dst, ii_weights = _random_connections(
        n_i, n_i, p_ii, params.J_ii, keys[3], allow_self=False
    )

    # Feedforward -> E
    fe_src, fe_dst, fe_weights = _random_connections(
        n_ff, n_e, p_fe, params.J_eF, keys[4], allow_self=True
    )

    # Feedforward -> I
    fi_src, fi_dst, fi_weights = _random_connections(
        n_ff, n_i, p_fi, params.J_iF, keys[5], allow_self=True
    )

    return Connectivity(
        ee_src=ee_src,
        ee_dst=ee_dst,
        ee_weights=ee_weights,
        ie_src=ie_src,
        ie_dst=ie_dst,
        ie_weights=ie_weights,
        ei_src=ei_src,
        ei_dst=ei_dst,
        ei_weights=ei_weights,
        ii_src=ii_src,
        ii_dst=ii_dst,
        ii_weights=ii_weights,
        fe_src=fe_src,
        fe_dst=fe_dst,
        fe_weights=fe_weights,
        fi_src=fi_src,
        fi_dst=fi_dst,
        fi_weights=fi_weights,
    )


def build_sbn(
    n_e: int,
    n_i: int,
    n_ff: int,
    params: NetworkParams,
    grid_size: float = 1.0,  # mm
    p_base: float = 0.2,  # Base connection probability
    rng_key: chex.PRNGKey = None,
) -> Connectivity:
    """
    Build Spatial Balanced Network (SBN) with distance-dependent connectivity.

    Neurons are placed on a square grid with periodic boundary conditions.
    Connection probability follows a wrapped Gaussian as a function of distance.

    Args:
        n_e: Number of excitatory neurons
        n_i: Number of inhibitory neurons
        n_ff: Number of feedforward inputs
        params: Network parameters with spatial widths (sigma_e, sigma_i, sigma_F)
        grid_size: Size of square grid (mm)
        p_base: Base connection probability
        rng_key: JAX random key

    Returns:
        Connectivity object with edge lists
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    keys = jax.random.split(rng_key, 8)

    # Generate neuron positions on grid
    pos_e = _generate_grid_positions(n_e, grid_size, keys[0])
    pos_i = _generate_grid_positions(n_i, grid_size, keys[1])
    pos_ff = _generate_grid_positions(n_ff, grid_size, keys[2])

    # E -> E connections (use sigma_e)
    ee_src, ee_dst, ee_weights = _spatial_connections(
        pos_e, pos_e, params.sigma_e, grid_size, p_base, params.J_ee, keys[3], allow_self=False
    )

    # I -> E connections (use sigma_i)
    ie_src, ie_dst, ie_weights = _spatial_connections(
        pos_i, pos_e, params.sigma_i, grid_size, p_base, params.J_ei, keys[4], allow_self=False
    )

    # E -> I connections (use sigma_e)
    ei_src, ei_dst, ei_weights = _spatial_connections(
        pos_e, pos_i, params.sigma_e, grid_size, p_base, params.J_ie, keys[5], allow_self=False
    )

    # I -> I connections (use sigma_i)
    ii_src, ii_dst, ii_weights = _spatial_connections(
        pos_i, pos_i, params.sigma_i, grid_size, p_base, params.J_ii, keys[6], allow_self=False
    )

    # Feedforward -> E (use sigma_F)
    fe_src, fe_dst, fe_weights = _spatial_connections(
        pos_ff, pos_e, params.sigma_F, grid_size, p_base, params.J_eF, keys[7], allow_self=True
    )

    # Feedforward -> I (use sigma_F)
    fi_src, fi_dst, fi_weights = _spatial_connections(
        pos_ff, pos_i, params.sigma_F, grid_size, p_base, params.J_iF, keys[7], allow_self=True
    )

    return Connectivity(
        ee_src=ee_src,
        ee_dst=ee_dst,
        ee_weights=ee_weights,
        ie_src=ie_src,
        ie_dst=ie_dst,
        ie_weights=ie_weights,
        ei_src=ei_src,
        ei_dst=ei_dst,
        ei_weights=ei_weights,
        ii_src=ii_src,
        ii_dst=ii_dst,
        ii_weights=ii_weights,
        fe_src=fe_src,
        fe_dst=fe_dst,
        fe_weights=fe_weights,
        fi_src=fi_src,
        fi_dst=fi_dst,
        fi_weights=fi_weights,
    )


def _random_connections(
    n_src: int,
    n_dst: int,
    prob: float,
    weight: float,
    rng_key: chex.PRNGKey,
    allow_self: bool = False,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Generate random connections with fixed probability."""
    # Sample connections
    conn_matrix = jax.random.bernoulli(rng_key, prob, shape=(n_src, n_dst))

    # Remove self-connections if not allowed
    if not allow_self and n_src == n_dst:
        conn_matrix = conn_matrix.at[jnp.diag_indices(n_src)].set(False)

    # Convert to edge list
    src_idx, dst_idx = jnp.where(conn_matrix)
    weights = jnp.full(src_idx.shape[0], weight, dtype=jnp.float32)

    return src_idx, dst_idx, weights


def _generate_grid_positions(
    n: int, grid_size: float, rng_key: chex.PRNGKey
) -> chex.Array:
    """Generate random positions on a square grid."""
    positions = jax.random.uniform(rng_key, shape=(n, 2), minval=0.0, maxval=grid_size)
    return positions


def _wrapped_distance(pos1: chex.Array, pos2: chex.Array, grid_size: float) -> chex.Array:
    """
    Compute wrapped (periodic) Euclidean distance on a square grid.

    Args:
        pos1: Positions (n1, 2)
        pos2: Positions (n2, 2)
        grid_size: Size of grid

    Returns:
        Distance matrix (n1, n2)
    """
    # Pairwise differences
    diff = pos1[:, None, :] - pos2[None, :, :]  # (n1, n2, 2)

    # Apply periodic wrapping
    diff = jnp.minimum(jnp.abs(diff), grid_size - jnp.abs(diff))

    # Euclidean distance
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))

    return dist


def _spatial_connections(
    pos_src: chex.Array,
    pos_dst: chex.Array,
    sigma: float,
    grid_size: float,
    p_base: float,
    weight: float,
    rng_key: chex.PRNGKey,
    allow_self: bool = False,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Generate connections with Gaussian distance-dependent probability.

    Connection probability: p(d) = p_base * exp(-d^2 / (2*sigma^2))

    Args:
        pos_src: Source positions (n_src, 2)
        pos_dst: Destination positions (n_dst, 2)
        sigma: Spatial width [mm]
        grid_size: Grid size [mm]
        p_base: Base connection probability
        weight: Synaptic weight [mV]
        rng_key: JAX random key
        allow_self: Allow self-connections

    Returns:
        (src_idx, dst_idx, weights) edge list
    """
    # Compute wrapped distances
    dist = _wrapped_distance(pos_src, pos_dst, grid_size)

    # Gaussian probability (handle sigma=0 case for CBN limit)
    if sigma > 1e-6:
        prob = p_base * jnp.exp(-0.5 * (dist / sigma) ** 2)
    else:
        # Uniform probability (CBN limit)
        prob = jnp.full_like(dist, p_base)

    # Sample connections
    conn_matrix = jax.random.bernoulli(rng_key, prob)

    # Remove self-connections if not allowed
    if not allow_self and pos_src.shape[0] == pos_dst.shape[0]:
        n = pos_src.shape[0]
        conn_matrix = conn_matrix.at[jnp.diag_indices(n)].set(False)

    # Convert to edge list
    src_idx, dst_idx = jnp.where(conn_matrix)
    weights = jnp.full(src_idx.shape[0], weight, dtype=jnp.float32)

    return src_idx, dst_idx, weights
