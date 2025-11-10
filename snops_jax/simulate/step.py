"""
Core time-stepping kernel for spiking network simulation.

GPU-optimized JAX implementation with branchless spike/reset handling.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable
import chex

from snops_jax.models.eif import (
    EIFParams,
    NeuronState,
    update_membrane_euler,
    update_membrane_heun,
    update_membrane_rk4,
    initialize_neurons,
)
from snops_jax.models.synapses import (
    SynapticParams,
    SynapticState,
    update_synapses,
    compute_total_current,
    initialize_synapses,
)
from snops_jax.models.connectivity import Connectivity


@chex.dataclass
class SimulationState:
    """Complete state of a spiking network simulation."""

    # Neuron states
    neurons_e: NeuronState  # Excitatory population
    neurons_i: NeuronState  # Inhibitory population

    # Synaptic states
    synapses_e: SynapticState  # E population synapses
    synapses_i: SynapticState  # I population synapses

    # Time
    t: float  # Current time (ms)


def step_network(
    state: SimulationState,
    conn: Connectivity,
    ff_spikes_e: chex.Array,  # Feedforward spikes to E (n_ff,) [bool]
    ff_spikes_i: chex.Array,  # Feedforward spikes to I (n_ff,) [bool]
    dt: float,
    eif_params: EIFParams,
    syn_params: SynapticParams,
    integrator: str = "euler",
) -> SimulationState:
    """
    Execute one time step of the network simulation.

    This is the core kernel that will be called in a lax.scan loop.

    Args:
        state: Current simulation state
        conn: Network connectivity (edge lists)
        ff_spikes_e: Feedforward spikes to E neurons
        ff_spikes_i: Feedforward spikes to I neurons
        dt: Time step (ms)
        eif_params: EIF neuron parameters
        syn_params: Synaptic parameters
        integrator: One of "euler", "heun", "rk4"

    Returns:
        Updated simulation state
    """
    # 1. Compute recurrent synaptic increments from previous spikes
    e_inc, i_inc = _compute_recurrent_increments(
        state.neurons_e.spikes, state.neurons_i.spikes, conn
    )

    # 2. Compute feedforward increments
    fe_inc = _compute_feedforward_increments(ff_spikes_e, conn.fe_src, conn.fe_dst, conn.fe_weights, state.synapses_e.s_e.shape[0])
    fi_inc = _compute_feedforward_increments(ff_spikes_i, conn.fi_src, conn.fi_dst, conn.fi_weights, state.synapses_i.s_e.shape[0])

    # 3. Update synaptic currents
    synapses_e_next = update_synapses(
        state.synapses_e,
        e_inc[0],  # E increments to E
        i_inc[0],  # I increments to E
        fe_inc,  # Feedforward to E
        dt,
        syn_params,
    )

    synapses_i_next = update_synapses(
        state.synapses_i,
        e_inc[1],  # E increments to I
        i_inc[1],  # I increments to I
        fi_inc,  # Feedforward to I
        dt,
        syn_params,
    )

    # 4. Compute total currents
    I_total_e = compute_total_current(synapses_e_next)
    I_total_i = compute_total_current(synapses_i_next)

    # 5. Update membrane potentials
    update_fn = {
        "euler": update_membrane_euler,
        "heun": update_membrane_heun,
        "rk4": update_membrane_rk4,
    }[integrator]

    neurons_e_next = update_fn(state.neurons_e, I_total_e, dt, eif_params)
    neurons_i_next = update_fn(state.neurons_i, I_total_i, dt, eif_params)

    # 6. Advance time
    t_next = state.t + dt

    return SimulationState(
        neurons_e=neurons_e_next,
        neurons_i=neurons_i_next,
        synapses_e=synapses_e_next,
        synapses_i=synapses_i_next,
        t=t_next,
    )


def _compute_recurrent_increments(
    spikes_e: chex.Array, spikes_i: chex.Array, conn: Connectivity
) -> tuple[tuple[chex.Array, chex.Array], tuple[chex.Array, chex.Array]]:
    """
    Compute synaptic increments from recurrent connections.

    Returns:
        ((e_inc_to_e, e_inc_to_i), (i_inc_to_e, i_inc_to_i))
    """
    n_e = spikes_e.shape[0]
    n_i = spikes_i.shape[0]

    # E -> E
    e_to_e = _scatter_spikes(spikes_e, conn.ee_src, conn.ee_dst, conn.ee_weights, n_e)

    # I -> E
    i_to_e = _scatter_spikes(spikes_i, conn.ie_src, conn.ie_dst, conn.ie_weights, n_e)

    # E -> I
    e_to_i = _scatter_spikes(spikes_e, conn.ei_src, conn.ei_dst, conn.ei_weights, n_i)

    # I -> I
    i_to_i = _scatter_spikes(spikes_i, conn.ii_src, conn.ii_dst, conn.ii_weights, n_i)

    return ((e_to_e, e_to_i), (i_to_e, i_to_i))


def _compute_feedforward_increments(
    ff_spikes: chex.Array,
    src_idx: chex.Array,
    dst_idx: chex.Array,
    weights: chex.Array,
    n_dst: int,
) -> chex.Array:
    """Compute feedforward synaptic increments."""
    return _scatter_spikes(ff_spikes, src_idx, dst_idx, weights, n_dst)


def _scatter_spikes(
    spikes: chex.Array,
    src_idx: chex.Array,
    dst_idx: chex.Array,
    weights: chex.Array,
    n_dst: int,
) -> chex.Array:
    """
    Scatter spike-triggered increments to postsynaptic neurons.

    Args:
        spikes: Spike indicators (n_src,) [bool]
        src_idx: Source neuron indices for each edge
        dst_idx: Destination neuron indices for each edge
        weights: Weight for each edge
        n_dst: Number of destination neurons

    Returns:
        Increments to destination neurons (n_dst,)
    """
    # Get spike contributions for each edge
    spike_contrib = spikes[src_idx].astype(jnp.float32) * weights

    # Sum contributions to each destination neuron using segment_sum
    increments = jnp.zeros(n_dst, dtype=jnp.float32)
    increments = increments.at[dst_idx].add(spike_contrib)

    return increments


def initialize_simulation(
    n_e: int,
    n_i: int,
    rng_key: chex.PRNGKey,
    eif_params: EIFParams,
) -> SimulationState:
    """
    Initialize simulation state.

    Args:
        n_e: Number of excitatory neurons
        n_i: Number of inhibitory neurons
        rng_key: JAX random key
        eif_params: EIF parameters

    Returns:
        Initial simulation state
    """
    key_e, key_i = jax.random.split(rng_key)

    neurons_e = initialize_neurons(n_e, key_e, eif_params)
    neurons_i = initialize_neurons(n_i, key_i, eif_params)

    synapses_e = initialize_synapses(n_e)
    synapses_i = initialize_synapses(n_i)

    return SimulationState(
        neurons_e=neurons_e,
        neurons_i=neurons_i,
        synapses_e=synapses_e,
        synapses_i=synapses_i,
        t=0.0,
    )


def sample_poisson_feedforward(
    n_ff: int, rate: float, dt: float, rng_key: chex.PRNGKey
) -> chex.Array:
    """
    Sample Poisson feedforward spikes.

    Args:
        n_ff: Number of feedforward inputs
        rate: Firing rate (sp/s)
        dt: Time step (ms)
        rng_key: JAX random key

    Returns:
        Spike indicators (n_ff,) [bool]
    """
    # Probability of spike in dt
    p_spike = rate * (dt / 1000.0)  # Convert dt to seconds
    p_spike = jnp.clip(p_spike, 0.0, 1.0)

    # Sample Bernoulli
    spikes = jax.random.bernoulli(rng_key, p_spike, shape=(n_ff,))

    return spikes
