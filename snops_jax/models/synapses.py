"""
Synaptic current dynamics for SNOPS.

Implements first-order exponential decay with spike-triggered increments.
"""

import jax.numpy as jnp
from typing import NamedTuple
import chex


class SynapticParams(NamedTuple):
    """Parameters for synaptic currents."""

    # Decay time constants (free parameters)
    tau_ed: float = 5.0  # E synaptic decay time constant (ms), range [1, 25]
    tau_id: float = 5.0  # I synaptic decay time constant (ms), range [1, 25]

    # Feedforward time constant (can be same as tau_ed or separate)
    tau_fd: float = 5.0  # Feedforward synaptic decay (ms)

    def to_dict(self):
        """Convert to dictionary."""
        return self._asdict()


@chex.dataclass
class SynapticState:
    """State of synaptic currents for a population."""
    s_e: chex.Array  # Excitatory synaptic current (N,) [mV]
    s_i: chex.Array  # Inhibitory synaptic current (N,) [mV]
    s_f: chex.Array  # Feedforward synaptic current (N,) [mV]


def update_synapses(
    state: SynapticState,
    e_increments: chex.Array,
    i_increments: chex.Array,
    f_increments: chex.Array,
    dt: float,
    params: SynapticParams,
) -> SynapticState:
    """
    Update synaptic currents with exponential decay and spike-triggered increments.

    Dynamics:
        ds/dt = -s/tau + Σ_spikes J_αβ δ(t - t_spike)

    Discrete update (Euler):
        s_{t+dt} = s_t + dt * (-s_t / tau) + Σ_spikes J_αβ

    Args:
        state: Current synaptic state
        e_increments: Sum of excitatory weights from spiking neurons (N,) [mV]
        i_increments: Sum of inhibitory weights from spiking neurons (N,) [mV]
        f_increments: Feedforward increments (N,) [mV]
        dt: Time step (ms)
        params: Synaptic parameters

    Returns:
        Updated synaptic state
    """
    # Exponential decay
    s_e_next = state.s_e + dt * (-state.s_e / params.tau_ed)
    s_i_next = state.s_i + dt * (-state.s_i / params.tau_id)
    s_f_next = state.s_f + dt * (-state.s_f / params.tau_fd)

    # Add spike-triggered increments
    s_e_next += e_increments
    s_i_next += i_increments
    s_f_next += f_increments

    return SynapticState(s_e=s_e_next, s_i=s_i_next, s_f=s_f_next)


def compute_total_current(state: SynapticState) -> chex.Array:
    """
    Compute total synaptic current.

    Total current I = s_e - s_i + s_f
    (Excitatory and feedforward are positive, inhibitory is negative)

    Args:
        state: Synaptic state

    Returns:
        Total current (N,) [mV]
    """
    return state.s_e - state.s_i + state.s_f


def initialize_synapses(n_neurons: int) -> SynapticState:
    """
    Initialize synaptic state to zero.

    Args:
        n_neurons: Number of neurons

    Returns:
        Initial synaptic state
    """
    return SynapticState(
        s_e=jnp.zeros(n_neurons, dtype=jnp.float32),
        s_i=jnp.zeros(n_neurons, dtype=jnp.float32),
        s_f=jnp.zeros(n_neurons, dtype=jnp.float32),
    )


def compute_synaptic_increments(
    spikes_e: chex.Array,
    spikes_i: chex.Array,
    weights_ee: chex.Array,
    weights_ie: chex.Array,
    weights_ei: chex.Array,
    weights_ii: chex.Array,
    edge_list_ee: tuple,
    edge_list_ie: tuple,
    edge_list_ei: tuple,
    edge_list_ii: tuple,
) -> tuple[chex.Array, chex.Array]:
    """
    Compute synaptic increments from recurrent spikes using edge lists.

    Args:
        spikes_e: E neuron spikes (n_e,) [bool]
        spikes_i: I neuron spikes (n_i,) [bool]
        weights_*: Weight arrays for each connection type
        edge_list_*: (src_idx, dst_idx) tuples for each connection type

    Returns:
        (e_increments, i_increments) for E and I populations
    """
    n_e = spikes_e.shape[0]
    n_i = spikes_i.shape[0]

    # Initialize increments
    e_inc = jnp.zeros(n_e, dtype=jnp.float32)
    i_inc = jnp.zeros(n_i, dtype=jnp.float32)

    # E -> E connections
    if edge_list_ee[0].shape[0] > 0:
        src_ee, dst_ee = edge_list_ee
        spike_contrib = spikes_e[src_ee].astype(jnp.float32) * weights_ee
        e_inc = e_inc.at[dst_ee].add(spike_contrib)

    # I -> E connections (inhibitory)
    if edge_list_ie[0].shape[0] > 0:
        src_ie, dst_ie = edge_list_ie
        spike_contrib = spikes_i[src_ie].astype(jnp.float32) * weights_ie
        i_inc = i_inc.at[dst_ie].add(spike_contrib)

    # E -> I connections
    if edge_list_ei[0].shape[0] > 0:
        src_ei, dst_ei = edge_list_ei
        spike_contrib = spikes_e[src_ei].astype(jnp.float32) * weights_ei
        e_inc_i = jnp.zeros(n_i, dtype=jnp.float32)
        e_inc_i = e_inc_i.at[dst_ei].add(spike_contrib)
    else:
        e_inc_i = jnp.zeros(n_i, dtype=jnp.float32)

    # I -> I connections (inhibitory)
    if edge_list_ii[0].shape[0] > 0:
        src_ii, dst_ii = edge_list_ii
        spike_contrib = spikes_i[src_ii].astype(jnp.float32) * weights_ii
        i_inc_i = jnp.zeros(n_i, dtype=jnp.float32)
        i_inc_i = i_inc_i.at[dst_ii].add(spike_contrib)
    else:
        i_inc_i = jnp.zeros(n_i, dtype=jnp.float32)

    return (e_inc, e_inc_i), (i_inc, i_inc_i)
