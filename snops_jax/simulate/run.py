"""
Main simulation loops using lax.scan for GPU efficiency.
"""

import jax
import jax.numpy as jnp
from functools import partial
import chex

from snops_jax.simulate.step import (
    SimulationState,
    step_network,
    initialize_simulation,
    sample_poisson_feedforward,
)
from snops_jax.models.eif import EIFParams
from snops_jax.models.synapses import SynapticParams
from snops_jax.models.connectivity import Connectivity


@chex.dataclass
class SimulationConfig:
    """Configuration for simulation."""

    dt: float = 0.05  # Time step (ms) - paper default
    duration: float = 140.5  # Total duration (ms)
    burn_in: float = 0.5  # Burn-in period to discard (ms)
    bin_size: float = 200.0  # Bin size for spike counting (ms)
    ff_rate: float = 10.0  # Feedforward firing rate (sp/s)
    integrator: str = "euler"  # "euler", "heun", or "rk4"


@chex.dataclass
class SimulationOutput:
    """Output from simulation."""

    spike_counts_e: chex.Array  # (n_e, n_bins) spike counts
    spike_counts_i: chex.Array  # (n_i, n_bins) spike counts
    spike_times_e: chex.Array  # (n_steps,) list of spike times for E
    spike_times_i: chex.Array  # (n_steps,) list of spike times for I
    final_state: SimulationState  # Final state


def run_simulation(
    n_e: int,
    n_i: int,
    n_ff: int,
    conn: Connectivity,
    config: SimulationConfig,
    eif_params: EIFParams,
    syn_params: SynapticParams,
    rng_key: chex.PRNGKey,
) -> SimulationOutput:
    """
    Run full simulation with spike binning.

    Args:
        n_e: Number of excitatory neurons
        n_i: Number of inhibitory neurons
        n_ff: Number of feedforward inputs
        conn: Network connectivity
        config: Simulation configuration
        eif_params: EIF parameters
        syn_params: Synaptic parameters
        rng_key: JAX random key

    Returns:
        Simulation output with binned spike counts
    """
    # Initialize
    key_init, key_sim = jax.random.split(rng_key)
    initial_state = initialize_simulation(n_e, n_i, key_init, eif_params)

    # Calculate number of steps
    n_steps = int(config.duration / config.dt)
    n_burn = int(config.burn_in / config.dt)
    bin_steps = int(config.bin_size / config.dt)
    n_bins = (n_steps - n_burn) // bin_steps

    # Pre-generate feedforward spikes for all timesteps
    key_ff = jax.random.split(key_sim, n_steps)

    @jax.jit
    def generate_ff_spikes(key):
        key_e, key_i = jax.random.split(key)
        ff_e = sample_poisson_feedforward(n_ff, config.ff_rate, config.dt, key_e)
        ff_i = sample_poisson_feedforward(n_ff, config.ff_rate, config.dt, key_i)
        return ff_e, ff_i

    ff_spikes = jax.vmap(generate_ff_spikes)(key_ff)
    ff_spikes_e_all = ff_spikes[0]  # (n_steps, n_ff)
    ff_spikes_i_all = ff_spikes[1]

    # Simulation step function for lax.scan
    @jax.jit
    def scan_fn(carry, inputs):
        state, spike_bin_e, spike_bin_i, step_idx = carry
        ff_e, ff_i = inputs

        # Step the network
        state_next = step_network(
            state, conn, ff_e, ff_i, config.dt, eif_params, syn_params, config.integrator
        )

        # Accumulate spikes in current bin (only after burn-in)
        is_recording = step_idx >= n_burn
        bin_idx = (step_idx - n_burn) // bin_steps
        in_valid_bin = (bin_idx >= 0) & (bin_idx < n_bins)

        # Update spike bins (branchless)
        spike_bin_e_next = jnp.where(
            is_recording & in_valid_bin,
            spike_bin_e.at[bin_idx].add(state_next.neurons_e.spikes.astype(jnp.float32)),
            spike_bin_e,
        )
        spike_bin_i_next = jnp.where(
            is_recording & in_valid_bin,
            spike_bin_i.at[bin_idx].add(state_next.neurons_i.spikes.astype(jnp.float32)),
            spike_bin_i,
        )

        carry_next = (state_next, spike_bin_e_next, spike_bin_i_next, step_idx + 1)

        # Return spikes for this timestep (for optional spike time extraction)
        outputs = (state_next.neurons_e.spikes, state_next.neurons_i.spikes)

        return carry_next, outputs

    # Initialize spike bins (n_bins, n_neurons)
    spike_bins_e = jnp.zeros((n_bins, n_e), dtype=jnp.float32)
    spike_bins_i = jnp.zeros((n_bins, n_i), dtype=jnp.float32)

    # Run simulation
    initial_carry = (initial_state, spike_bins_e, spike_bins_i, 0)
    inputs = (ff_spikes_e_all, ff_spikes_i_all)

    final_carry, outputs = jax.lax.scan(scan_fn, initial_carry, inputs)
    final_state, final_bins_e, final_bins_i, _ = final_carry

    # Transpose to (n_neurons, n_bins)
    spike_counts_e = final_bins_e.T
    spike_counts_i = final_bins_i.T

    # Extract spike times (optional, for detailed analysis)
    spikes_e_all, spikes_i_all = outputs

    return SimulationOutput(
        spike_counts_e=spike_counts_e,
        spike_counts_i=spike_counts_i,
        spike_times_e=spikes_e_all,  # (n_steps, n_e)
        spike_times_i=spikes_i_all,  # (n_steps, n_i)
        final_state=final_state,
    )


def run_short_simulation(
    n_e: int,
    n_i: int,
    n_ff: int,
    conn: Connectivity,
    eif_params: EIFParams,
    syn_params: SynapticParams,
    rng_key: chex.PRNGKey,
    duration: float = 10000.0,  # 10 seconds (ms)
    dt: float = 0.05,
    ff_rate: float = 10.0,
    integrator: str = "euler",
) -> tuple[chex.Array, chex.Array, SimulationState]:
    """
    Run short simulation for feasibility checking.

    Returns mean firing rates and final state for stability checks.

    Args:
        n_e, n_i, n_ff: Network sizes
        conn: Connectivity
        eif_params, syn_params: Model parameters
        rng_key: Random key
        duration: Simulation duration (ms)
        dt: Time step (ms)
        ff_rate: Feedforward rate (sp/s)
        integrator: Integration method

    Returns:
        (mean_fr_e, mean_fr_i, final_state)
        where mean_fr_* are firing rates in sp/s
    """
    # Use simplified config for short run
    config = SimulationConfig(
        dt=dt,
        duration=duration,
        burn_in=0.0,  # No burn-in for short run
        bin_size=200.0,
        ff_rate=ff_rate,
        integrator=integrator,
    )

    output = run_simulation(n_e, n_i, n_ff, conn, config, eif_params, syn_params, rng_key)

    # Compute mean firing rates
    total_spikes_e = jnp.sum(output.spike_counts_e, axis=1)
    total_spikes_i = jnp.sum(output.spike_counts_i, axis=1)

    n_bins = output.spike_counts_e.shape[1]
    total_time_s = (n_bins * config.bin_size) / 1000.0

    mean_fr_e = total_spikes_e / total_time_s
    mean_fr_i = total_spikes_i / total_time_s

    return mean_fr_e, mean_fr_i, output.final_state


@partial(jax.jit, static_argnums=(0, 1, 2))
def run_simulation_batched(
    n_e: int,
    n_i: int,
    n_ff: int,
    conn_batch: Connectivity,  # Batched connectivity (leading batch dim)
    config: SimulationConfig,
    eif_params: EIFParams,
    syn_params: SynapticParams,
    rng_keys: chex.Array,  # (batch_size,) array of keys
) -> SimulationOutput:
    """
    Run simulation in batch over multiple parameter sets (vmap).

    Args:
        n_e, n_i, n_ff: Network sizes
        conn_batch: Batched connectivity
        config: Simulation config
        eif_params: EIF params
        syn_params: Synaptic params
        rng_keys: Batch of random keys

    Returns:
        Batched simulation output
    """
    # vmap over batch dimension
    run_fn = partial(
        run_simulation,
        n_e=n_e,
        n_i=n_i,
        n_ff=n_ff,
        config=config,
        eif_params=eif_params,
    )

    # This requires connectivity to have a batch dimension
    # We'll vmap over (conn, syn_params, rng_key)
    batched_run = jax.vmap(run_fn, in_axes=(0, 0, 0))

    return batched_run(conn_batch, syn_params, rng_keys)
