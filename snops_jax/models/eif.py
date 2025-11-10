"""
Exponential Integrate-and-Fire (EIF) neuron model.

Implements the membrane dynamics from the SNOPS paper with branchless spike/reset handling.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple
import chex


class EIFParams(NamedTuple):
    """Parameters for EIF neuron model (paper values)."""

    # Membrane parameters
    C_m: float = 1.0  # Membrane capacitance (normalized, tau_m = C_m / g_L)
    g_L: float = 1.0 / 15.0  # Leak conductance (1/tau_m where tau_m = 15 ms)
    E_L: float = -60.0  # Leak reversal potential (mV)
    V_T: float = -50.0  # Spike threshold (mV)
    Delta_T: float = 2.0  # Spike slope factor (mV)
    V_th: float = -10.0  # Hard threshold for spike detection (mV)
    V_re: float = -65.0  # Reset potential (mV)
    tau_ref: float = 1.5  # Refractory period (ms)

    @property
    def tau_m(self) -> float:
        """Membrane time constant (ms)."""
        return self.C_m / self.g_L

    def to_dict(self):
        """Convert to dictionary."""
        return self._asdict()


@chex.dataclass
class NeuronState:
    """State of a population of EIF neurons."""
    V: chex.Array  # Membrane potential (N,) [mV]
    ref_count: chex.Array  # Refractory counter in steps (N,) [int]
    spikes: chex.Array  # Spike indicator for current timestep (N,) [bool]


def eif_derivative(V: chex.Array, I_total: chex.Array, params: EIFParams) -> chex.Array:
    """
    Compute dV/dt for the EIF model.

    Membrane dynamics:
        C_m * dV/dt = -g_L(V - E_L) + g_L*Delta_T*exp((V - V_T)/Delta_T) + I_total

    Args:
        V: Membrane potential (N,) [mV]
        I_total: Total input current (N,) [mV, since we use current in mV units]
        params: EIF parameters

    Returns:
        dV/dt (N,) [mV/ms]
    """
    # Leak current
    I_leak = -params.g_L * (V - params.E_L)

    # Exponential term (clamped to avoid overflow)
    exp_arg = (V - params.V_T) / params.Delta_T
    exp_arg = jnp.clip(exp_arg, -10.0, 10.0)  # Prevent numerical overflow
    I_exp = params.g_L * params.Delta_T * jnp.exp(exp_arg)

    # Total derivative
    dV_dt = (I_leak + I_exp + I_total) / params.C_m

    return dV_dt


def update_membrane_euler(
    state: NeuronState,
    I_total: chex.Array,
    dt: float,
    params: EIFParams,
) -> NeuronState:
    """
    Update membrane potential using forward Euler (paper baseline).

    Implements branchless spike detection and reset logic.

    Args:
        state: Current neuron state
        I_total: Total synaptic current (N,) [mV]
        dt: Time step (ms)
        params: EIF parameters

    Returns:
        Updated neuron state with spike flags
    """
    # 1. Compute derivative (only for non-refractory neurons)
    can_integrate = state.ref_count == 0
    dV_dt = eif_derivative(state.V, I_total, params)

    # 2. Euler step
    V_next = state.V + dt * dV_dt

    # 3. Detect spikes (only for non-refractory neurons)
    crossed_threshold = V_next >= params.V_th
    spikes = can_integrate & crossed_threshold

    # 4. Apply reset (branchless)
    V_next = jnp.where(spikes, params.V_re, V_next)

    # 5. Update refractory counter
    ref_steps = jnp.ceil(params.tau_ref / dt).astype(jnp.int32)
    ref_next = jnp.where(
        spikes,
        ref_steps,  # Start refractory period
        jnp.maximum(state.ref_count - 1, 0),  # Decrement if > 0
    )

    return NeuronState(V=V_next, ref_count=ref_next, spikes=spikes)


def update_membrane_heun(
    state: NeuronState,
    I_total: chex.Array,
    dt: float,
    params: EIFParams,
) -> NeuronState:
    """
    Update membrane potential using Heun's method (improved Euler).

    Args:
        state: Current neuron state
        I_total: Total synaptic current (N,) [mV]
        dt: Time step (ms)
        params: EIF parameters

    Returns:
        Updated neuron state
    """
    can_integrate = state.ref_count == 0

    # Predictor step (Euler)
    k1 = eif_derivative(state.V, I_total, params)
    V_pred = state.V + dt * k1

    # Corrector step
    k2 = eif_derivative(V_pred, I_total, params)  # Assumes I_total constant over dt
    V_next = state.V + (dt / 2.0) * (k1 + k2)

    # Spike detection and reset
    crossed_threshold = V_next >= params.V_th
    spikes = can_integrate & crossed_threshold
    V_next = jnp.where(spikes, params.V_re, V_next)

    # Refractory counter
    ref_steps = jnp.ceil(params.tau_ref / dt).astype(jnp.int32)
    ref_next = jnp.where(spikes, ref_steps, jnp.maximum(state.ref_count - 1, 0))

    return NeuronState(V=V_next, ref_count=ref_next, spikes=spikes)


def update_membrane_rk4(
    state: NeuronState,
    I_total: chex.Array,
    dt: float,
    params: EIFParams,
) -> NeuronState:
    """
    Update membrane potential using 4th-order Runge-Kutta.

    Note: This assumes I_total is constant over the timestep.
    For more accurate higher-order integration with varying inputs,
    consider using diffrax with event handling.

    Args:
        state: Current neuron state
        I_total: Total synaptic current (N,) [mV]
        dt: Time step (ms)
        params: EIF parameters

    Returns:
        Updated neuron state
    """
    can_integrate = state.ref_count == 0

    # RK4 stages
    k1 = eif_derivative(state.V, I_total, params)
    k2 = eif_derivative(state.V + 0.5 * dt * k1, I_total, params)
    k3 = eif_derivative(state.V + 0.5 * dt * k2, I_total, params)
    k4 = eif_derivative(state.V + dt * k3, I_total, params)

    V_next = state.V + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Spike detection and reset
    crossed_threshold = V_next >= params.V_th
    spikes = can_integrate & crossed_threshold
    V_next = jnp.where(spikes, params.V_re, V_next)

    # Refractory counter
    ref_steps = jnp.ceil(params.tau_ref / dt).astype(jnp.int32)
    ref_next = jnp.where(spikes, ref_steps, jnp.maximum(state.ref_count - 1, 0))

    return NeuronState(V=V_next, ref_count=ref_next, spikes=spikes)


def initialize_neurons(n_neurons: int, rng_key: chex.PRNGKey, params: EIFParams) -> NeuronState:
    """
    Initialize neuron state with random membrane potentials.

    Args:
        n_neurons: Number of neurons
        rng_key: JAX random key
        params: EIF parameters

    Returns:
        Initial neuron state
    """
    # Initialize V uniformly between E_L and V_T
    V_init = jax.random.uniform(
        rng_key,
        shape=(n_neurons,),
        minval=params.E_L,
        maxval=params.V_T,
    )

    ref_count = jnp.zeros(n_neurons, dtype=jnp.int32)
    spikes = jnp.zeros(n_neurons, dtype=bool)

    return NeuronState(V=V_init, ref_count=ref_count, spikes=spikes)
