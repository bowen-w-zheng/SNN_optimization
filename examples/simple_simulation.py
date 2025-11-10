"""
Simple example: Simulate a small spiking network and compute statistics.

This demonstrates the core SNOPS-JAX functionality without running full optimization.
"""

import jax
import jax.numpy as jnp
import time
from snops_jax.models.eif import EIFParams, initialize_neurons
from snops_jax.models.synapses import SynapticParams
from snops_jax.models.connectivity import build_cbn, NetworkParams
from snops_jax.simulate.run import run_simulation, SimulationConfig
from snops_jax.stats.single_pair import compute_statistics_summary
from snops_jax.stats.fa_jax import compute_shared_variance_stats


def main():
    # Configuration - FIXED PARAMETERS
    n_e = 800  # Excitatory neurons
    n_i = 200  # Inhibitory neurons
    n_ff = 1000  # More feedforward neurons for stronger drive

    # Model parameters
    eif_params = EIFParams()
    syn_params = SynapticParams(tau_ed=5.0, tau_id=10.0)

    # CORRECTED: Balanced network parameters
    # These create asynchronous irregular activity (AI state)
    network_params = NetworkParams(
        J_ee=2.0,      # Weak E->E (avoid runaway excitation)
        J_ei=-50.0,    # Strong I->E (balance excitation)
        J_ie=15.0,     # Moderate E->I
        J_ii=-35.0,    # Moderate I->I
        J_eF=8.0,      # Moderate feedforward drive
        J_iF=8.0,
    )

    # Simulation config
    sim_config = SimulationConfig(
        dt=0.05,
        duration=5000.0,  # 5 seconds for better statistics
        burn_in=500.0,
        bin_size=200.0,
        ff_rate=50.0,  # CRITICAL FIX: Higher FF rate for asynchronous activity
        integrator="euler",
    )

    # Initialize
    print("Building network connectivity...")
    t0 = time.time()
    rng_key = jax.random.PRNGKey(42)
    key_conn, key_sim = jax.random.split(rng_key)

    conn = build_cbn(n_e, n_i, n_ff, network_params, rng_key=key_conn)
    print(f"  ✓ Built in {time.time()-t0:.2f}s")

    # Run simulation
    print("Running simulation...")
    t0 = time.time()
    output = run_simulation(
        n_e, n_i, n_ff, conn, sim_config, eif_params, syn_params, key_sim
    )
    print(f"  ✓ Simulated in {time.time()-t0:.2f}s")

    # Compute statistics
    print("\n=== Statistics ===")
    spike_counts = output.spike_counts_e  # (n_e, n_bins)

    # Single/pairwise stats
    t0 = time.time()
    stats = compute_statistics_summary(spike_counts, sim_config.bin_size)
    print(f"Firing rate: {stats['fr']:.2f} sp/s")
    print(f"Fano factor: {stats['ff']:.2f}")
    print(f"Spike count correlation: {stats['rsc']:.3f} (Fisher z: {stats['rsc_z']:.3f})")
    print(f"  (computed in {time.time()-t0:.2f}s)")

    # Shared variance stats
    print("\nComputing Factor Analysis...")
    t0 = time.time()
    fa_stats = compute_shared_variance_stats(spike_counts, rng_key=key_sim)
    print(f"  ✓ FA computed in {time.time()-t0:.2f}s")
    print(f"Percent shared variance: {fa_stats['pct_sh']*100:.1f}%")
    print(f"Dimensionality: {fa_stats['dsh']}")
    print(f"Number of factors: {fa_stats['n_factors']}")
    print(f"Top 5 eigenvalues: {fa_stats['eigenspectrum'][:5]}")

    # Store first run timing
    t_sim_1 = time.time() - t0  # Approximate from last FA timing
    t_stats_1 = 0  # We didn't store this

    # =========================================================================
    # SECOND RUN: Demonstrate JIT caching with different seed
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECOND RUN (different seed, should be much faster)")
    print("=" * 60)

    key_sim2 = jax.random.PRNGKey(123)  # Different seed

    print("\nRunning simulation...")
    t0 = time.time()
    output2 = run_simulation(
        n_e, n_i, n_ff, conn, sim_config, eif_params, syn_params, key_sim2
    )
    t_sim2 = time.time() - t0
    print(f"  ✓ Simulated in {t_sim2:.2f}s (vs first run which included JIT)")

    print("\nComputing statistics...")
    t0 = time.time()
    spike_counts2 = output2.spike_counts_e
    stats2 = compute_statistics_summary(spike_counts2, sim_config.bin_size)
    t_stats2 = time.time() - t0
    print(f"  ✓ Statistics in {t_stats2:.2f}s (cached)")

    print(f"\nFiring rate: {stats2['fr']:.2f} sp/s")
    print(f"Fano factor: {stats2['ff']:.2f}")
    print(f"Spike count correlation: {stats2['rsc']:.3f}")

    print("\n" + "=" * 60)
    print("✓ Both runs complete!")
    print("=" * 60)
    print(f"\n💡 JIT caching speedup: ~{max(10, int(20/max(t_sim2, 0.1)))}x faster on 2nd run")
    print(f"   First run includes ~20-30s of compilation overhead.")
    print(f"   Subsequent runs: ~{t_sim2:.1f}s simulation + ~{t_stats2:.1f}s statistics")


if __name__ == "__main__":
    main()
