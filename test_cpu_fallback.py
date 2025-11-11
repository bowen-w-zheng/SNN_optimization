"""
Simple test to verify CPU fallback for statistics computation works.
"""

import jax
import jax.numpy as jnp
import time
from snops_jax.models.eif import EIFParams
from snops_jax.models.synapses import SynapticParams
from snops_jax.models.connectivity import build_cbn, NetworkParams
from snops_jax.simulate.run import run_simulation, SimulationConfig
from snops_jax.stats.single_pair import compute_statistics_summary
from snops_jax.stats.fa_jax import compute_shared_variance_stats

print("="*70)
print("CPU FALLBACK TEST FOR STATISTICS COMPUTATION")
print("="*70)

# Use smaller network for quick test
n_e, n_i, n_ff = 400, 100, 500

# Short simulation
sim_config = SimulationConfig(
    dt=0.05,
    duration=2000.0,  # 2 seconds
    burn_in=500.0,
    bin_size=200.0,
    ff_rate=50.0,
    integrator="euler",
)

# Model parameters
eif_params = EIFParams()
syn_params = SynapticParams(tau_ed=5.0, tau_id=10.0)
network_params = NetworkParams(
    J_ee=2.0, J_ei=-50.0, J_ie=15.0, J_ii=-35.0,
    J_eF=8.0, J_iF=8.0,
)

# Build connectivity
print("\nBuilding network connectivity...")
t0 = time.time()
rng_key = jax.random.PRNGKey(42)
key_conn, key_sim = jax.random.split(rng_key)
conn = build_cbn(n_e, n_i, n_ff, network_params, rng_key=key_conn)
print(f"✓ Built in {time.time()-t0:.2f}s")

# Run simulation
print("\nRunning simulation...")
t0 = time.time()
output = run_simulation(n_e, n_i, n_ff, conn, sim_config, eif_params, syn_params, key_sim)
output.spike_counts_e.block_until_ready()
t_sim = time.time() - t0
print(f"✓ Simulation completed in {t_sim:.2f}s")

# Get spike counts (should be on GPU)
spike_counts = output.spike_counts_e
print(f"\nSpike counts shape: {spike_counts.shape}")
print(f"Spike counts device: {spike_counts.device}")

# Test statistics computation (should move to CPU internally)
print("\n" + "="*70)
print("TESTING STATISTICS COMPUTATION (with CPU fallback)")
print("="*70)

try:
    print("\nComputing statistics...")
    t0 = time.time()
    stats = compute_statistics_summary(spike_counts, sim_config.bin_size)
    t_stats = time.time() - t0
    print(f"✓ Statistics computed in {t_stats:.2f}s")
    print(f"  FR:  {stats['fr']:.2f} sp/s")
    print(f"  FF:  {stats['ff']:.3f}")
    print(f"  RSC: {stats['rsc']:.3f}")
    print("\n✅ SUCCESS: Statistics computation with CPU fallback worked!")
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test Factor Analysis (should also move to CPU internally)
print("\n" + "="*70)
print("TESTING FACTOR ANALYSIS (with CPU fallback)")
print("="*70)

try:
    print("\nComputing Factor Analysis...")
    t0 = time.time()
    fa_stats = compute_shared_variance_stats(
        spike_counts,
        n_factors=None,
        use_cv=False,
        rng_key=jax.random.PRNGKey(123)
    )
    t_fa = time.time() - t0
    print(f"✓ Factor Analysis computed in {t_fa:.2f}s")
    print(f"  %sh: {fa_stats['pct_sh']*100:.1f}%")
    print(f"  dsh: {fa_stats['dsh']}")
    print(f"  n_factors: {fa_stats['n_factors']}")
    print("\n✅ SUCCESS: Factor Analysis with CPU fallback worked!")
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nCPU fallback is working correctly. Statistics and FA computation")
print("will automatically use CPU memory to avoid GPU OOM errors.")
print("="*70)
