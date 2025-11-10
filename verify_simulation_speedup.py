"""
Quick verification: Simulation JIT caching works correctly after fix.
"""

import jax
import jax.numpy as jnp
import time
from snops_jax.models.eif import EIFParams
from snops_jax.models.synapses import SynapticParams
from snops_jax.models.connectivity import build_cbn, NetworkParams
from snops_jax.simulate.run import run_simulation, SimulationConfig

print("="*70)
print("SIMULATION JIT SPEEDUP VERIFICATION")
print("="*70)

# Network configuration
n_e, n_i, n_ff = 800, 200, 1000

# Simulation config
sim_config = SimulationConfig(
    dt=0.05,
    duration=5000.0,
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
key_conn, _ = jax.random.split(rng_key)
conn = build_cbn(n_e, n_i, n_ff, network_params, rng_key=key_conn)
print(f"✓ Built in {time.time()-t0:.2f}s")

# =========================================================================
# RUN 1: First execution (includes JIT compilation)
# =========================================================================
print(f"\n{'='*70}")
print("RUN 1: First execution (includes JIT compilation)")
print(f"{'='*70}")

key_sim_1 = jax.random.PRNGKey(100)

t_start = time.time()
output_1 = run_simulation(n_e, n_i, n_ff, conn, sim_config, eif_params, syn_params, key_sim_1)
# Force computation
output_1.spike_counts_e.block_until_ready()
t_sim_1 = time.time() - t_start

print(f"\n⏱️  Simulation time: {t_sim_1:.2f}s")
print(f"📊 FR: {jnp.mean(jnp.sum(output_1.spike_counts_e, axis=1) / (sim_config.duration - sim_config.burn_in) * 1000):.2f} sp/s")

# =========================================================================
# RUN 2: Same config, different seed (should be FAST - cached JIT)
# =========================================================================
print(f"\n{'='*70}")
print("RUN 2: Same config, different seed (cached JIT)")
print(f"{'='*70}")

key_sim_2 = jax.random.PRNGKey(200)

t_start = time.time()
output_2 = run_simulation(n_e, n_i, n_ff, conn, sim_config, eif_params, syn_params, key_sim_2)
output_2.spike_counts_e.block_until_ready()
t_sim_2 = time.time() - t_start

print(f"\n⏱️  Simulation time: {t_sim_2:.2f}s")
print(f"📊 FR: {jnp.mean(jnp.sum(output_2.spike_counts_e, axis=1) / (sim_config.duration - sim_config.burn_in) * 1000):.2f} sp/s")

# =========================================================================
# RUN 3: One more time to confirm consistency
# =========================================================================
print(f"\n{'='*70}")
print("RUN 3: One more run (should also be fast)")
print(f"{'='*70}")

key_sim_3 = jax.random.PRNGKey(300)

t_start = time.time()
output_3 = run_simulation(n_e, n_i, n_ff, conn, sim_config, eif_params, syn_params, key_sim_3)
output_3.spike_counts_e.block_until_ready()
t_sim_3 = time.time() - t_start

print(f"\n⏱️  Simulation time: {t_sim_3:.2f}s")
print(f"📊 FR: {jnp.mean(jnp.sum(output_3.spike_counts_e, axis=1) / (sim_config.duration - sim_config.burn_in) * 1000):.2f} sp/s")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

speedup_2 = t_sim_1 / max(t_sim_2, 0.01)
speedup_3 = t_sim_1 / max(t_sim_3, 0.01)

print(f"\nSimulation Times:")
print(f"  Run 1 (first - JIT):  {t_sim_1:6.2f}s  ← includes compilation")
print(f"  Run 2 (cached):       {t_sim_2:6.2f}s  ← {speedup_2:.1f}x faster!")
print(f"  Run 3 (cached):       {t_sim_3:6.2f}s  ← {speedup_3:.1f}x faster!")

print(f"\n✅ VERIFICATION:")
if speedup_2 >= 2.0 and speedup_3 >= 2.0:
    print(f"  ✓ SUCCESS! Simulation JIT caching works correctly!")
    print(f"  ✓ Achieved {speedup_2:.1f}x and {speedup_3:.1f}x speedups")
    if speedup_2 >= 10.0:
        print(f"  ✓ EXCELLENT performance (≥10x speedup)")
elif speedup_2 >= 1.5:
    print(f"  ⚠  PARTIAL SUCCESS: {speedup_2:.1f}x speedup (expected ≥2x)")
    print(f"  ⚠  May still have some overhead, but significantly better than before")
else:
    print(f"  ✗ FAILED: Only {speedup_2:.1f}x speedup (expected ≥2x)")
    print(f"  ✗ JIT caching may not be working properly")

print("\n" + "="*70)
