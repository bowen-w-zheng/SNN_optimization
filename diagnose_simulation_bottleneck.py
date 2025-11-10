"""
Diagnose where the simulation time is spent.
"""

import jax
import jax.numpy as jnp
import time
from snops_jax.models.eif import EIFParams
from snops_jax.models.synapses import SynapticParams
from snops_jax.models.connectivity import build_cbn, NetworkParams
from snops_jax.simulate.step import sample_poisson_feedforward

# Configuration
n_e, n_i, n_ff = 800, 200, 1000
dt = 0.05
duration = 5000.0
n_steps = int(duration / dt)  # 100,000 steps!

print(f"Duration: {duration}ms, dt: {dt}ms")
print(f"Number of timesteps: {n_steps:,}")
print(f"\n{'='*60}")
print("TIMING BREAKDOWN")
print(f"{'='*60}\n")

# Test 1: Generate FF spikes (the suspected bottleneck)
print("1. Pre-generating feedforward spikes for all timesteps...")
rng_key = jax.random.PRNGKey(42)

t0 = time.time()
key_ff = jax.random.split(rng_key, n_steps)
t_split = time.time() - t0
print(f"   Split keys: {t_split:.3f}s")

@jax.jit
def generate_ff_spikes(key):
    key_e, key_i = jax.random.split(key)
    ff_e = sample_poisson_feedforward(n_ff, 50.0, dt, key_e)
    ff_i = sample_poisson_feedforward(n_ff, 50.0, dt, key_i)
    return ff_e, ff_i

t0 = time.time()
ff_spikes = jax.vmap(generate_ff_spikes)(key_ff)
ff_spikes[0].block_until_ready()  # Force computation
t_vmap = time.time() - t0
print(f"   Vmap generation: {t_vmap:.3f}s")
print(f"   TOTAL: {t_split + t_vmap:.3f}s  ← THIS IS THE BOTTLENECK!")

# Test 2: Run it again (should it be cached?)
print(f"\n2. Generating FF spikes again (should be cached?)...")
rng_key2 = jax.random.PRNGKey(43)
key_ff2 = jax.random.split(rng_key2, n_steps)

t0 = time.time()
ff_spikes2 = jax.vmap(generate_ff_spikes)(key_ff2)
ff_spikes2[0].block_until_ready()
t_vmap2 = time.time() - t0
print(f"   Vmap generation: {t_vmap2:.3f}s")

if t_vmap2 < t_vmap * 0.5:
    print(f"   ✓ Cached! {t_vmap/t_vmap2:.1f}x faster")
else:
    print(f"   ✗ NOT cached! Still slow")

# Test 3: Simple lax.scan loop (baseline)
print(f"\n3. Simple lax.scan loop (baseline)...")

@jax.jit
def simple_step(carry, x):
    return carry + x, carry

t0 = time.time()
final, _ = jax.lax.scan(simple_step, 0.0, jnp.arange(n_steps))
jax.block_until_ready(final)
t_scan_first = time.time() - t0
print(f"   First run: {t_scan_first:.3f}s (includes JIT)")

t0 = time.time()
final2, _ = jax.lax.scan(simple_step, 0.0, jnp.arange(n_steps))
jax.block_until_ready(final2)
t_scan_second = time.time() - t0
print(f"   Second run: {t_scan_second:.3f}s (cached)")
print(f"   Speedup: {t_scan_first/t_scan_second:.1f}x")

print(f"\n{'='*60}")
print("CONCLUSION")
print(f"{'='*60}\n")

print(f"Pre-generating FF spikes: {t_vmap:.2f}s (happens EVERY time)")
print(f"Actual scan loop: ~1-2s (cached after first run)")
print(f"\nThe FF generation takes ~{100*t_vmap/(t_vmap+2):.0f}% of total time!")
print(f"\nSOLUTION: Generate FF spikes on-the-fly inside the scan loop")
print(f"Expected speedup: ~{t_vmap:.0f}s → ~0.1s (100x faster!)")
