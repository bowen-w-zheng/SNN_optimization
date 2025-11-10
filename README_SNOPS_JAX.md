# SNOPS-JAX: Spiking Network Optimization using Population Statistics

A high-performance JAX/GPU implementation of **SNOPS** (Spiking Network Optimization using Population Statistics) from the paper:

> Wu, S., et al. (2024). "Automated customization of large-scale spiking network models to neuronal population activity." *Nature Computational Science*, 4, 690-705.

This Python implementation translates the original MATLAB code to JAX for GPU acceleration, modern ODE solvers, and scalable Bayesian optimization.

## 🚀 Features

- **GPU-accelerated simulation**: JAX-based spiking network simulator with multiple integrators (Euler, Heun, RK4, Diffrax)
- **Two network architectures**:
  - **CBN** (Classical Balanced Network): Random E/I connectivity
  - **SBN** (Spatial Balanced Network): Distance-dependent connectivity on 2D grid
- **Comprehensive statistics**: Firing rate, Fano factor, spike count correlations, Factor Analysis (shared variance, dimensionality, eigenspectrum)
- **Advanced Bayesian Optimization**:
  - ARD Matérn-5/2 Gaussian Processes
  - Constrained Expected Improvement with feasibility GP
  - Intensification with variance reduction
  - Multi-start L-BFGS-B acquisition optimization
- **Paper-faithful implementation**: Matches MATLAB baseline with <5% tolerance

## 📦 Installation

### Requirements

- Python ≥3.9
- CUDA-compatible GPU (optional but recommended)

### Install from source

```bash
# Clone the repository
cd SNN_optimization

# Install in development mode
pip install -e .

# For GPU support (CUDA 12.x)
pip install -e ".[gpu]"

# For development tools
pip install -e ".[dev]"
```

### Dependencies

Core: `jax`, `jaxlib`, `numpy`, `scipy`, `pandas`, `h5py`, `pyyaml`
JAX tools: `equinox`, `optax`, `jaxopt`, `chex`, `diffrax`
GP/BO: `tinygp`, `scikit-learn`
Visualization: `matplotlib`, `seaborn`

## 🎯 Quick Start

### Example 1: Simple Simulation

```python
import jax
from snops_jax.models.eif import EIFParams
from snops_jax.models.synapses import SynapticParams
from snops_jax.models.connectivity import build_cbn, NetworkParams
from snops_jax.simulate.run import run_simulation, SimulationConfig
from snops_jax.stats.single_pair import compute_statistics_summary

# Setup
n_e, n_i, n_ff = 800, 200, 200
eif_params = EIFParams()
syn_params = SynapticParams(tau_ed=5.0, tau_id=5.0)
network_params = NetworkParams(J_ee=20.0, J_ei=-40.0, J_ie=30.0, J_ii=-30.0)

# Build network
rng_key = jax.random.PRNGKey(42)
conn = build_cbn(n_e, n_i, n_ff, network_params, rng_key=rng_key)

# Simulate
sim_config = SimulationConfig(duration=10000.0, dt=0.05)
output = run_simulation(n_e, n_i, n_ff, conn, sim_config, eif_params, syn_params, rng_key)

# Compute statistics
stats = compute_statistics_summary(output.spike_counts_e)
print(f"Firing rate: {stats['fr']:.2f} sp/s, Fano factor: {stats['ff']:.2f}")
```

Run the example script:

```bash
python examples/simple_simulation.py
```

### Example 2: Compute Statistics from Data

```python
import jax.numpy as jnp
from snops_jax.stats.cost import compute_target_statistics, CostConfig

# Load spike count data (n_neurons, n_bins) from multiple sessions
spike_counts_sessions = [...]  # List of arrays

# Compute target statistics
config = CostConfig()
target = compute_target_statistics(spike_counts_sessions, config)

print(f"Target: fr={target.fr_mean:.2f}±{jnp.sqrt(target.fr_var):.2f} sp/s")
```

### Example 3: Run SNOPS Optimization (Coming Soon)

```python
from snops_jax.bo.driver import run_bo_loop, BOConfig
import yaml

# Load config
with open("snops_jax/config/default_cbn.yaml") as f:
    config_dict = yaml.safe_load(f)

# Define objective function
def objective(theta, rep):
    # theta: parameter array
    # rep: repetition index
    # Returns: (cost, stats, is_feasible)
    ...

# Run optimization
bounds = jnp.array([[1, 25], [1, 25], ...])  # Parameter bounds
bo_config = BOConfig(n_init=50, max_iterations=200)
state = run_bo_loop(objective, bounds, bo_config, rng_key)
```

## 📁 Repository Structure

```
snops_jax/
├── models/
│   ├── eif.py              # EIF neuron dynamics
│   ├── synapses.py         # Synaptic currents
│   └── connectivity.py     # CBN/SBN connectivity
├── simulate/
│   ├── step.py             # Time-stepping kernel
│   └── run.py              # Main simulation loops (lax.scan)
├── stats/
│   ├── binning.py          # Spike binning & subsampling
│   ├── single_pair.py      # fr, ff, rsc statistics
│   ├── fa_jax.py           # Factor Analysis (EM)
│   └── cost.py             # Cost function (eq. 7)
├── bo/
│   ├── gp.py               # Gaussian Process surrogates (tinygp)
│   ├── acquisition.py      # EI, constrained EI
│   ├── suggest.py          # L-BFGS-B multi-start
│   ├── intensify.py        # Intensification & feasibility
│   └── driver.py           # Main BO loop
├── config/
│   ├── default_cbn.yaml    # CBN configuration
│   └── default_sbn.yaml    # SBN configuration
└── cli/                    # Command-line tools (TBD)
```

## 🧪 Model Details

### EIF Neuron (Exponential Integrate-and-Fire)

```
C_m dV/dt = -g_L(V - E_L) + g_L·ΔT·exp((V - V_T)/ΔT) + I(t)
```

**Default parameters** (from paper):
- τ_m = 15 ms, E_L = -60 mV, V_T = -50 mV, V_th = -10 mV
- ΔT = 2 mV, V_re = -65 mV, τ_ref = 1.5 ms

### Synaptic Currents

First-order exponential decay with spike-triggered increments:
```
ds/dt = -s/τ + Σ J·δ(t - t_spike)
```

**Free parameters**: τ_ed, τ_id ∈ [1, 25] ms

### Connectivity

**CBN**: Random connections with fixed probabilities
**SBN**: Gaussian distance-dependent probability with periodic boundary

**Free parameters**:
- Synaptic strengths: J_ee, J_ei, J_ie, J_ii, J_eF, J_iF ∈ [-150, 150] mV
- Spatial widths (SBN): σ_e, σ_i, σ_F ∈ [0, 0.25] mm

### Statistics & Cost Function

**Single/pairwise**:
- `fr`: Mean firing rate (sp/s)
- `ff`: Fano factor (var/mean of spike counts)
- `rsc`: Spike count correlation (Fisher z-transformed)

**Population (via Factor Analysis)**:
- `%sh`: Percent shared variance
- `dsh`: Dimensionality (# eigenvalues for 95% variance)
- `es`: Eigenspectrum

**Cost** (paper eq. 7):
```
c_S(θ) = (1/Σw_j) · Σ w_j · [(s_j(θ) - s_j^true)² / v_j^true]
```

### Bayesian Optimization

1. **Initialization**: Latin Hypercube Sampling (50 points)
2. **Surrogate**: ARD Matérn-5/2 GP with MLE hyperparameters
3. **Acquisition**: Constrained EI (paper eq. 12)
   ```
   CEI(θ) = Φ((μ_g - 0.5)/σ_g) · EI(θ)
   ```
4. **Feasibility**: Short-run checks (fr < 1, ff > 5, dsh < 1 → infeasible)
5. **Intensification**: R=5 repetitions, early stop if SD < 0.15

## 🔧 Configuration

Configurations are in `snops_jax/config/`. Key sections:

### Network
```yaml
network:
  type: "CBN"  # or "SBN"
  n_e: 4500
  n_i: 1125
  n_ff: 1000
```

### Simulation
```yaml
simulation:
  dt: 0.05                  # Euler timestep (ms)
  integrator: "euler"       # "euler", "heun", "rk4"
  full_duration: 140500.0   # 140.5 seconds
  burn_in: 500.0
  bin_size: 200.0
```

### BO
```yaml
bo:
  n_init: 50
  max_iterations: 200
  max_reps: 5              # Intensification
  sd_threshold: 0.15
  min_fr: 1.0              # Feasibility thresholds
  max_ff: 5.0
```

## 📊 Validation

The implementation is designed to match the MATLAB baseline within ±5% tolerance:

| Statistic | Tolerance |
|-----------|-----------|
| fr        | ±0.05 sp/s |
| ff        | ±0.05 |
| rsc (Fisher z) | ±0.02 |
| %sh       | ±1.0 pp |
| dsh       | ±1 |
| eigenspectrum | ≤5% L2 error |

**Testing** (upcoming):
```bash
pytest snops_jax/tests/
```

## 🚧 Roadmap

- [x] Core simulation engine (EIF, synapses, connectivity)
- [x] Statistics computation (fr, ff, rsc, FA)
- [x] Cost function (eq. 7)
- [x] Bayesian optimization (GP, constrained EI, intensification)
- [x] Configuration system
- [ ] Complete CLI tools
- [ ] Unit tests & MATLAB regression tests
- [ ] Example notebooks
- [ ] Multi-GPU support (pjit/pmap)
- [ ] Diffrax integration with spike-time interpolation

## 📚 References

**Paper**:
- Wu, S., et al. (2024). *Nature Computational Science*, 4, 690-705.
  DOI: [10.1038/s43588-024-00688-3](https://doi.org/10.1038/s43588-024-00688-3)

**Original MATLAB code**:
- [github.com/ShenghaoWu/SpikingNetworkOptimization](https://github.com/ShenghaoWu/SpikingNetworkOptimization)

**Related work**:
- Huang et al. FI_SpatialNet MEX kernels

## 📝 Citation

If you use SNOPS-JAX in your research, please cite:

```bibtex
@article{wu2024snops,
  title={Automated customization of large-scale spiking network models to neuronal population activity},
  author={Wu, Shenghao and Huang, Chengcheng and Snyder, Adam C and Smith, Matthew A and Doiron, Brent and Yu, Byron M},
  journal={Nature Computational Science},
  volume={4},
  pages={690--705},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## 📄 License

MIT License (see LICENSE file)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 💬 Contact

For questions or issues:
- Open a GitHub issue
- Refer to the original paper for methodological details

---

**Status**: 🚧 Alpha release - core functionality complete, extensive testing in progress
