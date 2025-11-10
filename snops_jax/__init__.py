"""
SNOPS-JAX: Spiking Network Optimization using Population Statistics
JAX/GPU implementation of SNOPS for efficient large-scale spiking network optimization.
"""

__version__ = "0.1.0"

from snops_jax import models, simulate, stats, bo

__all__ = ["models", "simulate", "stats", "bo", "__version__"]
