"""Neural models for SNOPS."""

from snops_jax.models.eif import EIFParams, eif_derivative
from snops_jax.models.connectivity import build_cbn, build_sbn
from snops_jax.models.synapses import SynapticParams, update_synapses

__all__ = [
    "EIFParams",
    "eif_derivative",
    "SynapticParams",
    "update_synapses",
    "build_cbn",
    "build_sbn",
]
