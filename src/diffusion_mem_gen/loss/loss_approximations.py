from typing import Callable
from jax import Array, numpy as jnp
import jax
from jax.scipy.special import gammaln
from diffusionlab.vector_fields import VectorFieldType
from diffusionlab.dynamics import DiffusionProcess
from diffusion_mem_gen.loss.loss_scaling import loss_scaling_factor_factory


def iso_hom_gmm_gen_vf_excess_train_loss_compared_to_mem_vf_approx_factory(
    dim: int,
    diffusion_process: DiffusionProcess,
    vector_field_type: VectorFieldType,
    true_var: Array,  # (,)
    loss_weight_function: Callable[[Array], Array],
) -> Callable[[Array], Array]:
    loss_scaling_factor = loss_scaling_factor_factory(
        diffusion_process, VectorFieldType.X0, vector_field_type
    )

    def compute_loss_approx(t: Array) -> Array:
        alpha = diffusion_process.alpha(t)
        sigma = diffusion_process.sigma(t)
        weight = loss_scaling_factor(t) * loss_weight_function(t)
        return weight * dim * true_var / (1 + ((alpha / sigma) ** 2) * true_var)

    return compute_loss_approx


def iso_hom_gmm_pmem_vf_excess_train_loss_compared_to_mem_vf_approx_factory(
    dim: int,
    diffusion_process: DiffusionProcess,
    vector_field_type: VectorFieldType,
    true_var: Array,  # (,)
    loss_weight_function: Callable[[Array], Array],
) -> Callable[[Array, int, int], Array]:
    loss_scaling_factor = loss_scaling_factor_factory(
        diffusion_process, VectorFieldType.X0, vector_field_type
    )

    def compute_loss_approx(t: Array, num_components_model: int, num_samples: int) -> Array:
        weight = loss_scaling_factor(t) * loss_weight_function(t)
        loss = dim * true_var * (1 - num_components_model / num_samples)
        return weight * loss

    return compute_loss_approx
