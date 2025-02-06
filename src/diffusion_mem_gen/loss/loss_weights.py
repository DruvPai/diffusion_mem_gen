from typing import Callable
from jax import Array
from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.vector_fields import VectorFieldType
from diffusion_mem_gen.loss.loss_scaling import loss_scaling_factor_factory

def elbo_loss_weight(diffusion: DiffusionProcess, prediction_vector_field_type: VectorFieldType) -> Callable[[Array], Array]:
    loss_scaling_factor = loss_scaling_factor_factory(diffusion, VectorFieldType.EPS, prediction_vector_field_type)
    def loss_weight(t: Array) -> Array:
        alpha = diffusion.alpha(t)
        sigma = diffusion.sigma(t)
        alpha_prime = diffusion.alpha_prime(t)
        sigma_prime = diffusion.sigma_prime(t)
        scale = loss_scaling_factor(t)
        return scale * 2/(sigma_prime/sigma - alpha_prime/alpha)
    return loss_weight

