from typing import Callable, Tuple
from jax import Array, numpy as jnp
from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.vector_fields import VectorFieldType


def _get_scales(diffusion_process: DiffusionProcess, t: Array) -> Tuple[Array, Array, Array, Array]:
    alpha = diffusion_process.alpha(t)
    sigma = diffusion_process.sigma(t)
    alpha_prime = diffusion_process.alpha_prime(t)
    sigma_prime = diffusion_process.sigma_prime(t)
    return alpha, sigma, alpha_prime, sigma_prime


def loss_scaling_factor_factory(
    diffusion_process: DiffusionProcess, from_vf_type: VectorFieldType, to_vf_type: VectorFieldType
) -> Callable[[Array], Array]:
    if from_vf_type == VectorFieldType.X0:
        if to_vf_type == VectorFieldType.X0:

            def loss_scaling_factor(t: Array) -> Array:
                return jnp.ones_like(t)

        elif to_vf_type == VectorFieldType.EPS:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return (alpha / sigma) ** 2

        elif to_vf_type == VectorFieldType.V:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return alpha**2 * (sigma_prime / sigma - alpha_prime / alpha) ** 2

        elif to_vf_type == VectorFieldType.SCORE:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return alpha**2 / sigma**4

        else:
            raise ValueError(f"Invalid vector field type: {to_vf_type}")

    elif from_vf_type == VectorFieldType.EPS:
        if to_vf_type == VectorFieldType.X0:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return (sigma / alpha) ** 2

        elif to_vf_type == VectorFieldType.EPS:

            def loss_scaling_factor(t: Array) -> Array:
                return jnp.ones_like(t)

        elif to_vf_type == VectorFieldType.V:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return sigma**2 * (sigma_prime / sigma - alpha_prime / alpha) ** 2

        elif to_vf_type == VectorFieldType.SCORE:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return 1 / sigma**2

        else:
            raise ValueError(f"Invalid vector field type: {to_vf_type}")

    elif from_vf_type == VectorFieldType.V:
        if to_vf_type == VectorFieldType.X0:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return 1 / (alpha**2 * (sigma_prime / sigma - alpha_prime / alpha) ** 2)

        elif to_vf_type == VectorFieldType.EPS:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return 1 / (sigma**2 * (sigma_prime / sigma - alpha_prime / alpha) ** 2)

        elif to_vf_type == VectorFieldType.V:

            def loss_scaling_factor(t: Array) -> Array:
                return jnp.ones_like(t)

        elif to_vf_type == VectorFieldType.SCORE:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return 1 / (sigma**4 * (sigma_prime / sigma - alpha_prime / alpha) ** 2)

        else:
            raise ValueError(f"Invalid vector field type: {to_vf_type}")

    elif from_vf_type == VectorFieldType.SCORE:
        if to_vf_type == VectorFieldType.X0:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return sigma**4 / alpha**2

        elif to_vf_type == VectorFieldType.EPS:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return sigma**2

        elif to_vf_type == VectorFieldType.V:

            def loss_scaling_factor(t: Array) -> Array:
                alpha, sigma, alpha_prime, sigma_prime = _get_scales(diffusion_process, t)
                return sigma**4 * (sigma_prime / sigma - alpha_prime / alpha) ** 2

        elif to_vf_type == VectorFieldType.SCORE:

            def loss_scaling_factor(t: Array) -> Array:
                return jnp.ones_like(t)

        else:
            raise ValueError(f"Invalid vector field type: {to_vf_type}")
    else:
        raise ValueError(f"Invalid vector field type: {from_vf_type}")

    return loss_scaling_factor
