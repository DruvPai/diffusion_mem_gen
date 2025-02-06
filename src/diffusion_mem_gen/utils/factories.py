from typing import Any, Callable, Dict, cast

import jax
from jax import Array, numpy as jnp
from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.losses import DiffusionLoss


def inject_diffusion_process_to_vf(
    vf: Callable[[Array, Array, DiffusionProcess], Array], diffusion_process: DiffusionProcess
) -> Callable[[Array, Array], Array]:
    """Injects the diffusion process into a vector field function."""
    return lambda x, t: vf(x, t, diffusion_process)


def compute_loss_factory(loss_fn: DiffusionLoss, t_val: Array) -> Callable[[Array, Callable, Array], Array]:
    """
    Creates a function to compute the loss for a given model, data, and time.

    Args:
        loss_fn_obj: The diffusion loss object.
        t_val: The specific time step (scalar array) for which to compute the loss.

    Returns:
        A function that takes a JAX PRNG key, a vector field model (Callable),
        and data (Array), and returns the mean loss.
    """
    def compute_loss(key: Array, vf_model: Callable, x_data: Array) -> Array:
        """Computes the mean loss for the given model, data, and pre-specified time t_val."""
        subkeys = jax.random.split(key, x_data.shape[0])
        return jnp.mean(
            jax.vmap(loss_fn.loss, in_axes=(0, None, 0, None))(
                subkeys, vf_model, x_data, t_val
            )
        )
    return compute_loss 