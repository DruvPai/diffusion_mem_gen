from curses import color_pair
from typing import Dict, List, Tuple, Callable, cast
import math
from jax import Array, numpy as jnp
import jax
import numpy as np
import equinox as eqx
from PIL import Image
from diffusionlab.samplers import Sampler
import wandb


class VisualizeImage(eqx.Module):
    num_samples_to_visualize: int = eqx.field(static=True)
    img_shape: Tuple[int, int] = eqx.field(static=True)
    flatten_img: bool = eqx.field(static=True)
    color_dim: int = eqx.field(static=True)
    ts: Array = eqx.field(static=True)
    sampler_closure: Callable[[Callable[[Array, Array], Array]], Sampler] = eqx.field(static=True)

    def __init__(
            self, 
            num_samples_to_visualize: int, 
            img_shape: Tuple[int, int], 
            flatten_img: bool,
            color_dim: int,
            ts: Array , 
            sampler_closure: Callable[[Callable[[Array, Array], Array]], Sampler],
    ):
        super().__init__()
        self.num_samples_to_visualize: int = num_samples_to_visualize
        self.img_shape: Tuple[int, int] = img_shape
        self.flatten_img: bool = flatten_img
        self.color_dim: int = color_dim
        self.ts: Array = ts
        self.sampler_closure: Callable[[Callable[[Array, Array], Array]], Sampler] = sampler_closure

    
    def __call__(self, key: Array, net: eqx.Module) -> Dict[str, List[wandb.Image]]:
        num_times = self.ts.shape[0]
        data_dims = (self.color_dim, math.prod(self.img_shape), ) if self.flatten_img else (self.color_dim, *self.img_shape)
        key, subkey = jax.random.split(key)
        x_init = jax.random.normal(key, (self.num_samples_to_visualize, *data_dims))
        key, subkey = jax.random.split(key)
        zs = jax.random.normal(key, (self.num_samples_to_visualize, num_times-1, *data_dims))

        sampler = self.sampler_closure(cast(Callable[[Array, Array], Array], net))

        X_sample = jax.vmap(sampler.sample, in_axes=(0, 0, None))(x_init, zs, self.ts)  # (num_samples_to_visualize, *data_shape)

        X_sample = X_sample.reshape((self.num_samples_to_visualize, self.color_dim, *self.img_shape)).transpose(0, 2, 3, 1)
        X_sample = jnp.round((X_sample - X_sample.min(axis=(1, 2, 3), keepdims=True)) / (X_sample.max(axis=(1, 2, 3), keepdims=True) - X_sample.min(axis=(1, 2, 3), keepdims=True)) * 255)
        X_sample_img = [wandb.Image(Image.fromarray(np.asarray(x.astype(jnp.uint8)))) for x in X_sample]

        return {"": X_sample_img}