from typing import Callable, Dict, cast
import jax 
from jax import Array, numpy as jnp
import equinox as eqx
from diffusionlab.samplers import Sampler

from diffusion_mem_gen.metrics.utils import closest_k_points_in_dataset_indices

class RatioOfDistancesMetric(eqx.Module):
    num_samples_to_evaluate: int = eqx.field(static=True)
    X_train: Array = eqx.field(static=True)
    ts: Array = eqx.field(static=True)
    sampler_closure: Callable[[Callable[[Array, Array], Array]], Sampler] = eqx.field(static=True)
    threshold: float = eqx.field(static=True)

    def __init__(
            self, 
            num_samples_to_evaluate: int, 
            X_train: Array, 
            ts: Array , 
            sampler_closure: Callable[[Callable[[Array, Array], Array]], Sampler],
            threshold: float = 1/3
    ):
        super().__init__()
        self.num_samples_to_evaluate: int = num_samples_to_evaluate
        self.X_train: Array = X_train
        self.ts: Array = ts
        self.sampler_closure: Callable[[Callable[[Array, Array], Array]], Sampler] = sampler_closure
        self.threshold: float = threshold

    def __call__(self, key: Array, net: eqx.Module) -> Dict[str, Array]:
        num_times = self.ts.shape[0]
        data_dims = self.X_train.shape[1:]
        key, subkey = jax.random.split(key)
        x_init = jax.random.normal(key, (self.num_samples_to_evaluate, *data_dims))
        key, subkey = jax.random.split(key)
        zs = jax.random.normal(key, (self.num_samples_to_evaluate, num_times-1, *data_dims))

        sampler = self.sampler_closure(cast(Callable[[Array, Array], Array], net))

        X_sample = jax.vmap(sampler.sample, in_axes=(0, 0, None))(x_init, zs, self.ts)  # (num_samples_to_evaluate, *data_shape)

        min_distances_to_samples, _ = jax.vmap(lambda x: closest_k_points_in_dataset_indices(x, self.X_train, k=2))(X_sample)
        memorized_samples = min_distances_to_samples[:, 0] < self.threshold * min_distances_to_samples[:, 1]  # (num_samples_to_evaluate,)
        return {"": jnp.mean(memorized_samples)}