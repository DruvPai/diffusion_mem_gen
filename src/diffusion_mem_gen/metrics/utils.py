from typing import Tuple
from jax import Array, numpy as jnp
import jax

def closest_k_points_in_dataset_indices(
    x: Array,  # (*data_shape)
    x_dataset: Array,  # (num_samples_dataset, *data_shape)
    k: int = 2
) -> Tuple[Array, Array]:
    assert x_dataset.shape[0] >= k
    distances = jax.vmap(lambda b: jnp.sum((x - b) ** 2) ** (1/2))(x_dataset)  # (num_samples_dataset,)
    neg_min_distances, min_indices = jax.lax.top_k(-distances, k=k)  # (num_samples, k)
    min_distances = -neg_min_distances  # (num_samples, k)
    return min_distances, min_indices
