from typing import Iterable, Tuple
from jax import Array, numpy as jnp


def batch_data(xs: Tuple[Array, ...], batch_size: int) -> Iterable[Tuple[Array, ...]]:
    num_samples = xs[0].shape[0]
    assert all(x.shape[0] == num_samples for x in xs)
    if batch_size > num_samples:
        batches = [xs]
    else:
        assert num_samples % batch_size == 0
        overall_batches = []
        for x in xs:
            individual_batches = jnp.split(x, num_samples // batch_size, axis=0)
            overall_batches.append(individual_batches)
        batches = list(zip(*overall_batches))
    return batches
