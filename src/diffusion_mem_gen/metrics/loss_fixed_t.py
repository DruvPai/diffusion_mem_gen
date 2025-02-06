from typing import Any, Callable, Dict, cast
import jax 
from jax import Array, numpy as jnp
import equinox as eqx
from diffusionlab.losses import DiffusionLoss


class LossEachT(eqx.Module):
    ts: Array = eqx.field(static=True)
    loss: DiffusionLoss = eqx.field(static=True)

    def __init__(self, ts: Array, loss: DiffusionLoss):
        super().__init__()
        assert ts.ndim == 1
        self.ts: Array = ts
        self.loss: DiffusionLoss = loss

    def __call__(self, key: Array, net: eqx.Module, x: Array, y: Array) -> Dict[str, Array]:
        losses = {}
        batch_size = x.shape[0]
        for i in range(self.ts.shape[0]):
            t = self.ts[i]
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, batch_size)
            loss = jax.vmap(self.loss.loss, in_axes=(0, None, 0, None))(subkeys, cast(Callable, net), x, t)
            losses[f"t={t:.3f}"] = jnp.mean(loss)
        return losses
