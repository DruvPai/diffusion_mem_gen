import jax
from jax import Array, numpy as jnp 
import equinox as eqx

from diffusion_mem_gen.models.layers.mlp import MLP

class AdaLNZero(eqx.Module):
    dim: int = eqx.field(static=True)
    condition_dim: int = eqx.field(static=True)
    mlp_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)

    mlp: MLP = eqx.field(static=False)
    ln: eqx.nn.LayerNorm = eqx.field(static=False)

    def __init__(self, key: Array, dim: int, condition_dim: int, mlp_dim: int, num_layers: int):
        super().__init__()
        self.dim: int = dim
        self.condition_dim: int = condition_dim
        self.mlp_dim: int = mlp_dim
        self.num_layers: int = num_layers

        key, subkey = jax.random.split(key)
        self.mlp = MLP(
            key,
            condition_dim, 
            2 * dim, 
            mlp_dim, 
            num_layers, 
        )
        key, subkey = jax.random.split(key)
        self.ln = eqx.nn.LayerNorm(dim)

    def __call__(self, x: Array, condition: Array) -> Array:  # (dim,), (condition_dim,) -> (dim,)
        scale, shift = jnp.split(self.mlp(condition), 2, axis=-1)  # (dim,), (dim,)
        return scale * self.ln(x) + shift
