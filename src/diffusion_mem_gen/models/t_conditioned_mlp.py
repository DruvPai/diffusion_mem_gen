from typing import List
import jax 
from jax import Array, nn, numpy as jnp 
import equinox as eqx

from diffusion_mem_gen.models.layers.adaln import AdaLNZero


class TimeConditionedMLP(eqx.Module):
    dim: int = eqx.field(static=True)
    mlp_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    adaln_mlp_dim: int = eqx.field(static=True)
    adaln_num_layers: int = eqx.field(static=True)

    in_layer: eqx.nn.Linear = eqx.field(static=False)
    out_layer: eqx.nn.Linear = eqx.field(static=False)
    hidden_layers: List[eqx.nn.Linear] = eqx.field(static=False)
    total_residual_scale: Array = eqx.field(static=False)
    hidden_lns: List[AdaLNZero] = eqx.field(static=False)
    hidden_residual_scales_postln: List[Array] = eqx.field(static=False)
    hidden_residual_scales_postrelu: List[Array] = eqx.field(static=False)
    out_ln: AdaLNZero = eqx.field(static=False)
    out_residual_scale_postln: Array = eqx.field(static=False)
    
    def __init__(self, key: jax.Array, dim: int, mlp_dim: int, num_layers: int, adaln_mlp_dim: int, adaln_num_layers: int):
        super().__init__()
        self.dim: int = dim
        self.mlp_dim: int = mlp_dim
        self.num_layers: int = num_layers
        self.adaln_mlp_dim: int = adaln_mlp_dim
        self.adaln_num_layers: int = adaln_num_layers

        num_hidden_layers = num_layers - 2
        
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num_hidden_layers)
        self.hidden_lns: List[AdaLNZero] = [
            AdaLNZero(subkeys[i], mlp_dim, 1, adaln_mlp_dim, adaln_num_layers)
            for i in range(num_hidden_layers)
        ]
        
        key, subkey = jax.random.split(key)
        self.in_layer = eqx.nn.Linear(dim, mlp_dim, key=subkey)
        
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num_hidden_layers)
        self.hidden_layers: List[eqx.nn.Linear] = [
            eqx.nn.Linear(mlp_dim, mlp_dim, key=subkeys[i])
            for i in range(num_hidden_layers)
        ]
        
        key, subkey = jax.random.split(key)
        self.out_layer = eqx.nn.Linear(mlp_dim, dim, key=subkey)
        
        key, subkey = jax.random.split(key)
        self.out_ln = AdaLNZero(key, mlp_dim, 1, adaln_mlp_dim, adaln_num_layers)

        self.total_residual_scale: Array = jnp.ones((dim,))
        self.hidden_residual_scales_postln: List[Array] = [jnp.ones((mlp_dim,)) for _ in range(num_hidden_layers)]
        self.hidden_residual_scales_postrelu: List[Array] = [jnp.ones((mlp_dim,)) for _ in range(num_hidden_layers)]
        self.out_residual_scale_postln: Array = jnp.ones((mlp_dim,))


    def __call__(self, x_t: Array, t: Array) -> Array:
        x = nn.relu(self.in_layer(x_t))
        for i in range(self.num_layers - 2):
            x = self.hidden_residual_scales_postln[i] * x + self.hidden_lns[i](x, t[None])
            x = self.hidden_residual_scales_postrelu[i] * x + nn.relu(self.hidden_layers[i](x))
        x = self.out_residual_scale_postln * x + self.out_ln(x, t[None])
        x = self.total_residual_scale * x_t  + self.out_layer(x)
        return x
