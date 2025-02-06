

from typing import List

import jax 
from jax import Array
import equinox as eqx


class MLP(eqx.Module):
    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    
    in_layer: eqx.nn.Linear = eqx.field(static=False)
    hidden_layers: List[eqx.nn.Linear] = eqx.field(static=False)
    out_layer: eqx.nn.Linear = eqx.field(static=False)

    def __init__(self, key: Array, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers

        key, subkey = jax.random.split(key)
        self.in_layer = eqx.nn.Linear(in_dim, hidden_dim, key=subkey)
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num_layers - 2)
        self.hidden_layers = [eqx.nn.Linear(hidden_dim, hidden_dim, key=subkey) for subkey in subkeys]
        key, subkey = jax.random.split(key)
        self.out_layer = eqx.nn.Linear(hidden_dim, out_dim, key=subkey)
        
    def __call__(self, x: Array) -> Array:
        x = self.in_layer(x)
        for hidden_layer in self.hidden_layers:
            x = jax.nn.relu(x)
            x = hidden_layer(x)
        x = jax.nn.relu(x)
        x = self.out_layer(x)
        return x
        