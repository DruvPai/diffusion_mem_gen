from typing import List
import jax
from jax import Array, nn, numpy as jnp
import equinox as eqx

from diffusion_mem_gen.models.layers.adaln import AdaLNZero
from diffusion_mem_gen.models.layers.mlp import MLP

class MultiHeadAttention(eqx.Module):
    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    qkv_proj: eqx.nn.Linear = eqx.field(static=False)
    out_proj: eqx.nn.Linear = eqx.field(static=False)

    def __init__(self, key: Array, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.dim: int = dim
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim

        key, subkey = jax.random.split(key)
        self.qkv_proj: eqx.nn.Linear = eqx.nn.Linear(dim, 3 * num_heads * head_dim, key=subkey)

        key, subkey = jax.random.split(key)
        self.out_proj: eqx.nn.Linear = eqx.nn.Linear(num_heads * head_dim, dim, key=subkey)

    def __call__(self, x: Array) -> Array:  # (seq_len, dim)
        seq_len, dim = x.shape
        qkv = jax.vmap(self.qkv_proj)(x)  # (seq_len, 3 * num_heads * head_dim)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim).transpose(
            0, 1
        )  # (3, seq_len, num_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (seq_len, num_heads, head_dim)
        attn_output_per_head = jax.nn.dot_product_attention(
            q, k, v
        )  # (seq_len, num_heads, head_dim)
        attn_output_stacked = attn_output_per_head.reshape(
            seq_len, self.num_heads * self.head_dim
        )  # (seq_len, num_heads * head_dim)
        attn_output = jax.vmap(self.out_proj)(attn_output_stacked)  # (seq_len, dim)
        return attn_output


class DiTBlock(eqx.Module):
    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    mlp_dim: int = eqx.field(static=True)
    adaln_mlp_dim: int = eqx.field(static=True)
    adaln_num_layers: int = eqx.field(static=True)

    attn: MultiHeadAttention = eqx.field(static=False)
    mlp: MLP = eqx.field(static=False)
    ln1: AdaLNZero = eqx.field(static=False)
    ln2: AdaLNZero = eqx.field(static=False)
    post_attn_scale: MLP = eqx.field(static=False)
    post_mlp_scale: MLP = eqx.field(static=False)

    def __init__(
        self,
        key: Array,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        adaln_mlp_dim: int,
        adaln_num_layers: int,
    ):
        super().__init__()
        self.dim: int = dim
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.mlp_dim: int = mlp_dim
        self.adaln_mlp_dim: int = adaln_mlp_dim
        self.adaln_num_layers: int = adaln_num_layers

        key, subkey = jax.random.split(key)
        self.ln1: AdaLNZero = AdaLNZero(key, dim, 1, adaln_mlp_dim, adaln_num_layers)
        key, subkey = jax.random.split(key)
        self.ln2: AdaLNZero = AdaLNZero(key, dim, 1, adaln_mlp_dim, adaln_num_layers)
        key, subkey = jax.random.split(key)
        self.attn: MultiHeadAttention = MultiHeadAttention(key, dim, num_heads, head_dim)
        key, subkey = jax.random.split(key)
        self.mlp: MLP = MLP(key, dim, dim, mlp_dim, 2)
        key, subkey = jax.random.split(key)
        self.post_attn_scale: MLP = MLP(
            key, 1, dim, adaln_mlp_dim, adaln_num_layers
        )
        key, subkey = jax.random.split(key)
        self.post_mlp_scale: MLP = MLP(
            key, 1, dim, adaln_mlp_dim, adaln_num_layers
        )

    def __call__(self, x: Array, t: Array) -> Array:  # (seq_len, dim)
        x = x + self.post_attn_scale(t[None]) * self.attn(self.ln1(x, t))
        x = x + self.post_mlp_scale(t[None]) * self.mlp(self.ln2(x, t))
        return x


class DiT(eqx.Module):
    max_seq_len: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    mlp_dim: int = eqx.field(static=True)
    adaln_mlp_dim: int = eqx.field(static=True)
    adaln_num_layers: int = eqx.field(static=True)

    layers: List[DiTBlock] = eqx.field(static=False)
    pos_emb: Array = eqx.field(static=False)

    def __init__(
        self,
        key: Array,
        max_seq_len: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        adaln_mlp_dim: int,
        adaln_num_layers: int,
    ):
        super().__init__()
        self.max_seq_len: int = max_seq_len
        self.num_layers: int = num_layers
        self.dim: int = dim
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.mlp_dim: int = mlp_dim
        self.adaln_mlp_dim: int = adaln_mlp_dim
        self.adaln_num_layers: int = adaln_num_layers

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num_layers)
        self.layers: List[DiTBlock] = [
            DiTBlock(subkeys[i], dim, num_heads, head_dim, mlp_dim, adaln_mlp_dim, adaln_num_layers)
            for i in range(num_layers)
        ]
        key, subkey = jax.random.split(key)
        self.pos_emb: Array = jax.random.normal(subkey, (max_seq_len, dim))

    def __call__(self, x_t: Array, t: Array) -> Array:  # (seq_len, dim)
        seq_len = x_t.shape[0]
        x = x_t + self.pos_emb[:seq_len]
        for layer in self.layers:
            x = layer(x, t)  # (seq_len, dim)
        return x
