from typing import List, Tuple, cast
from jax import Array, numpy as jnp
from datasets import load_dataset, Dataset
import jax
from PIL import Image

def generate_fashion_mnist_templates(
    num_templates: int,
    template_shape: Tuple[int, int],
    flatten_templates: bool,
) -> Array:
    """
    Generate a set of MNIST templates.

    Args:
        num_templates (``int``): The number of templates to generate.
        template_shape (``Tuple[int, int]``): The size of the templates.
        flatten_templates (``bool``): Whether to flatten the templates.

    Returns:
        ``Array[num_templates, *template_dims]``: A set of MNIST templates.
    """
    seed = 1
    key = jax.random.PRNGKey(seed)
    dataset = cast(Dataset, load_dataset("zalando-datasets/fashion_mnist", split="train"))
    indices = jax.random.choice(key, jnp.arange(len(dataset)), (num_templates,), replace=False)
    templates = cast(List[Image.Image], dataset[indices]["image"])
    templates = list(map(lambda x: jnp.array(x.resize(template_shape, Image.Resampling.BICUBIC)), templates))
    templates = jnp.stack(templates)
    if flatten_templates:
        templates = jax.vmap(lambda x: x.flatten())(templates)
    templates = (templates - templates.min()) / (templates.max() - templates.min())
    return templates

