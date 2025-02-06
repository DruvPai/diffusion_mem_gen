from typing import List, Tuple, cast
from jax import Array, numpy as jnp
from datasets import load_dataset, Dataset
import jax
from PIL import Image

def generate_mnist_templates(
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
    mnist_dataset = cast(Dataset, load_dataset("ylecun/mnist", split="train"))
    templates = cast(List[Image.Image], mnist_dataset[:num_templates]["image"])
    templates = map(lambda x: jnp.array(x.resize(template_shape, Image.Resampling.BICUBIC)), templates)
    templates = jnp.stack(list(templates))
    if flatten_templates:
        templates = jax.vmap(lambda x: x.flatten())(templates)
    templates = templates / 255.0
    return templates

