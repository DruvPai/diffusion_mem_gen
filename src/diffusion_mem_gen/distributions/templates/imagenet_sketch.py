from typing import List, Tuple, cast
from jax import Array, numpy as jnp
from datasets import load_dataset, Dataset
import jax
from PIL import Image

def generate_imagenet_sketch_templates(
    num_templates: int,
    template_shape: Tuple[int, int],
    flatten_templates: bool,
) -> Array:
    """
    Generate a set of ImageNet Sketch templates.

    Args:
        num_templates (``int``): The number of templates to generate.
        template_shape (``Tuple[int, int]``): The size of the templates.
        flatten_templates (``bool``): Whether to flatten the templates.

    Returns:
        ``Array[num_templates, *template_dims]``: A set of ImageNet Sketch templates.
    """
    seed = 5
    key = jax.random.PRNGKey(seed)
    dataset = cast(Dataset, load_dataset("songweig/imagenet_sketch", split="train", trust_remote_code=True))
    indices = jax.random.choice(key, jnp.arange(len(dataset)), (num_templates,), replace=False)
    templates = cast(List[Image.Image], dataset[indices]["image"])
    templates = list(map(lambda x: preprocess_imagenet_sketch_template(x, template_shape), templates))
    templates = jnp.stack(templates)
    if flatten_templates:
        templates = jax.vmap(lambda x: x.flatten())(templates)
    return templates

def preprocess_imagenet_sketch_template(template: Image.Image, template_shape: Tuple[int, int]) -> Array:
    template = template.convert("L").resize(template_shape, Image.Resampling.BICUBIC)
    template_arr = jnp.array(template)
    template_arr = 1 - template_arr / 255.0
    return template_arr
