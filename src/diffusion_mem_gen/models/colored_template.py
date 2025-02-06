from enum import Enum, auto
from typing import Any, Callable, Dict, Tuple
import jax
from jax import Array, numpy as jnp
import equinox as eqx

from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.vector_fields import VectorFieldType

from diffusion_mem_gen.distributions.colored_signal_template_data import (
    ColoredSignalTemplateDistribution,
    colored_signal_template_x0,
    colored_signal_template_eps,
    colored_signal_template_v,
    colored_signal_template_score,
)


def _get_correct_vf(
    vf_type: VectorFieldType,
) -> Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array]:
    match vf_type:
        case VectorFieldType.X0:
            vf = colored_signal_template_x0
        case VectorFieldType.EPS:
            vf = colored_signal_template_eps
        case VectorFieldType.V:
            vf = colored_signal_template_v
        case VectorFieldType.SCORE:
            vf = colored_signal_template_score
        case _:
            raise ValueError(f"Invalid vector field type: {vf_type}")
    return vf


def get_color_vectors(X: Array, y: Array, X_templates: Array) -> Array:
    color_vectors = jax.vmap(
        lambda x, x_template: jnp.sum(x, axis=tuple(range(1, x.ndim))) / jnp.sum(x_template)
    )(X, X_templates)  # (num_templates, color_dim)
    return color_vectors


class ColoredSignalTemplateInitStrategy(Enum):
    GAUSSIAN = auto()
    SPHERE = auto()
    PRIOR = auto()
    PMEM = auto()


def colored_signal_template_create_initialization_parameters(
    key: Array,
    init_strategy: ColoredSignalTemplateInitStrategy,
    img_shape: Tuple[int, int],
    flatten_img: bool,
    color_dim: int,
    num_templates: int,
    context: Dict[str, Any],
) -> Tuple[Array, Array, Array]:
    match init_strategy:
        case ColoredSignalTemplateInitStrategy.GAUSSIAN:
            assert "init_template_sparsity" in context and isinstance(
                context["init_template_sparsity"], float
            )
            assert "init_color_means_scale" in context and isinstance(
                context["init_color_means_scale"], float
            )
            assert "init_color_var_scale" in context and isinstance(
                context["init_color_var_scale"], float
            )
            templates_key, color_means_key = jax.random.split(key)
            templates = jax.random.bernoulli(
                templates_key, context["init_template_sparsity"], (num_templates, *img_shape)
            )
            if flatten_img:
                templates = templates.reshape(num_templates, -1)
            color_means = (
                jax.random.normal(color_means_key, (num_templates, color_dim))
                * context["init_color_means_scale"]
                / (color_dim ** (1 / 2))
            )
            color_var = jnp.array(context["init_color_var_scale"])

        case ColoredSignalTemplateInitStrategy.SPHERE:
            assert "init_template_sparsity" in context and isinstance(
                context["init_template_sparsity"], float
            )
            assert "init_color_means_scale" in context and isinstance(
                context["init_color_means_scale"], float
            )
            assert "init_color_var_scale" in context and isinstance(
                context["init_color_var_scale"], float
            )
            templates_key, color_means_key = jax.random.split(key)
            templates = jax.random.bernoulli(
                templates_key, context["init_template_sparsity"], (num_templates, *img_shape)
            )
            if flatten_img:
                templates = templates.reshape(num_templates, -1)
            color_means = jax.random.normal(color_means_key, (num_templates, color_dim))
            color_means = (
                color_means / jnp.linalg.norm(color_means, axis=-1, keepdims=True)
            ) * context["init_color_means_scale"]
            color_var = jnp.array(context["init_color_var_scale"])

        case ColoredSignalTemplateInitStrategy.PRIOR:
            assert "gt_templates" in context and isinstance(context["gt_templates"], Array)
            assert "gt_color_means" in context and isinstance(context["gt_color_means"], Array)
            assert (
                "gt_color_var" in context
                and isinstance(context["gt_color_var"], Array)
                and context["gt_color_var"].shape == ()
            )
            gt_templates, gt_color_means, gt_color_var = (
                context["gt_templates"],
                context["gt_color_means"],
                context["gt_color_var"],
            )
            X_new, y_new = ColoredSignalTemplateDistribution(
                gt_templates, gt_color_means, gt_color_var
            ).sample(key, num_templates)
            templates = gt_templates[y_new]
            color_means = get_color_vectors(X_new, y_new, templates)
            color_var = gt_color_var.copy()
        case ColoredSignalTemplateInitStrategy.PMEM:
            assert (
                "X_train" in context
                and isinstance(context["X_train"], Array)
                and context["X_train"].shape[1] == color_dim
            )
            assert "y_train" in context and isinstance(context["y_train"], Array)
            assert "init_color_var_scale" in context and isinstance(
                context["init_color_var_scale"], float
            )
            assert "init_means_noise_var" in context and isinstance(
                context["init_means_noise_var"], float
            )
            idx_key, noise_key = jax.random.split(key)
            if num_templates > context["X_train"].shape[0]:
                idx_to_memorize = jnp.concatenate(
                    [
                        jnp.arange(context["X_train"].shape[0]),
                        jax.random.randint(
                            idx_key,
                            (num_templates - context["X_train"].shape[0],),
                            0,
                            context["X_train"].shape[0],
                        ),
                    ]
                )
            else:
                idx_to_memorize = jax.random.choice(
                    idx_key,
                    jnp.arange(context["X_train"].shape[0]),
                    (num_templates,),
                    replace=False,
                )
            X_train = context["X_train"][idx_to_memorize]
            y_train = context["y_train"][idx_to_memorize]
            templates = context["gt_templates"][y_train]
            color_vectors = get_color_vectors(X_train, y_train, templates)
            color_means = color_vectors + jax.random.normal(noise_key, (num_templates, color_dim)) * (
                context["init_means_noise_var"] ** (1 / 2)
            )
            color_var = jnp.array(context["init_color_var_scale"])
        case _:
            raise ValueError(f"Invalid initialization strategy: {init_strategy}")

    return templates, color_means, color_var


class ColoredTemplatesSharedParametersEstimator(eqx.Module):
    num_templates: int = eqx.field(static=True)
    img_shape: Tuple[int, int] = eqx.field(static=True)
    flatten_img: bool = eqx.field(static=True)
    color_dim: int = eqx.field(static=True)
    vf_type: VectorFieldType = eqx.field(static=True)
    diffusion_process: DiffusionProcess = eqx.field(static=True)
    colored_templates_vf: Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array] = (
        eqx.field(static=True)
    )

    templates: Array = eqx.field(static=False)
    color_means: Array = eqx.field(static=False)
    color_std: Array = eqx.field(static=False)

    def __init__(
        self,
        num_templates: int,
        img_shape: Tuple[int, int],
        flatten_img: bool,
        color_dim: int,
        vf_type: VectorFieldType,
        diffusion_process: DiffusionProcess,
        init_templates: Array,
        init_color_means: Array,
        init_color_std: Array,
    ):
        super().__init__()
        self.diffusion_process: DiffusionProcess = diffusion_process
        self.num_templates: int = num_templates
        self.img_shape: Tuple[int, int] = img_shape
        self.flatten_img: bool = flatten_img
        self.color_dim: int = color_dim

        self.templates: Array = init_templates
        self.color_means: Array = init_color_means
        self.color_std: Array = init_color_std

        self.vf_type: VectorFieldType = vf_type
        self.colored_templates_vf: Callable[
            [Array, Array, DiffusionProcess, Array, Array, Array], Array
        ] = _get_correct_vf(vf_type)

    def __call__(self, x_t: Array, t: Array) -> Array:
        color_var = self.color_std**2
        return self.colored_templates_vf(
            x_t, t, self.diffusion_process, self.templates, self.color_means, color_var
        )
