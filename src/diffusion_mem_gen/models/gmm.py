from enum import Enum, auto
from typing import Any, Callable, Dict, Tuple
import jax
from jax import Array, numpy as jnp
import equinox as eqx

from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.distributions.gmm.iso_hom_gmm import (
    IsoHomGMM,
    iso_hom_gmm_x0,
    iso_hom_gmm_eps,
    iso_hom_gmm_v,
    iso_hom_gmm_score,
)
from diffusionlab.vector_fields import VectorFieldType


def _get_correct_vf(
    vf_type: VectorFieldType,
) -> Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array]:
    match vf_type:
        case VectorFieldType.X0:
            vf = iso_hom_gmm_x0
        case VectorFieldType.EPS:
            vf = iso_hom_gmm_eps
        case VectorFieldType.V:
            vf = iso_hom_gmm_v
        case VectorFieldType.SCORE:
            vf = iso_hom_gmm_score
        case _:
            raise ValueError(f"Invalid vector field type: {vf_type}")
    return vf


class IsoHomGMMInitStrategy(Enum):
    GAUSSIAN = auto()
    SPHERE = auto()
    PRIOR = auto()
    PMEM = auto()


def iso_hom_gmm_create_initialization_parameters(
    key: Array,
    init_strategy: IsoHomGMMInitStrategy,
    dim: int,
    num_components: int,
    context: Dict[str, Any],
) -> Tuple[Array, Array, Array]:
    match init_strategy:
        case IsoHomGMMInitStrategy.GAUSSIAN:
            assert "init_means_scale" in context and isinstance(context["init_means_scale"], float)
            assert "init_var_scale" in context and isinstance(context["init_var_scale"], float)
            means = (
                jax.random.normal(key, (num_components, dim))
                * context["init_means_scale"]
                / (dim ** (1 / 2))
            )
            var = jnp.array(context["init_var_scale"])
            prior = jnp.ones((num_components,)) / num_components
        case IsoHomGMMInitStrategy.SPHERE:
            assert "init_means_scale" in context and isinstance(context["init_means_scale"], float)
            assert "init_var_scale" in context and isinstance(context["init_var_scale"], float)
            means = jax.random.normal(key, (num_components, dim))
            means = (means / jnp.linalg.norm(means, axis=-1, keepdims=True)) * context[
                "init_means_scale"
            ]
            var = jnp.ones(context["init_var_scale"])
            prior = jnp.ones((num_components,)) / num_components
        case IsoHomGMMInitStrategy.PRIOR:
            assert "gt_means" in context and isinstance(context["gt_means"], Array)
            assert (
                "gt_var" in context
                and isinstance(context["gt_var"], Array)
                and context["gt_var"].shape == ()
            )
            assert "gt_priors" in context and isinstance(context["gt_priors"], Array)
            gt_means, gt_var, gt_priors = (
                context["gt_means"],
                context["gt_var"],
                context["gt_priors"],
            )
            X_new, _ = IsoHomGMM(gt_means, gt_var, gt_priors).sample(key, num_components)
            means = X_new
            var = gt_var
            prior = gt_priors
        case IsoHomGMMInitStrategy.PMEM:
            assert (
                "X_train" in context
                and isinstance(context["X_train"], Array)
                and context["X_train"].shape[1] == dim
            )
            assert "init_var_scale" in context and isinstance(context["init_var_scale"], float)
            assert "init_means_noise_var" in context and isinstance(
                context["init_means_noise_var"], float
            )
            sk1, sk2 = jax.random.split(key)
            if num_components > context["X_train"].shape[0]:
                idx_to_memorize = jnp.concatenate(
                    [
                        jnp.arange(context["X_train"].shape[0]),
                        jax.random.randint(
                            sk1,
                            (num_components - context["X_train"].shape[0],),
                            0,
                            context["X_train"].shape[0],
                        ),
                    ]
                )
            else:
                idx_to_memorize = jax.random.choice(
                    sk1, context["X_train"].shape[0], (num_components,), replace=False
                )
            means = context["X_train"][idx_to_memorize] + jax.random.normal(
                sk2, (num_components, dim)
            ) * (context["init_means_noise_var"] ** (1 / 2))
            var = jnp.array(context["init_var_scale"])
            prior = jnp.ones((num_components,)) / num_components
        case _:
            raise ValueError(f"Invalid initialization strategy: {init_strategy}")
    return means, var, prior


class IsoHomGMMSharedParametersEstimator(eqx.Module):
    num_components: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    vf_type: VectorFieldType = eqx.field(static=True)
    diffusion_process: DiffusionProcess = eqx.field(static=True)
    gmm_vf: Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array] = eqx.field(
        static=True
    )

    means: Array = eqx.field(static=False)
    std: Array = eqx.field(static=False)
    priors: Array = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_components: int,
        vf_type: VectorFieldType,
        diffusion_process: DiffusionProcess,
        init_means: Array,
        init_var: Array,
        priors: Array,
    ):
        super().__init__()
        self.diffusion_process: DiffusionProcess = diffusion_process
        self.num_components: int = num_components
        self.dim: int = dim

        self.means: Array = init_means
        self.std: Array = jnp.sqrt(init_var)
        self.priors: Array = priors

        self.vf_type: VectorFieldType = vf_type
        self.gmm_vf: Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array] = (
            _get_correct_vf(vf_type)
        )

    def __call__(self, x_t: Array, t: Array) -> Array:
        var = self.std**2
        return self.gmm_vf(x_t, t, self.diffusion_process, self.means, var, self.priors)


class IsoHomGMMSplitParametersEstimator(eqx.Module):
    num_components: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    vf_type: VectorFieldType = eqx.field(static=True)
    diffusion_process: DiffusionProcess = eqx.field(static=True)
    ts: Array = eqx.field(static=True)
    gmm_vf: Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array] = eqx.field(
        static=True
    )

    means: Array = eqx.field(static=False)
    std: Array = eqx.field(static=False)
    priors: Array = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_components: int,
        vf_type: VectorFieldType,
        diffusion_process: DiffusionProcess,
        ts: Array,
        init_means: Array,
        init_var: Array,
        priors: Array,
    ):
        super().__init__()
        self.diffusion_process: DiffusionProcess = diffusion_process
        self.num_components: int = num_components
        self.dim: int = dim
        self.ts: Array = ts

        self.means: Array = init_means
        self.std: Array = jnp.sqrt(init_var)
        self.priors: Array = priors

        self.vf_type: VectorFieldType = vf_type
        self.gmm_vf: Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array] = (
            _get_correct_vf(vf_type)
        )

    def __call__(self, x_t: Array, t: Array) -> Array:
        idx = jnp.searchsorted(-self.ts, -t)  # negative since ts is in descending order
        idx = jnp.clip(idx, 0, self.num_components - 1)
        means = self.means[idx]
        var = self.std[idx] ** 2
        return self.gmm_vf(x_t, t, self.diffusion_process, means, var, self.priors)
