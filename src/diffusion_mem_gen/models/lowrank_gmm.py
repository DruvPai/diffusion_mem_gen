from enum import Enum, auto
from typing import Any, Callable, Dict, Tuple
import jax
from jax import Array, numpy as jnp
import equinox as eqx

from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.distributions.gmm.low_rank_gmm import (
    LowRankGMM,
    low_rank_gmm_x0,
    low_rank_gmm_eps,
    low_rank_gmm_v,
    low_rank_gmm_score,
)
from diffusionlab.vector_fields import VectorFieldType


def _get_correct_vf(
    vf_type: VectorFieldType,
) -> Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array]:
    match vf_type:
        case VectorFieldType.X0:
            vf = low_rank_gmm_x0
        case VectorFieldType.EPS:
            vf = low_rank_gmm_eps
        case VectorFieldType.V:
            vf = low_rank_gmm_v
        case VectorFieldType.SCORE:
            vf = low_rank_gmm_score
        case _:
            raise ValueError(f"Invalid vector field type: {vf_type}")
    return vf


class LowRankGMMInitStrategy(Enum):
    GAUSSIAN = auto()
    SPHERE = auto()
    PRIOR = auto()
    PMEM = auto()


def low_rank_gmm_create_initialization_parameters(
    key: Array,
    init_strategy: LowRankGMMInitStrategy,
    dim: int,
    num_components: int,
    context: Dict[str, Any],
) -> Tuple[Array, Array, Array]:
    match init_strategy:
        case LowRankGMMInitStrategy.GAUSSIAN:
            assert "init_means_scale" in context and isinstance(context["init_means_scale"], float)
            assert "init_cov_scale" in context and isinstance(context["init_cov_scale"], float)
            assert "rank" in context and isinstance(context["rank"], int)
            key1, key2 = jax.random.split(key)
            means = (
                jax.random.normal(key1, (num_components, dim))
                * context["init_means_scale"]
                / (dim ** (1 / 2))
            )
            cov_factors = jax.random.normal(key2, (num_components, dim, context["rank"]))
            cov_factors = cov_factors / jnp.linalg.norm(cov_factors)
            cov_factors = cov_factors * jnp.sqrt(jnp.array(context["init_cov_scale"]))
            prior = jnp.ones((num_components,)) / num_components
        case LowRankGMMInitStrategy.SPHERE:
            assert "init_means_scale" in context and isinstance(context["init_means_scale"], float)
            assert "init_cov_scale" in context and isinstance(context["init_cov_scale"], float)
            assert "rank" in context and isinstance(context["rank"], int)
            key1, key2 = jax.random.split(key)
            means = jax.random.normal(key1, (num_components, dim))
            means = (means / jnp.linalg.norm(means, axis=-1, keepdims=True)) * context[
                "init_means_scale"
            ]
            cov_factors = jax.random.normal(key2, (num_components, dim, context["rank"]))
            cov_factors = cov_factors / jnp.linalg.norm(cov_factors)
            cov_factors = cov_factors * jnp.sqrt(jnp.array(context["init_cov_scale"]))
            prior = jnp.ones((num_components,)) / num_components
        case LowRankGMMInitStrategy.PRIOR:
            assert "rank" in context and isinstance(context["rank"], int)
            assert "gt_means" in context and isinstance(context["gt_means"], Array)
            assert (
                "gt_cov_factors" in context
                and isinstance(context["gt_cov_factors"], Array)
                and context["gt_cov_factors"].shape == (num_components, dim, context["rank"])
            )
            assert "gt_priors" in context and isinstance(context["gt_priors"], Array)
            gt_means, gt_cov_factors, gt_priors = (
                context["gt_means"],
                context["gt_cov_factors"],
                context["gt_priors"],
            )
            X_new, _ = LowRankGMM(gt_means, gt_cov_factors, gt_priors).sample(key, num_components)
            means = X_new
            cov_factors = gt_cov_factors
            prior = gt_priors
        case LowRankGMMInitStrategy.PMEM:
            assert (
                "X_train" in context
                and isinstance(context["X_train"], Array)
                and context["X_train"].shape[1] == dim
            )
            assert "init_cov_scale" in context and isinstance(context["init_cov_scale"], float)
            assert "init_means_noise_var" in context and isinstance(
                context["init_means_noise_var"], float
            )
            sk1, sk2, sk3 = jax.random.split(key, 3)
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
            cov_factors = jax.random.normal(sk3, (num_components, dim, context["rank"]))
            cov_factors = cov_factors / jnp.linalg.norm(cov_factors)
            cov_factors = cov_factors * jnp.sqrt(jnp.array(context["init_cov_scale"]))
            prior = jnp.ones((num_components,)) / num_components
        case _:
            raise ValueError(f"Invalid initialization strategy: {init_strategy}")
    return means, cov_factors, prior


class LowRankGMMSharedParametersEstimator(eqx.Module):
    num_components: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    vf_type: VectorFieldType = eqx.field(static=True)
    diffusion_process: DiffusionProcess = eqx.field(static=True)
    gmm_vf: Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array] = eqx.field(
        static=True
    )

    means: Array = eqx.field(static=False)
    cov_factors: Array = eqx.field(static=False)
    priors: Array = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_components: int,
        vf_type: VectorFieldType,
        diffusion_process: DiffusionProcess,
        init_means: Array,
        init_cov_factors: Array,
        priors: Array,
    ):
        super().__init__()
        self.diffusion_process: DiffusionProcess = diffusion_process
        self.num_components: int = num_components
        self.dim: int = dim

        self.means: Array = init_means
        self.cov_factors: Array = init_cov_factors
        self.priors: Array = priors

        self.vf_type: VectorFieldType = vf_type
        self.gmm_vf: Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array] = (
            _get_correct_vf(vf_type)
        )

    def __call__(self, x_t: Array, t: Array) -> Array:
        return self.gmm_vf(x_t, t, self.diffusion_process, self.means, self.cov_factors, self.priors)
