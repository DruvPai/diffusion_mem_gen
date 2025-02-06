from dataclasses import dataclass
from turtle import color
from typing import Any, Dict, Tuple, cast
import jax
from jax import Array, numpy as jnp
from math import prod

from diffusionlab.vector_fields import convert_vector_field_type, VectorFieldType
from diffusionlab.distributions.gmm.utils import _logdet_psd, _lstsq
from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.distributions.base import Distribution
from matplotlib.pylab import pinv


@dataclass(frozen=True)
class ColoredSignalTemplateDistribution(Distribution):
    """
    A distribution for signal template data. Each sample is a signal X (``Array[*data_dims]``), colored using a color vector ``c ~ N(μ, σ^2 I)``, i.e., the signal ``(c_1 X, ..., c_C X)`` which is an ``Array[color_dim, *data_dims]``.

    Attributes:
        - dist_params (``Dict[str, Array]``): Dictionary of parameters for the distribution, containing the following keys:

            - ``templates`` (``Array[num_templates, *data_dims]``): Signal templates.
            - ``color_means`` (``Array[num_templates, color_dim]``): Mean color values.
            - ``color_var`` (``Array[]``): Color variance.

        - dist_hparams (``Dict[str, Any]``): Dictionary containing hyperparameters for the distribution (unused for now).
    """
    def __init__(
        self,
        templates: Array,
        color_means: Array,
        color_var: Array,
    ):
        super().__init__(
            dist_params={
                "templates": templates,
                "color_means": color_means,
                "color_var": color_var,
            },
            dist_hparams={},
        )

    def sample(
        self,
        key: jax.Array,
        num_samples: int,
    ) -> Tuple[Array, Array]:
        """
        Sample from the colored signal template distribution.

        Args:
            key (``jax.Array``): The random key.
            num_samples (``int``): The number of samples to draw.

        Returns:
            ``Tuple[Array[num_samples, color_dim, *data_dims], Array[num_samples]]``: A tuple containing the samples and the labels.
        """
        templates = self.dist_params["templates"]  # (num_templates, *data_dims)
        color_means = self.dist_params["color_means"]  # (num_templates, color_dim)
        color_var= self.dist_params["color_var"]  # (num_templates)
        
        data_dims = templates.shape[1:]
        color_dim = color_means.shape[1]
        prod_data_dims = prod(data_dims)

        y_key, c_key = jax.random.split(key)
        y = jax.random.randint(y_key, (num_samples,), 0, templates.shape[0])  # (num_samples,)
        x0 = templates[y]  # (num_samples, *data_dims)
        c = jax.random.multivariate_normal(c_key, color_means[y], color_var * jnp.eye(color_dim), (num_samples,), method="eigh")  # (num_samples, color_dim)

        x0_flat = x0.reshape(num_samples, prod_data_dims)  # (num_samples, prod(*data_dims))
        x_flat = c[:, :, None] * x0_flat[:, None, :]  # (num_samples, color_dim, prod(*data_dims))
        x = x_flat.reshape(num_samples, color_dim, *data_dims)  # (num_samples, color_dim, *data_dims)
        return x, y

    def x0(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Compute the denoiser ``x0(x_t, t) = E[x_0 | x_t]`` of a colored signal template distribution w.r.t. a given ``diffusion_process``.

        Args:
            x_t (``Array[color_dim, *data_dims]``): The signal at time t.
            t (``Array[]``): The timestep.
            diffusion_process (``DiffusionProcess``): The diffusion process.

        Returns:
            ``Array[color_dim, *data_dims]``: The denoised signal.
        """
        return colored_signal_template_x0(
            x_t,
            t,
            diffusion_process,
            self.dist_params["templates"],
            self.dist_params["color_means"],
            self.dist_params["color_var"],
        )

    def eps(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Compute the noise prediction field ``eps(x_t, t) = E[ε | x_t]`` based on the provided x0 function.

        Args:
            x_t (``Array[color_dim, *data_dims]``): The signal at time t.
            t (``Array[]``): The timestep.
            diffusion_process (``DiffusionProcess``): The diffusion process.

        Returns:
            ``Array[color_dim, *data_dims]``: The noise prediction field.
        """
        return colored_signal_template_eps(
            x_t,
            t,
            diffusion_process,
            self.dist_params["templates"],
            self.dist_params["color_means"],
            self.dist_params["color_var"],
        )

    def v(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Compute the velocity prediction field ``v(x_t, t) = E[(d/dt)x_t | x_t]`` based on the provided x0 function.

        Args:
            x_t (``Array[color_dim, *data_dims]``): The signal at time t.
            t (``Array[]``): The timestep.
            diffusion_process (``DiffusionProcess``): The diffusion process.

        Returns:
            ``Array[color_dim, *data_dims]``: The velocity prediction field.
        """
        return colored_signal_template_v(
            x_t,
            t,
            diffusion_process,
            self.dist_params["templates"],
            self.dist_params["color_means"],
            self.dist_params["color_var"],
        )

    def score(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Compute the score ``score(x_t, t) = ∇_x log p_t(x_t)`` of the colored signal template distribution.

        Args:
            x_t (``Array[color_dim, *data_dims]``): The signal at time t.
            t (``Array[]``): The timestep.
            diffusion_process (``DiffusionProcess``): The diffusion process.

        Returns:
            ``Array[color_dim, *data_dims]``: The score of the colored signal template distribution.
        """
        return colored_signal_template_score(
            x_t,
            t,
            diffusion_process,
            self.dist_params["templates"],
            self.dist_params["color_means"],
            self.dist_params["color_var"],
        )


def colored_signal_template_x0(
    x_t: Array,
    t: Array,
    diffusion_process: DiffusionProcess,
    templates: Array,
    color_means: Array,
    color_var: Array,
) -> Array:
    """
    Compute the denoiser ``x0(x_t, t) = E[x_0 | x_t]`` of a colored signal template distribution w.r.t. a given ``diffusion_process``.

    Args:
        x_t (``Array[color_dim, *data_dims]``): The signal at time t.
        t (``Array[]``): The timestep.
        diffusion_process (``DiffusionProcess``): The diffusion process.
        templates (``Array[num_templates, *data_dims]``): The templates.
        color_means (``Array[num_templates, color_dim]``): The mean color values.
        color_var (``Array[]``): The color variance.

    Returns:
        ``Array[color_dim, *data_dims]``: The denoised signal.
    """
    color_dim = x_t.shape[0]
    data_dims = x_t.shape[1:]
    num_templates = templates.shape[0]
    prod_data_dims = prod(data_dims)

    alpha_t = diffusion_process.alpha(t)  # (,)
    sigma_t = diffusion_process.sigma(t)  # (,)

    x_t = x_t.reshape(color_dim, prod_data_dims)  # (color_dim, prod(*data_dims))
    templates = templates.reshape(
        num_templates, prod_data_dims
    )  # (num_templates, prod(*data_dims))

    means = jax.vmap(lambda color_mean, template: color_mean[:, None] * template[None, :])(
        color_means, templates
    )  # (num_templates, color_dim, prod(*data_dims))
    xbars = jax.vmap(lambda mean: x_t - alpha_t * mean)(
        means
    )  # (num_templates, color_dim, prod(*data_dims))
    template_squared_norms = jax.vmap(lambda template: jnp.sum(template**2))(
        templates
    )  # (num_templates,)

    At_xbars = jax.vmap(lambda template, xbar: jnp.sum(template[None, :] * xbar, axis=-1))(
        templates, xbars
    )  # (num_templates, color_dim)
    inner_covs_t_inv_At_xbars = jax.vmap(
        lambda template_squared_norm, At_xbar: ((alpha_t ** 2) * color_var) / (
            sigma_t ** 2 + (alpha_t ** 2) * color_var *  template_squared_norm
        ) * At_xbar
    )(template_squared_norms, At_xbars)  # (num_templates, color_dim)
    A_inner_covs_t_inv_At_xbars = jax.vmap(lambda template, c: template[None, :] * c[:, None])(
        templates, inner_covs_t_inv_At_xbars
    )  # (num_templates, color_dim, prod(*data_dims))
    covs_t_inv_xbars = (1 / sigma_t**2) * (
        xbars - A_inner_covs_t_inv_At_xbars
    )  # (num_templates, color_dim, prod(*data_dims))

    mahalanobis_dists = jax.vmap(lambda xbar, covs_t_inv_xbar: jnp.sum(xbar * covs_t_inv_xbar))(
        xbars, covs_t_inv_xbars
    )  # (num_templates, )
    logdets_covs_t = 2 * color_dim * prod_data_dims * jnp.log(sigma_t) + jax.vmap(
        lambda template_squared_norm: jnp.log(1 + alpha_t ** 2 * color_var * template_squared_norm / (sigma_t ** 2))
    )(template_squared_norms)  # (num_templates, )

    weights = -1 / 2 * (logdets_covs_t + mahalanobis_dists)  # (num_templates)
    posterior_probs = jax.nn.softmax(weights)  # (num_templates)

    weighted_normalized_x = jnp.sum(
        jax.vmap(lambda posterior_prob, covs_t_inv_xbar: posterior_prob * covs_t_inv_xbar)(
            posterior_probs, covs_t_inv_xbars
        ),
        axis=0,
    )  # (color_dim, prod(*data_dims))
    x0_hat = (1 / alpha_t) * (
        x_t - (sigma_t**2) * weighted_normalized_x
    )  # (color_dim, prod(*data_dims))

    x0_hat = x0_hat.reshape(color_dim, *data_dims)  # (color_dim, *data_dims)
    return x0_hat


def colored_signal_template_eps(
    x_t: Array,
    t: Array,
    diffusion_process: DiffusionProcess,
    templates: Array,
    color_means: Array,
    color_var: Array,
) -> Array:
    """
    Compute the noise prediction field ε based on the provided x0 function.
    """
    f_x_t = colored_signal_template_x0(x_t, t, diffusion_process, templates, color_means, color_var)
    eps_x_t = convert_vector_field_type(
        x_t,
        f_x_t,
        diffusion_process.alpha(t),
        diffusion_process.sigma(t),
        diffusion_process.alpha_prime(t),
        diffusion_process.sigma_prime(t),
        VectorFieldType.X0,
        VectorFieldType.EPS,
    )
    return eps_x_t


def colored_signal_template_v(
    x_t: Array,
    t: Array,
    diffusion_process: DiffusionProcess,
    templates: Array,
    color_means: Array,
    color_var: Array,
) -> Array:
    """
    Compute the noise prediction field v based on the provided x0 function.
    """
    f_x_t = colored_signal_template_x0(x_t, t, diffusion_process, templates, color_means, color_var)
    v_x_t = convert_vector_field_type(
        x_t,
        f_x_t,
        diffusion_process.alpha(t),
        diffusion_process.sigma(t),
        diffusion_process.alpha_prime(t),
        diffusion_process.sigma_prime(t),
        VectorFieldType.X0,
        VectorFieldType.V,
    )
    return v_x_t


def colored_signal_template_score(
    x_t: Array,
    t: Array,
    diffusion_process: DiffusionProcess,
    templates: Array,
    color_means: Array,
    color_var: Array,
) -> Array:
    """
    Compute the score of the colored signal template distribution.
    """
    f_x_t = colored_signal_template_x0(x_t, t, diffusion_process, templates, color_means, color_var)
    score_x_t = convert_vector_field_type(
        x_t,
        f_x_t,
        diffusion_process.alpha(t),
        diffusion_process.sigma(t),
        diffusion_process.alpha_prime(t),
        diffusion_process.sigma_prime(t),
        VectorFieldType.X0,
        VectorFieldType.SCORE,
    )
    return score_x_t
