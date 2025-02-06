from typing import Dict, List, Tuple
import jax 
from jax import Array, numpy as jnp
import equinox as eqx

from diffusion_mem_gen.metrics.utils import closest_k_points_in_dataset_indices
from diffusion_mem_gen.models.gmm import IsoHomGMMSharedParametersEstimator, IsoHomGMMSplitParametersEstimator
from diffusion_mem_gen.models.colored_template import ColoredTemplatesSharedParametersEstimator

class LearnedMeans(eqx.Module):
    true_means: Array = eqx.field(static=True)
    X_train: Array = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    percentiles: List[float] = eqx.field(static=True)

    def __init__(self, true_means: Array, X_train: Array, threshold: float = 1/3, percentiles: List[float] = [10.0, 25.0, 50.0, 75.0, 90.0]):
        super().__init__()
        self.true_means: Array = true_means  # (num_components, *data_shape)
        self.X_train: Array = X_train  # (num_samples, *data_shape)
        self.threshold: float = threshold
        self.percentiles: List[float] = percentiles

    def compute_metrics(self, learned_means: Array) -> Tuple[Array, Array, Array, Array, Array, Array, Dict[str, Array], Dict[str, Array]]:
        more_than_one_true_mean = self.true_means.shape[0] > 1
        more_than_one_sample = self.X_train.shape[0] > 1

        if more_than_one_true_mean: 
            min_distances_to_means, _ = jax.vmap(lambda x: closest_k_points_in_dataset_indices(x, self.true_means, k=2))(learned_means)
        else:
            min_distances_to_means, _ = jax.vmap(lambda x: closest_k_points_in_dataset_indices(x, self.true_means, k=1))(learned_means)

        if more_than_one_sample:
            min_distances_to_samples, _ = jax.vmap(lambda x: closest_k_points_in_dataset_indices(x, self.X_train, k=2))(learned_means)
        else:
            min_distances_to_samples, _ = jax.vmap(lambda x: closest_k_points_in_dataset_indices(x, self.X_train, k=1))(learned_means)

        learned_mean_near_true_mean = (min_distances_to_means[:, 0] < self.threshold * min_distances_to_samples[:, 0])
        learned_mean_near_sample = (min_distances_to_samples[:, 0] < self.threshold * min_distances_to_means[:, 0])
        if more_than_one_true_mean:
            learned_mean_near_true_mean &= (min_distances_to_means[:, 0] < self.threshold * min_distances_to_means[:, 1])
        if more_than_one_sample:
            learned_mean_near_sample &= (min_distances_to_samples[:, 0] < self.threshold * min_distances_to_samples[:, 1])
        
        num_learned_mean_near_true_mean = jnp.sum(learned_mean_near_true_mean)
        num_learned_mean_near_sample = jnp.sum(learned_mean_near_sample)
        avg_distance_to_nearest_true_mean = jnp.mean(min_distances_to_means[:, 0])
        avg_distance_to_nearest_sample = jnp.mean(min_distances_to_samples[:, 0])
        
        # New metrics
        if more_than_one_true_mean:
            avg_distance_to_second_nearest_true_mean = jnp.mean(min_distances_to_means[:, 1])
        else:
            avg_distance_to_second_nearest_true_mean = jnp.nan * jnp.ones((1,))
        if more_than_one_sample:
            avg_distance_to_second_nearest_sample = jnp.mean(min_distances_to_samples[:, 1])
        else:
            avg_distance_to_second_nearest_sample = jnp.nan * jnp.ones((1,))
        
        # Compute percentiles for distances to nearest true mean/sample
        percentiles = jnp.array(self.percentiles)

        true_mean_percentiles = {
            f"p{p}_distance_to_nearest_true_mean": jnp.percentile(min_distances_to_means[:, 0], p)
            for p in percentiles
        }
        
        sample_percentiles = {
            f"p{p}_distance_to_nearest_sample": jnp.percentile(min_distances_to_samples[:, 0], p)
            for p in percentiles
        }

        return (
            num_learned_mean_near_true_mean, 
            num_learned_mean_near_sample, 
            avg_distance_to_nearest_true_mean, 
            avg_distance_to_nearest_sample,
            avg_distance_to_second_nearest_true_mean,
            avg_distance_to_second_nearest_sample,
            true_mean_percentiles,
            sample_percentiles
        )

    def __call__(self, key: Array, net: eqx.Module) -> Dict[str, Array]:
        metric_dict = dict()
        if isinstance(net, IsoHomGMMSplitParametersEstimator):
            for i in range(len(net.ts)):
                (
                    num_learned_mean_near_true_mean, 
                    num_learned_mean_near_sample, 
                    avg_distance_to_nearest_true_mean, 
                    avg_distance_to_nearest_sample,
                    avg_distance_to_second_nearest_true_mean,
                    avg_distance_to_second_nearest_sample,
                    true_mean_percentiles,
                    sample_percentiles
                ) = self.compute_metrics(net.means[i])
                
                metric_dict[f"num_learned_mean_near_true_mean/t={net.ts[i]:.3f}"] = num_learned_mean_near_true_mean
                metric_dict[f"num_learned_mean_near_sample/t={net.ts[i]:.3f}"] = num_learned_mean_near_sample
                metric_dict[f"avg_distance_to_nearest_true_mean/t={net.ts[i]:.3f}"] = avg_distance_to_nearest_true_mean
                metric_dict[f"avg_distance_to_nearest_sample/t={net.ts[i]:.3f}"] = avg_distance_to_nearest_sample
                metric_dict[f"avg_distance_to_second_nearest_true_mean/t={net.ts[i]:.3f}"] = avg_distance_to_second_nearest_true_mean
                metric_dict[f"avg_distance_to_second_nearest_sample/t={net.ts[i]:.3f}"] = avg_distance_to_second_nearest_sample
                
                # Add percentiles to the metric dictionary with the timestamp
                for k, v in true_mean_percentiles.items():
                    metric_dict[f"{k}/t={net.ts[i]:.3f}"] = v
                for k, v in sample_percentiles.items():
                    metric_dict[f"{k}/t={net.ts[i]:.3f}"] = v

        elif isinstance(net, IsoHomGMMSharedParametersEstimator):
            (
                num_learned_mean_near_true_mean, 
                num_learned_mean_near_sample, 
                avg_distance_to_nearest_true_mean, 
                avg_distance_to_nearest_sample,
                avg_distance_to_second_nearest_true_mean,
                avg_distance_to_second_nearest_sample,
                true_mean_percentiles,
                sample_percentiles
            ) = self.compute_metrics(net.means)
            
            metric_dict["num_learned_mean_near_true_mean"] = num_learned_mean_near_true_mean
            metric_dict["num_learned_mean_near_sample"] = num_learned_mean_near_sample
            metric_dict["avg_distance_to_nearest_true_mean"] = avg_distance_to_nearest_true_mean
            metric_dict["avg_distance_to_nearest_sample"] = avg_distance_to_nearest_sample
            metric_dict["avg_distance_to_second_nearest_true_mean"] = avg_distance_to_second_nearest_true_mean
            metric_dict["avg_distance_to_second_nearest_sample"] = avg_distance_to_second_nearest_sample
            
            # Add percentiles to the metric dictionary
            for k, v in true_mean_percentiles.items():
                metric_dict[k] = v
            for k, v in sample_percentiles.items():
                metric_dict[k] = v

        elif isinstance(net, ColoredTemplatesSharedParametersEstimator):
            means = jax.vmap(lambda color_mean, template: (color_mean[:, None] * template.reshape(-1)[None, :]).reshape(net.color_dim, *net.img_shape))(
                net.color_means, net.templates
            )  # (num_templates, color_dim, prod(*data_dims))
            (
                num_learned_mean_near_true_mean, 
                num_learned_mean_near_sample, 
                avg_distance_to_nearest_true_mean, 
                avg_distance_to_nearest_sample,
                avg_distance_to_second_nearest_true_mean,
                avg_distance_to_second_nearest_sample,
                true_mean_percentiles,
                sample_percentiles
            ) = self.compute_metrics(means)
            
            metric_dict["num_learned_mean_near_true_mean"] = num_learned_mean_near_true_mean
            metric_dict["num_learned_mean_near_sample"] = num_learned_mean_near_sample
            metric_dict["avg_distance_to_nearest_true_mean"] = avg_distance_to_nearest_true_mean
            metric_dict["avg_distance_to_nearest_sample"] = avg_distance_to_nearest_sample
            metric_dict["avg_distance_to_second_nearest_true_mean"] = avg_distance_to_second_nearest_true_mean
            metric_dict["avg_distance_to_second_nearest_sample"] = avg_distance_to_second_nearest_sample
            
            # Add percentiles to the metric dictionary
            for k, v in true_mean_percentiles.items():
                metric_dict[k] = v
            for k, v in sample_percentiles.items():
                metric_dict[k] = v


        else:
            raise ValueError(f"Unknown model type: {type(net)}")
                
        return metric_dict
