#!/usr/bin/env python3

import argparse
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, Mapping, cast

import dill as pickle
import jax
import optax
from diffusionlab.dynamics import VariancePreservingProcess
from diffusionlab.losses import DiffusionLoss
from diffusionlab.samplers import DDMSampler
from diffusionlab.schedulers import UniformScheduler
from jax import numpy as jnp

import wandb
from diffusion_mem_gen import constants
from diffusion_mem_gen.distributions.colored_signal_template_data import ColoredSignalTemplateDistribution
from diffusion_mem_gen.metrics.learned_means import LearnedMeans
from diffusion_mem_gen.metrics.learned_variance import LearnedVariances
from diffusion_mem_gen.metrics.loss_fixed_t import LossEachT
from diffusion_mem_gen.metrics.memorization import RatioOfDistancesMetric
from diffusion_mem_gen.metrics.utils import closest_k_points_in_dataset_indices
from diffusion_mem_gen.metrics.visualize_image import VisualizeImage
from diffusion_mem_gen.models.colored_template import ColoredTemplatesSharedParametersEstimator, colored_signal_template_create_initialization_parameters
from diffusion_mem_gen.utils.factories import compute_loss_factory
from diffusion_mem_gen.models.gmm import (
    IsoHomGMMSharedParametersEstimator,
    iso_hom_gmm_create_initialization_parameters,
)
from diffusion_mem_gen.optim.batching import batch_data
from diffusion_mem_gen.optim.wsd import wsd
from diffusion_mem_gen.trainer import DiffusionTrainer


def create_simplified_metrics_dict(
    num_times: int, learned_means_percentiles: list[float]
) -> Dict[str, Any]:
    """Creates a simplified structure for the metrics dictionary for a single model."""
    metrics_dict = {
        "memorization_ratio": None,
        "variance": None,
        "num_learned_mean_near_true_mean": None,
        "num_learned_mean_near_sample": None,
        "avg_distance_to_nearest_true_mean": None,
        "avg_distance_to_nearest_sample": None,
        "avg_distance_to_second_nearest_true_mean": None,
        "avg_distance_to_second_nearest_sample": None,
        "train_loss": [
            {
                "trained": None,
            }
            for _ in range(num_times)
        ],
        "val_loss": [
            {
                "trained": None,
            }
            for _ in range(num_times)
        ],
    }

    # Add percentile metrics
    for p in learned_means_percentiles:
        metrics_dict[f"p{p}_distance_to_nearest_true_mean"] = None
        metrics_dict[f"p{p}_distance_to_nearest_sample"] = None

    return metrics_dict


def train_model(
    training_config_file: Path, data_file: Path, output_dir: Path, use_wandb: bool = False
):
    """
    Read config and data from disk, initialize model, train it, and save results.

    Args:
        training_config_file: Path to the pickle training config file
        data_file: Path to the pickle file containing data
        output_dir: Directory to save results
        use_wandb: Whether to log results to wandb
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(training_config_file, "rb") as f:
        training_config = pickle.load(f)

    # Load data
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    # Extract data
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    gt_templates = data["gt_templates"]
    gt_color_means = data["gt_color_means"]
    gt_color_var = data["gt_color_var"]
    ts = data["ts"]

    # Setup parameters from config
    key = training_config["train_key"]
    num_templates = training_config["num_templates"]
    vf_type = training_config["vf_type"]
    img_shape = training_config["img_shape"]
    flatten_img = training_config["flatten_img"]
    gt_num_templates = training_config["gt_num_templates"]
    gt_color_dim = training_config["gt_color_dim"]
    t_min = training_config["t_min"]
    t_max = training_config["t_max"]
    num_times = training_config["num_times"]
    batch_size = training_config["batch_size"]
    num_noise_draws_per_sample = training_config["num_noise_draws_per_sample"]
    memorization_ratio_threshold = training_config["memorization_ratio_threshold"]
    num_samples_eval = training_config["num_samples_eval"]
    num_samples_viz = training_config["num_samples_viz"]
    num_epochs_train = training_config["num_epochs_train"]
    num_epochs_per_log = training_config["num_epochs_per_log"]
    init_strategy = training_config["init_strategy"]
    init_template_sparsity = training_config["init_template_sparsity"]
    init_color_means_scale = training_config["init_color_means_scale"]
    init_color_var_scale = training_config["init_color_var_scale"]
    init_means_noise_var = training_config["init_means_noise_var"]
    init_lr = training_config["init_lr"]
    peak_lr = training_config["peak_lr"]
    final_lr = training_config["final_lr"]
    num_steps_train = training_config["num_steps_train"]
    num_steps_warmup = training_config["num_steps_warmup"]
    num_steps_decay = training_config["num_steps_decay"]
    adam_b1 = training_config["adam_b1"]
    adam_b2 = training_config["adam_b2"]
    adam_eps = training_config["adam_eps"]
    gradient_clipping = training_config["gradient_clipping"]
    ema = training_config["ema"]
    experiment_group_name = training_config["experiment_group_name"]
    sub_experiment_name = training_config["sub_experiment_name"]

    # compute means 
    gt_means = jax.vmap(lambda mean, template: (mean[:, None] * template.reshape(-1)[None, :]).reshape(gt_color_dim, *img_shape))(
        gt_color_means, gt_templates
    )  # (num_templates, color_dim, prod(*data_dims))

    # Setup JAX random key
    key_init_model, key_train = jax.random.split(key)

    # Setup diffusion process
    diffusion_process = VariancePreservingProcess()
    scheduler = UniformScheduler()
    ts_weights = jax.numpy.ones_like(ts)
    ts_probs = jax.numpy.ones_like(ts) / len(ts)

    # Create data loaders
    train_dataloader = cast(Any, batch_data((X_train, y_train), batch_size))
    val_dataloader = cast(Any, batch_data((X_val, y_val), batch_size))

    # Setup loss and metrics
    loss_obj = DiffusionLoss(diffusion_process, vf_type, num_noise_draws_per_sample)

    sampler_class = DDMSampler

    # Initialize metrics
    train_loss_each_t_metric = LossEachT(ts, loss_obj)
    val_loss_each_t_metric = LossEachT(ts, loss_obj)
    learned_means_metric_obj = LearnedMeans(gt_means, X_train, memorization_ratio_threshold)
    memorization_metric_obj = RatioOfDistancesMetric(
        num_samples_eval,
        X_train,
        ts,
        lambda net: sampler_class(diffusion_process, net, vf_type, use_stochastic_sampler=False),
        memorization_ratio_threshold,
    )
    learned_variance_metric_obj = LearnedVariances()
    visualization_metric_obj = VisualizeImage(
        num_samples_viz, img_shape, flatten_img, gt_color_dim, ts, 
        lambda net: sampler_class(diffusion_process, net, vf_type, use_stochastic_sampler=False)
    )

    # Setup trainer metrics - Using Mapping instead of dict for type compatibility
    trainer_train_metrics: Mapping[str, Any] = {"loss_each_t": train_loss_each_t_metric}
    trainer_val_metrics: Mapping[str, Any] = {"loss_each_t": val_loss_each_t_metric}
    trainer_batchfree_metrics = {
        "memorization_metric": memorization_metric_obj,
        "learned_variance": learned_variance_metric_obj,
        "learned_means": learned_means_metric_obj,
        "visualize_samples": visualization_metric_obj,
    }

    # Initialize trainer
    trainer = DiffusionTrainer(
        diffusion_process,
        vf_type,
        loss_obj,
        ts,
        ts_probs,
        ts_weights,
        trainer_train_metrics,
        trainer_val_metrics,
        trainer_batchfree_metrics,
        num_epochs_train=num_epochs_train,
        num_epochs_per_metrics_log=num_epochs_per_log,
    )

    # Create simplified metrics dictionary for a single model
    metrics_dict = create_simplified_metrics_dict(num_times, learned_means_metric_obj.percentiles)

    # Configure optimizer
    def optimizer_closure(lr_sched: optax.Schedule) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(gradient_clipping),
            optax.adam(lr_sched, b1=adam_b1, b2=adam_b2, eps=adam_eps),
            optax.ema(ema),
        )

    lr_schedule_fn = wsd(
        init_lr, peak_lr, final_lr, num_steps_train, num_steps_warmup, num_steps_decay
    )

    # Initialize model
    init_context_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "init_template_sparsity": init_template_sparsity,
        "init_color_means_scale": init_color_means_scale,
        "init_color_var_scale": init_color_var_scale,
        "init_means_noise_var": init_means_noise_var,
        "gt_templates": gt_templates,
        "gt_color_means": gt_color_means,
        "gt_color_var": gt_color_var,
    }

    init_templates, init_color_means, init_color_var = colored_signal_template_create_initialization_parameters(
        key_init_model, init_strategy, img_shape, flatten_img, gt_color_dim, num_templates, init_context_dict
    )

    model = ColoredTemplatesSharedParametersEstimator(
        num_templates, img_shape, flatten_img, gt_color_dim, vf_type, diffusion_process, init_templates, init_color_means, init_color_var
    )

    # === Inline train_model_and_populate_metrics with simplified metrics structure ===
    (
        key_train,
        key_metrics,
        key_loss_main_root,
    ) = jax.random.split(key_train, 3)

    # Initialize wandb logger if requested
    logger = None
    if use_wandb:
        logger = wandb.init(
            entity=constants.WANDB_TEAM,
            project=constants.WANDB_PROJECT,
            group=experiment_group_name,
            name=sub_experiment_name,
            config=training_config,
        )

    # Train the model
    trained_model = trainer.train(
        key_train,
        model,
        optimizer_closure,
        lr_schedule_fn,
        train_dataloader,
        val_dataloader,
        logger,
    )
    trained_model = cast(IsoHomGMMSharedParametersEstimator, trained_model)
    if logger is not None:
        logger_dir = Path(logger.dir)
        logger.finish()
        shutil.rmtree(logger_dir.parent)

    # Compute metrics for trained model
    sub_key_mem, sub_key_var, sub_key_means, sub_key_viz = jax.random.split(key_metrics, 4)
    memorization_metric_value = memorization_metric_obj(sub_key_mem, trained_model)
    variance_value = learned_variance_metric_obj(sub_key_var, trained_model)
    learned_means_values = learned_means_metric_obj(sub_key_means, trained_model)
    visualization_values = visualization_metric_obj(sub_key_viz, trained_model)

    # Store metrics directly without nesting by model size
    metrics_dict["memorization_ratio"] = memorization_metric_value[""]
    metrics_dict["variance"] = variance_value[""]
    metrics_dict["num_learned_mean_near_true_mean"] = learned_means_values[
        "num_learned_mean_near_true_mean"
    ]
    metrics_dict["num_learned_mean_near_sample"] = learned_means_values[
        "num_learned_mean_near_sample"
    ]
    metrics_dict["avg_distance_to_nearest_true_mean"] = learned_means_values[
        "avg_distance_to_nearest_true_mean"
    ]
    metrics_dict["avg_distance_to_nearest_sample"] = learned_means_values[
        "avg_distance_to_nearest_sample"
    ]
    metrics_dict["avg_distance_to_second_nearest_true_mean"] = learned_means_values[
        "avg_distance_to_second_nearest_true_mean"
    ]
    metrics_dict["avg_distance_to_second_nearest_sample"] = learned_means_values[
        "avg_distance_to_second_nearest_sample"
    ]
    metrics_dict["visualize_samples"] = [x.image for x in visualization_values[""]]

    for p in learned_means_metric_obj.percentiles:
        metrics_dict[f"p{p}_distance_to_nearest_true_mean"] = learned_means_values[
            f"p{p}_distance_to_nearest_true_mean"
        ]
        metrics_dict[f"p{p}_distance_to_nearest_sample"] = learned_means_values[
            f"p{p}_distance_to_nearest_sample"
        ]

    # --- Loss calculations for the trained model ---
    for t_idx in range(training_config["num_times"]):
        compute_loss_at_t = compute_loss_factory(loss_obj, ts[t_idx])
        key_t_main_fold = jax.random.fold_in(key_loss_main_root, t_idx)
        key_t_main_train, key_t_main_val = jax.random.split(key_t_main_fold, 2)

        train_loss = compute_loss_at_t(key_t_main_train, cast(Callable, trained_model), X_train)
        val_loss = compute_loss_at_t(key_t_main_val, cast(Callable, trained_model), X_val)

        metrics_dict["train_loss"][t_idx]["trained"] = train_loss
        metrics_dict["val_loss"][t_idx]["trained"] = val_loss

    # Save results
    results = metrics_dict

    output_file = output_dir / f"model_results_{num_templates}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    # Log completion
    print(f"Model training completed and saved to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Train a GMM model based on config file.")
    parser.add_argument(
        "--training-config-file",
        type=str,
        required=True,
        help="Path to training config pickle file",
    )
    parser.add_argument("--data-file", type=str, required=True, help="Path to data pickle file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--use-wandb", action="store_true", help="Whether to log results to wandb")

    args = parser.parse_args()

    train_model(
        Path(args.training_config_file), Path(args.data_file), Path(args.output_dir), args.use_wandb
    )


if __name__ == "__main__":
    main()
