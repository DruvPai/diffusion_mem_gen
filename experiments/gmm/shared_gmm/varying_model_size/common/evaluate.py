#!/usr/bin/env python3

import argparse
import shutil
import dill as pickle
from pathlib import Path
from typing import Any, Dict, Callable, cast

import jax
from jax import Array, numpy as jnp
import wandb
from tqdm import tqdm

from diffusionlab.dynamics import VariancePreservingProcess, DiffusionProcess
from diffusionlab.losses import DiffusionLoss
from diffusionlab.samplers import DDMSampler
from diffusionlab.schedulers import UniformScheduler
from diffusionlab.vector_fields import VectorFieldType
from diffusionlab.distributions.gmm.iso_hom_gmm import IsoHomGMM
from diffusionlab.distributions.empirical import EmpiricalDistribution

from diffusion_mem_gen.loss.loss_approximations import (
    iso_hom_gmm_gen_vf_excess_train_loss_compared_to_mem_vf_approx_factory,
    iso_hom_gmm_pmem_vf_excess_train_loss_compared_to_mem_vf_approx_factory,
    iso_hom_gmm_pmem_vf_excess_train_loss_compared_to_mem_vf_instancewise_approx_factory,
)
from diffusion_mem_gen import constants
from diffusion_mem_gen.utils.factories import inject_diffusion_process_to_vf, compute_loss_factory


def create_simplified_baseline_metrics_dict(num_times: int) -> Dict[str, Any]:
    """Creates a simplified structure for the baseline vector fields metrics dictionary."""
    metrics_dict = {
        "train_loss": [
            {
                "ground_truth": None,
                "memorizing": None,
                "partial_mem": None,
                "partial_mem_with_gt_means": None,
                "ground_truth_approx": None,
                "partial_mem_approx": None,
                "partial_mem_instancewise_approx": None,
            }
            for _ in range(num_times)
        ],
        "val_loss": [
            {
                "ground_truth": None,
                "memorizing": None,
                "partial_mem": None,
                "partial_mem_with_gt_means": None,
            }
            for _ in range(num_times)
        ],
    }
    return metrics_dict


def evaluate_baseline_vector_fields(baseline_config_file: Path, data_file: Path, output_dir: Path, use_wandb: bool = False):
    """
    Evaluate baseline vector fields for a specific number of components.
    These include ground truth, memorizing, and partial memorization vector fields.

    Args:
        baseline_config_file: Path to the pickle file containing the config for the baseline vector fields
        data_file: Path to the pickle file containing data
        output_dir: Directory to save results
        use_wandb: Whether to log results to wandb
    """
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(baseline_config_file, 'rb') as f:
        config = pickle.load(f)

    # Load data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    # Extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    gt_means = data['gt_means']
    gt_var = data['gt_var']
    gt_priors = data['gt_priors']
    ts = data['ts']

    # Setup parameters from config
    key = config['eval_key']
    num_components = config['num_components']  # Specific component count to evaluate
    vf_type = config['vf_type']
    dim = config['dim']
    gt_num_components = config['gt_num_components']
    num_times = config['num_times']
    experiment_group_name = config['experiment_group_name']
    sub_experiment_name = config['sub_experiment_name']

    # Setup JAX random key
    key_gt_approx, key_pmem_approx, key_pmem_inst_approx, key_eval = jax.random.split(key, 4)

    # Setup diffusion process
    diffusion_process = VariancePreservingProcess()
    gt_dist = IsoHomGMM(gt_means, gt_var, gt_priors)

    # Create loss function
    num_noise_draws_per_sample = config['num_noise_draws_per_sample']
    loss_obj = DiffusionLoss(diffusion_process, vf_type, num_noise_draws_per_sample)

    # Create simplified metrics dictionary
    metrics_dict = create_simplified_baseline_metrics_dict(num_times)

    # Create approximation functions
    time_dependent_ones = lambda t: jnp.ones_like(t)

    gt_approx_loss_fn = iso_hom_gmm_gen_vf_excess_train_loss_compared_to_mem_vf_approx_factory(
        dim, diffusion_process, vf_type, gt_var, time_dependent_ones
    )

    pmem_approx_loss_fn = iso_hom_gmm_pmem_vf_excess_train_loss_compared_to_mem_vf_approx_factory(
        dim, diffusion_process, vf_type, gt_var, time_dependent_ones
    )


    # Initialize wandb logger if requested
    logger = None
    if use_wandb:
        logger = wandb.init(
            entity=constants.WANDB_TEAM,
            project=constants.WANDB_PROJECT,
            group=experiment_group_name,
            name=sub_experiment_name,
            config=config,
        )

    # === Evaluate baseline vector fields ===
    key_gt_mem, key_eval_loops = jax.random.split(key_eval, 2)

    num_samples_train = X_train.shape[0]

    # Ground truth vector field
    dist_model_gt_vf = inject_diffusion_process_to_vf(
        gt_dist.get_vector_field(vf_type), diffusion_process
    )

    # Full memorizing vector field (using all training samples)
    dist_memorizing = EmpiricalDistribution([(X_train, None)])
    dist_model_memorizing_vf = inject_diffusion_process_to_vf(
        dist_memorizing.get_vector_field(vf_type), diffusion_process
    )

    for t_idx in range(num_times):
        t = ts[t_idx]
        compute_loss_at_t = compute_loss_factory(loss_obj, t)

        key_t_gt_mem_fold = jax.random.fold_in(key_gt_mem, t_idx)
        key_t_gt_train, key_t_gt_val, key_t_mem_train, key_t_mem_val = jax.random.split(key_t_gt_mem_fold, 4)

        # Ground Truth Denoiser
        metrics_dict["train_loss"][t_idx]["ground_truth"] = compute_loss_at_t(
            key_t_gt_train, dist_model_gt_vf, X_train
        )
        metrics_dict["val_loss"][t_idx]["ground_truth"] = compute_loss_at_t(
            key_t_gt_val, dist_model_gt_vf, X_val
        )

        # Memorizing Denoiser
        metrics_dict["train_loss"][t_idx]["memorizing"] = compute_loss_at_t(
            key_t_mem_train, dist_model_memorizing_vf, X_train
        )
        metrics_dict["val_loss"][t_idx]["memorizing"] = compute_loss_at_t(
            key_t_mem_val, dist_model_memorizing_vf, X_val
        )

        # Ground Truth Approximation
        metrics_dict["train_loss"][t_idx]["ground_truth_approx"] = (
            gt_approx_loss_fn(t) + metrics_dict["train_loss"][t_idx]["memorizing"]
        )

        key_iter_root = jax.random.fold_in(key_eval_loops, t_idx)
        key_pmem_choice, key_pmem_loss_train, key_pmem_loss_val, \
        key_pmem_gtm_choice_samples, key_pmem_gtm_choice_means, \
        key_pmem_gtm_loss_train, key_pmem_gtm_loss_val = jax.random.split(key_iter_root, 7)

        # Partial Memorization (using subset of training samples)
        if num_components < num_samples_train:
            samples_to_memorize_indices = jax.random.choice(
                key_pmem_choice, num_samples_train, (num_components,), replace=False
            )
        else:
            samples_to_memorize_indices = jnp.arange(num_samples_train)

        pmem_samples = X_train[samples_to_memorize_indices]
        pmem_dist = EmpiricalDistribution([(pmem_samples, None)])
        dist_model_partial_mem_vf = inject_diffusion_process_to_vf(
            pmem_dist.get_vector_field(vf_type), diffusion_process
        )
        metrics_dict["train_loss"][t_idx]["partial_mem"] = compute_loss_at_t(
            key_pmem_loss_train, dist_model_partial_mem_vf, X_train
        )
        metrics_dict["val_loss"][t_idx]["partial_mem"] = compute_loss_at_t(
            key_pmem_loss_val, dist_model_partial_mem_vf, X_val
        )

        # Partial Memorization Approximations
        if num_components < num_samples_train:
            memorize_sample_mask = jnp.zeros(num_samples_train, dtype=bool)
            memorize_sample_mask = memorize_sample_mask.at[samples_to_memorize_indices].set(True)

            metrics_dict["train_loss"][t_idx]["partial_mem_approx"] = (
                pmem_approx_loss_fn(
                    t,
                    num_components,
                    num_samples_train
                ) + metrics_dict["train_loss"][t_idx]["memorizing"]
            )
        else:
            metrics_dict["train_loss"][t_idx]["partial_mem_approx"] = metrics_dict["train_loss"][t_idx]["memorizing"]

    # Save results
    results = metrics_dict

    output_file = output_dir_path / f"baseline_results_{num_components}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    if logger:
        logger_dir = Path(logger.dir)
        logger.finish()
        shutil.rmtree(logger_dir.parent)

    # Log completion
    print(f"Baseline vector fields evaluation completed and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vector fields for a specific component count.")
    parser.add_argument("--baseline-config-file", type=str, required=True, help="Path to baseline config pickle file")
    parser.add_argument("--data-file", type=str, required=True, help="Path to data pickle file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--use-wandb", action="store_true", help="Whether to log results to wandb")

    args = parser.parse_args()

    evaluate_baseline_vector_fields(Path(args.baseline_config_file), Path(args.data_file), Path(args.output_dir), args.use_wandb)


if __name__ == "__main__":
    main()
