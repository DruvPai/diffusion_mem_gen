import argparse
import dill as pickle
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set


import jax
import numpy as np
import wandb
from tqdm import tqdm

from diffusionlab.dynamics import VariancePreservingProcess
from diffusionlab.schedulers import UniformScheduler
from diffusionlab.distributions.gmm.iso_hom_gmm import IsoHomGMM
from diffusion_mem_gen import constants
from diffusion_mem_gen.utils.scheduler import GPUJobScheduler

def generate_base_config(base_config: Dict[str, Any], output_dir: Path) -> Path:
    """
    Generate a config file and save it to disk.
    """
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    base_config_file = config_dir / "base_config.pkl"
    with open(base_config_file, 'wb') as f:
        pickle.dump(base_config, f)
    return base_config_file


def generate_data(base_config_file: Path, output_dir: Path) -> Path:
    """
    Generate data based on config and save it to disk.
    
    Args:
        base_config_path: Path to the base config file
        output_dir: Output directory path
        
    Returns:
        Path to the saved data file
    """
    # Extract config parameters
    with open(base_config_file, "rb") as f:
        config: Dict[str, Any] = pickle.load(f)

    key = config["overall_data_key"]
    dim = config["dim"]
    gt_num_components = config["gt_num_components"]
    gt_means_scale = config["gt_means_scale"]
    gt_var_scale = config["gt_var_scale"]
    num_samples_train = config["num_samples_train"]
    num_samples_val = config['num_samples_val']
    t_min = config['t_min']
    t_max = config['t_max']
    num_times = config['num_times']
    
    # Set up JAX key
    key_data, key_init_models = jax.random.split(key)
    key_gt_means, key_train_data, key_val_data = jax.random.split(key_data, 3)
    
    # Generate ground truth distribution
    gt_means= jax.random.normal(key_gt_means, (gt_num_components, dim))
    gt_means = (
        gt_means / jax.numpy.linalg.norm(gt_means, axis=-1, keepdims=True)
    ) * gt_means_scale
    gt_var = jax.numpy.array(gt_var_scale)
    gt_priors = jax.numpy.ones(gt_num_components) / gt_num_components
    gt_dist = IsoHomGMM(gt_means, gt_var, gt_priors)
    
    # Sample data
    X_train, y_train = gt_dist.sample(key_train_data, num_samples_train)
    X_val, y_val = gt_dist.sample(key_val_data, num_samples_val)
    
    # Generate time steps
    diffusion_process = VariancePreservingProcess()
    scheduler = UniformScheduler()
    ts = scheduler.get_ts(t_min=t_min, t_max=t_max, num_steps=num_times - 1)
    
    # Save data to disk
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'gt_means': gt_means,
        'gt_var': gt_var,
        'gt_priors': gt_priors,
        'ts': ts,
    }
    
    data_file = output_dir / "data.pkl"
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    
    return data_file


def generate_training_configs(
    base_config_file: Path, 
    output_dir: Path
) -> List[Path]:
    """
    Generate training configuration files for each component size.
    
    Args:
        base_config_file: Path to the base configuration file
        output_dir: Output directory path
        
    Returns:
        List of paths to the generated config files
    """
    # Create configs directory if it doesn't exist
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    
    # Get the list of component sizes to train
    with open(base_config_file, "rb") as f:
        base_config: Dict[str, Any] = pickle.load(f)
    train_components = base_config.get('num_components_to_train', [])
    if not train_components:
        return []
    
    # Generate configs for each component size
    key = base_config['overall_train_key']
    config_paths = []
    for num_components in train_components:
        # Create a copy of the base config
        config = base_config.copy()
        
        # Update config with component-specific settings
        config['train_key'] = jax.random.fold_in(key, num_components)
        config['num_components'] = num_components
        config['sub_experiment_name'] = f'{config["experiment_name"]}_train_model_size_{num_components}'
        
        # Save config to disk
        config_file = configs_dir / f"train_config_{num_components}.pkl"
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        
        config_paths.append(config_file)
    
    return config_paths


def generate_baseline_configs(
    base_config_file: Path, 
    output_dir: Path
) -> List[Path]:
    """
    Generate baseline evaluation configuration files for each component size.
    
    Args:
        base_config_file: Path to the base configuration file
        output_dir: Output directory path
        
    Returns:
        List of paths to the generated config files
    """
    # Create configs directory if it doesn't exist
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    
    # Get the list of component sizes to evaluate
    with open(base_config_file, "rb") as f:
        base_config: Dict[str, Any] = pickle.load(f)
    baseline_components = base_config.get('num_components_to_evaluate', [])
    if not baseline_components:
        return []
    
    # Generate configs for each component size
    key = base_config['overall_eval_key']
    config_files = []
    for num_components in baseline_components:
        # Create a copy of the base config
        config = base_config.copy()
        
        # Update config with component-specific settings
        config['eval_key'] = jax.random.fold_in(key, num_components)
        config['num_components'] = num_components
        config['sub_experiment_name'] = f'{config["experiment_name"]}_baseline_model_size_{num_components}'
        
        # Save config to disk
        config_file = configs_dir / f"baseline_config_{num_components}.pkl"
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        
        config_files.append(config_file)
    
    return config_files


def run_training_config(training_config_file: Path, data_file: Path, output_dir: Path, use_wandb: bool, gpu_id: int) -> Path:
    """
    Run a training task with the given configuration on a specific GPU.
    
    Args:
        config_file: Path to the config file
        data_file: Path to the data file
        output_dir: Directory to save results
        use_wandb: Whether to log results to wandb
        gpu_id: GPU ID to use for this job
        
    Returns:
        Path to the result file
    """
    # Extract component number from config filename
    try:
        n_components = int(training_config_file.stem.split("_")[-1])
    except (ValueError, IndexError):
        # If we can't extract the component number, use a default name
        n_components = "unknown"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [
        "python", "-m", 
        "common.train", 
        "--training-config-file", str(training_config_file),
        "--data-file", str(data_file),
        "--output-dir", str(output_dir)
    ]
    if use_wandb:
        cmd.append("--use-wandb")
    
    cwd = Path(__file__).parent.parent
    subprocess.run(cmd, check=True, env=env, cwd=cwd)
    
    # Return the path to the result file
    result_file = output_dir / f"model_results_{n_components}.pkl"
    return result_file


def run_baseline_config(baseline_config_file: Path, data_file: Path, output_dir: Path, use_wandb: bool, gpu_id: int) -> Path:
    """
    Run a baseline evaluation task with the given configuration on a specific GPU.
    
    Args:
        baseline_config_file: Path to the baseline config file
        data_file: Path to the data file
        output_dir: Directory to save results
        use_wandb: Whether to log results to wandb
        gpu_id: GPU ID to use for this job
        
    Returns:
        Path to the result file
    """
    # Extract component number from config filename
    try:
        n_components = int(baseline_config_file.stem.split("_")[-1])
    except (ValueError, IndexError):
        # If we can't extract the component number, use a default name
        n_components = "unknown"
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    cmd = [
        "python", "-m",
        "common.evaluate", 
        "--baseline-config-file", str(baseline_config_file),
        "--data-file", str(data_file),
        "--output-dir", str(output_dir)
    ]
    if use_wandb:
        cmd.append("--use-wandb")
    
    cwd = Path(__file__).parent.parent
    subprocess.run(cmd, check=True, env=env, cwd=cwd)
    
    # Return the path to the result file
    result_file = output_dir / f"baseline_results_{n_components}.pkl"
    return result_file


def run_training_configs(
    training_config_paths: List[Path], 
    data_path: Path, 
    output_dir: Path, 
    max_workers: int,
    num_gpus: int,
    completed_runs: Optional[Set[int]] = None,
    use_wandb: bool = True
) -> List[Path]:
    """
    Run training configs that haven't been executed yet, distributing across GPUs.
    
    Args:
        training_config_paths: List of paths to training config files
        data_path: Path to the data file
        output_dir: Output directory path
        max_workers: Maximum number of parallel workers
        num_gpus: Number of GPUs to use
        completed_runs: Set of component sizes that have already been run (optional, will check files if not provided)
        use_wandb: Whether to log results to wandb
    """
    # Create results directory if it doesn't exist
    train_results_dir = output_dir / "train_results"
    train_results_dir.mkdir(exist_ok=True)
    
    # Filter configs to only include ones that haven't been run yet
    result_files = []
    pending_configs = []
    for training_config_path in training_config_paths:
        # Extract component number from config filename
        try:
            n_components = int(training_config_path.stem.split("_")[-1])
            
            # Check if this run is already completed
            result_file = train_results_dir / f"model_results_{n_components}.pkl"
            
            # Skip if the result file exists or if it's in the completed_runs set
            if result_file.exists() or (completed_runs is not None and n_components in completed_runs):
                print(f"Skipping training for model_size={n_components} - result file already exists")
                result_files.append(result_file)
                continue
                
            pending_configs.append(training_config_path)
        except (ValueError, IndexError):
            # If we can't extract the component number, include the config anyway
            pending_configs.append(training_config_path)
    
    if not pending_configs:
        print("No new training configs to run.")
        return result_files
    
    # Use the minimum of max_workers and num_gpus
    effective_workers = min(max_workers, num_gpus)
    print(f"Running {len(pending_configs)} remaining training configs on {effective_workers} workers with {num_gpus} GPUs...")
    
    # Initialize the GPU scheduler
    scheduler = GPUJobScheduler(num_gpus)
    
    # Execute jobs with dynamic GPU allocation
    futures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        with tqdm(total=len(pending_configs), desc="Training models") as pbar:
            for config_path in pending_configs:
                future = executor.submit(
                    scheduler.run_job,
                    run_training_config,
                    config_path,
                    data_path,
                    train_results_dir,
                    use_wandb
                )
                future.add_done_callback(lambda _: pbar.update(1))
                futures.append((future, config_path))
                
            # Wait for all futures to complete
            for future, config_path in futures:
                try:
                    result_file = future.result()
                    result_files.append(result_file)
                    completed += 1
                    print(f"Completed training for {config_path} ({completed}/{len(pending_configs)})")
                except Exception as e:
                    print(f"Error in training for {config_path}: {e}")
    
    return result_files


def run_baseline_configs(
    baseline_config_paths: List[Path], 
    data_path: Path, 
    output_dir: Path, 
    max_workers: int,
    num_gpus: int,
    completed_runs: Optional[Set[int]] = None,
    use_wandb: bool = True
) -> List[Path]:
    """
    Run baseline configs that haven't been executed yet, distributing across GPUs.
    
    Args:
        baseline_config_paths: List of paths to baseline config files
        data_path: Path to the data file
        output_dir: Output directory path
        max_workers: Maximum number of parallel workers
        num_gpus: Number of GPUs to use
        completed_runs: Set of component sizes that have already been run (optional, will check files if not provided)
        use_wandb: Whether to log results to wandb
        
    Returns:
        List of paths to the result files
    """
    # Create results directory if it doesn't exist
    baseline_results_dir = output_dir / "baseline_results"
    baseline_results_dir.mkdir(exist_ok=True)
    
    # Filter configs to only include ones that haven't been run yet
    result_files = []
    pending_configs = []
    for baseline_config_path in baseline_config_paths:
        # Extract component number from config filename
        try:
            n_components = int(baseline_config_path.stem.split("_")[-1])
            
            # Check if this run is already completed
            result_file = baseline_results_dir / f"baseline_results_{n_components}.pkl"
            
            # Skip if the result file exists or if it's in the completed_runs set
            if result_file.exists() or (completed_runs is not None and n_components in completed_runs):
                print(f"Skipping baseline evaluation for model_size={n_components} - result file already exists")
                result_files.append(result_file)
                continue
                
            pending_configs.append(baseline_config_path)
        except (ValueError, IndexError):
            # If we can't extract the component number, include the config anyway
            pending_configs.append(baseline_config_path)
    
    if not pending_configs:
        print("No new baseline configs to run.")
        return result_files
    
    # Use the minimum of max_workers and num_gpus
    effective_workers = min(max_workers, num_gpus)
    print(f"Running {len(pending_configs)} remaining baseline configs on {effective_workers} workers with {num_gpus} GPUs...")
    
    # Initialize the GPU scheduler
    scheduler = GPUJobScheduler(num_gpus)
    
    # Execute jobs with dynamic GPU allocation
    futures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        with tqdm(total=len(pending_configs), desc="Evaluating baseline vector fields") as pbar:
            for config_path in pending_configs:
                future = executor.submit(
                    scheduler.run_job,
                    run_baseline_config,
                    config_path,
                    data_path,
                    baseline_results_dir,
                    use_wandb
                )
                future.add_done_callback(lambda _: pbar.update(1))
                futures.append((future, config_path))
                
            # Wait for all futures to complete
            for future, config_path in futures:
                try:
                    result_file = future.result()
                    result_files.append(result_file)
                    completed += 1
                    print(f"Completed baseline evaluation for {config_path} ({completed}/{len(pending_configs)})")
                except Exception as e:
                    print(f"Error in baseline evaluation for {config_path}: {e}")
    
    return result_files


def aggregate_metrics(
    training_results_files: List[Path],
    baseline_results_files: List[Path],
    output_dir: Path
) -> Path:
    """
    Aggregate results from multiple component runs.
    
    Args:
        base_config_path: Path to the base configuration file
        output_dir: Directory containing results
        
    Returns:
        Path to the results file
    """
    metrics = {
        "trained": {},
        "baselines": {},
    }

    for result_file in training_results_files:
        num_components = int(result_file.stem.split("_")[-1])
        with open(result_file, 'rb') as f:
            data = pickle.load(f)
            metrics["trained"][num_components] = data
    
    for result_file in baseline_results_files:
        num_components = int(result_file.stem.split("_")[-1])
        with open(result_file, 'rb') as f:
            data = pickle.load(f)
            metrics["baselines"][num_components] = data
    
    results_file = output_dir / "results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"Aggregated results saved to {results_file}")
    return results_file
