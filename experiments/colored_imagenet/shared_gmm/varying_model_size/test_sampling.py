from datetime import datetime
from pathlib import Path
from pkgutil import get_data
from turtle import distance

import jax
from jax import numpy as jnp
from diffusionlab.vector_fields import VectorFieldType
from diffusionlab.samplers import DDMSampler
from diffusionlab.schedulers import UniformScheduler
from diffusionlab.dynamics import VariancePreservingProcess
from diffusionlab.distributions.empirical import EmpiricalDistribution

from diffusion_mem_gen.distributions.colored_signal_template_data import ColoredSignalTemplateDistribution
from diffusion_mem_gen.distributions.templates.fashion_mnist import generate_fashion_mnist_templates
from diffusion_mem_gen.distributions.templates.imagenet_sketch import generate_imagenet_sketch_templates
from diffusion_mem_gen.models.colored_template import ColoredSignalTemplateInitStrategy
from common.orchestrate import aggregate_metrics, generate_base_config, generate_training_configs, generate_baseline_configs, generate_data, run_baseline_configs, run_training_configs
from common.log_plot import log_and_plot_aggregated_metrics
from diffusion_mem_gen.utils.factories import inject_diffusion_process_to_vf

timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
experiment_name = f"data_colored_imagenet_model_shared_colored_imagenet_expt_sampling_test_DEBUG_{timestamp}"

experiment_dir = Path(__file__).parent / "results" / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=True)

num_gpus = len(jax.devices())

# === Configuration ===
seed = 2
big_img_shape = (224, 224)
small_img_shape = (15, 15)
flatten_img = False
gt_num_templates = 6
gt_color_dim = 3
gt_color_var_scale = 1.0
num_samples_train = 100
num_samples_val = 100
num_samples_eval = 50
memorization_ratio_threshold = 1 / 3
t_min = 0.001
t_max = 0.999
num_times = 26
vf_type = VectorFieldType.EPS
num_noise_draws_per_sample = 100
num_epochs_train = 100_000
num_epochs_per_log = num_epochs_train // 10
batch_size = num_samples_train
init_lr = 0.0
peak_lr = 1e-3
final_lr = 1e-6
adam_b1 = 0.9
adam_b2 = 0.999
adam_eps = 1e-8
gradient_clipping = 1.0e9
ema = 0.99
init_strategy = ColoredSignalTemplateInitStrategy.PMEM
init_template_sparsity = 0.15
init_color_means_scale = 1.0
init_color_var_scale = 1e-6
init_means_noise_var = 0.0

assert num_samples_train % batch_size == 0

train_skip = num_samples_train // 10
eval_skip = num_samples_train // 100

assert num_samples_train % train_skip == 0
assert num_samples_train % eval_skip == 0

num_templates_to_train = [10, 50, 100] # list(range(train_skip, num_samples_train + 1, train_skip))
num_templates_to_evaluate = [10, 50, 100] #  list(range(eval_skip, num_samples_train + 1, eval_skip))

sampler_class = DDMSampler  # For RatioOfDistancesMetric

num_steps_train = num_epochs_train * num_samples_train // batch_size
num_steps_warmup = num_steps_train // 20
num_steps_decay = num_steps_train - num_steps_warmup

key = jax.random.PRNGKey(seed)
overall_data_key, overall_train_key, overall_eval_key = jax.random.split(key, 3)

config = {
    "experiment_group_name": experiment_name + "_GROUP_TEST",
    "experiment_name": experiment_name,
    "seed": seed,
    "overall_data_key": overall_data_key,
    "overall_train_key": overall_train_key,
    "overall_eval_key": overall_eval_key,
    "big_img_shape": big_img_shape,
    "small_img_shape": small_img_shape,
    "flatten_img": flatten_img,
    "gt_num_templates": gt_num_templates,
    "gt_color_dim": gt_color_dim,
    "gt_color_var_scale": gt_color_var_scale,
    "num_samples_train": num_samples_train,
    "num_samples_val": num_samples_val,
    "num_samples_eval": num_samples_eval,
    "memorization_ratio_threshold": memorization_ratio_threshold,
    "t_min": t_min,
    "t_max": t_max,
    "num_times": num_times,
    "vf_type": vf_type,
    "num_noise_draws_per_sample": num_noise_draws_per_sample,
    "num_epochs_train": num_epochs_train,
    "num_epochs_per_log": num_epochs_per_log,
    "num_steps_train": num_steps_train,
    "num_steps_warmup": num_steps_warmup,
    "num_steps_decay": num_steps_decay,
    "batch_size": batch_size,
    "init_lr": init_lr,
    "peak_lr": peak_lr,
    "final_lr": final_lr,
    "adam_b1": adam_b1,
    "adam_b2": adam_b2,
    "adam_eps": adam_eps,
    "gradient_clipping": gradient_clipping,
    "ema": ema,
    "init_strategy": init_strategy,
    "init_template_sparsity": init_template_sparsity,
    "init_color_means_scale": init_color_means_scale,
    "init_color_var_scale": init_color_var_scale,
    "init_means_noise_var": init_means_noise_var,
    "train_skip": train_skip,
    "eval_skip": eval_skip,
    "num_templates_to_train": num_templates_to_train,
    "num_templates_to_evaluate": num_templates_to_evaluate,
    "sampler_class": sampler_class,
}

# Set up JAX key
key, subkey = jax.random.split(key)
key_gt_means, key_train_data, key_val_data = jax.random.split(subkey, 3)

# Generate ground truth distribution
big_gt_templates = generate_fashion_mnist_templates(gt_num_templates, big_img_shape, flatten_templates=flatten_img)
small_gt_templates = generate_fashion_mnist_templates(gt_num_templates, small_img_shape, flatten_templates=flatten_img)
gt_color_means = jnp.zeros((gt_num_templates, gt_color_dim,))
gt_color_var = jnp.array(gt_color_var_scale)
big_gt_dist = ColoredSignalTemplateDistribution(big_gt_templates, gt_color_means, gt_color_var)
small_gt_dist = ColoredSignalTemplateDistribution(small_gt_templates, gt_color_means, gt_color_var)

# Sample data
big_X_train, y_train = big_gt_dist.sample(key_train_data, num_samples_train)
big_X_val, y_val = big_gt_dist.sample(key_val_data, num_samples_val)
small_X_train, y_train = small_gt_dist.sample(key_train_data, num_samples_train)
small_X_val, y_val = small_gt_dist.sample(key_val_data, num_samples_val)

# Generate time steps
diffusion_process = VariancePreservingProcess()
scheduler = UniformScheduler()
ts = scheduler.get_ts(t_min=t_min, t_max=t_max, num_steps=num_times - 1)

# Use empirical sampler (check bias of sampler setup)
empirical_dist = EmpiricalDistribution([(big_X_train, None)])
empirical_vf = inject_diffusion_process_to_vf(empirical_dist.get_vector_field(vf_type), diffusion_process)
empirical_sampler = sampler_class(diffusion_process, empirical_vf, vf_type, use_stochastic_sampler=False)

# Use GT sampler (check bias of GT denoise)
gt_dist = ColoredSignalTemplateDistribution(big_gt_templates, gt_color_means, gt_color_var)
gt_vf = inject_diffusion_process_to_vf(gt_dist.get_vector_field(vf_type), diffusion_process)
gt_sampler = sampler_class(diffusion_process, gt_vf, vf_type, use_stochastic_sampler=False)

key, subkey = jax.random.split(key)
x0_key, zs_key = jax.random.split(subkey)
x0 = jax.random.normal(x0_key, (num_samples_eval, gt_color_dim, *big_img_shape))
zs = jax.random.normal(zs_key, (num_samples_eval, num_times - 1, gt_color_dim, *big_img_shape))

empirical_samples = jax.vmap(lambda x0i, zsi: empirical_sampler.sample(x0i, zsi, ts))(x0, zs)
empirical_distance_to_X_train = jax.vmap(lambda x: jnp.min(jax.vmap(lambda x_train: jnp.linalg.norm(x - x_train))(big_X_train)))(empirical_samples)

gt_samples = jax.vmap(lambda x0i, zsi: gt_sampler.sample(x0i, zsi, ts))(x0, zs)
gt_distance_to_X_train = jax.vmap(lambda x: jnp.min(jax.vmap(lambda x_train: jnp.linalg.norm(x - x_train))(big_X_train)))(gt_samples)

# Save data to disk
data = {
    'big_X_train': big_X_train,
    'y_train': y_train,
    'big_X_val': big_X_val,
    'y_val': y_val,
    "big_gt_templates": big_gt_templates,
    "gt_color_means": gt_color_means,
    "gt_color_var": gt_color_var,
    'ts': ts,
}
# Visualize X_train samples
import matplotlib.pyplot as plt
import numpy as np

def visualize_samples(samples, num_to_show=5, save_path=None, img_shape=big_img_shape):
    """
    Visualize a subset of the training samples.
    
    Args:
        samples: Array of samples to visualize
        num_to_show: Number of samples to display
        save_path: Path to save the visualization (if None, just displays)
    """
    num_to_show = min(num_to_show, samples.shape[0])
    fig, axes = plt.subplots(1, num_to_show, figsize=(num_to_show * 3, 3))
    
    for i in range(num_to_show):
        if flatten_img:
            # Reshape if the images are flattened
            img = samples[i].reshape(img_shape + (-1,))
        else:
            img = samples[i]
            
        # Handle grayscale or RGB
        if img.shape[-1] == 1:
            axes[i].imshow(np.array(img.squeeze()), cmap='gray')
        else:
            # Ensure values are in proper range for RGB display
            img_display = np.array(img)
            axes[i].imshow(img_display.transpose(1, 2, 0))
            
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

# Create visualization directory
vis_dir = experiment_dir / "visualizations"
vis_dir.mkdir(exist_ok=True)

# Visualize and save training samples
visualize_samples(
    big_X_train, 
    num_to_show=5, 
    save_path=vis_dir / "big_training_samples.png",
    img_shape=big_img_shape
)

visualize_samples(
    small_X_train, 
    num_to_show=5, 
    save_path=vis_dir / "small_training_samples.png",
    img_shape=small_img_shape
)

visualize_samples(
    empirical_samples,
    num_to_show=5,
    save_path=vis_dir / "big_empirical_samples.png",
    img_shape=big_img_shape
)

visualize_samples(
    gt_samples,
    num_to_show=5,
    save_path=vis_dir / "big_gt_samples.png",
    img_shape=big_img_shape
)

print(f"distance of empirical distribution to X_train: {jnp.mean(empirical_distance_to_X_train)}")
print(f"distance of GT distribution to X_train: {jnp.mean(gt_distance_to_X_train)}")
print(f"average size of X_train: {jnp.mean(jax.vmap(lambda x: jnp.linalg.norm(x))(big_X_train))}")
print(f"big_X_train shape: {big_X_train.shape}")
print(f"small_X_train shape: {small_X_train.shape}")
print(f"Sample visualization saved to {vis_dir}")
