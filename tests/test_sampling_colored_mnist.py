import jax 
from jax import numpy as jnp, vmap
from diffusionlab.dynamics import VariancePreservingProcess
from diffusionlab.schedulers import UniformScheduler
from diffusionlab.samplers import DDMSampler
from diffusionlab.distributions.gmm.gmm import GMM
from diffusionlab.distributions.empirical import EmpiricalDistribution
from diffusionlab.vector_fields import VectorFieldType 
from datasets import load_dataset
from diffusion_mem_gen.distributions.colored_signal_template_data import ColoredSignalTemplateDistribution
from diffusion_mem_gen.distributions.templates.mnist import generate_mnist_templates

import lovely_jax as lj
lj.monkey_patch()
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

key = jax.random.key(1)

num_samples_ground_truth = 100
num_samples_ddim = 50

num_templates = 3
template_shape = (20, 20)
flatten_templates = True

templates = generate_mnist_templates(num_templates, template_shape, flatten_templates)

color_dim = 3

color_means = jnp.zeros((num_templates, color_dim))
color_var = jnp.ones(())

gt_dist = ColoredSignalTemplateDistribution(templates, color_means, color_var)

key, subkey = jax.random.split(key)
X_ground_truth, y_ground_truth = gt_dist.sample(key, num_samples_ground_truth)


num_steps = 100
t_min = 0.0001 
t_max = 0.999

diffusion_process = VariancePreservingProcess()
scheduler = UniformScheduler()
ts = scheduler.get_ts(t_min=t_min, t_max=t_max, num_steps=num_steps)

key, subkey = jax.random.split(key)
X_noise = jax.random.normal(subkey, (num_samples_ddim, *X_ground_truth.shape[1:]))

zs = jax.random.normal(key, (num_samples_ddim, num_steps, *X_ground_truth.shape[1:]))

ground_truth_sampler = DDMSampler(diffusion_process, lambda x, t: gt_dist.x0(x, t, diffusion_process), VectorFieldType.X0, use_stochastic_sampler=False)
X_ddim_ground_truth = jax.vmap(lambda x_init, z: ground_truth_sampler.sample(x_init, z, ts))(X_noise, zs)

empirical_distribution = EmpiricalDistribution([(X_ground_truth, y_ground_truth)])
empirical_sampler = DDMSampler(diffusion_process, lambda x, t: empirical_distribution.x0(x, t, diffusion_process), VectorFieldType.X0, use_stochastic_sampler=False)
X_ddim_empirical = jax.vmap(lambda x_init, z: empirical_sampler.sample(x_init, z, ts))(X_noise, zs)

min_distance_to_gt_empirical = vmap(lambda x: jnp.min(vmap(lambda x_gt: jnp.linalg.norm(x - x_gt))(X_ground_truth)))(X_ddim_empirical)
min_distance_to_gt_ground_truth = vmap(lambda x: jnp.min(vmap(lambda x_gt: jnp.linalg.norm(x - x_gt))(X_ground_truth)))(X_ddim_ground_truth)

print(f"Min distance to ground truth samples from DDIM samples using empirical denoiser: {min_distance_to_gt_empirical}")
print(f"Min distance to ground truth samples from DDIM samples using ground truth denoiser: {min_distance_to_gt_ground_truth}")
# Plot samples from each set
import matplotlib.pyplot as plt

def plot_colored_mnist_samples(samples, title, num_to_plot=5):
    fig, axes = plt.subplots(1, num_to_plot, figsize=(15, 3))
    
    # Find global min and max across all samples for consistent normalization
    all_samples = jnp.concatenate([X_ground_truth[:num_to_plot], 
                                  X_ddim_ground_truth[:num_to_plot], 
                                  X_ddim_empirical[:num_to_plot]])
    global_min = all_samples.min()
    global_max = all_samples.max()
    
    for i in range(num_to_plot):
        # Transpose to get (height, width, channels) format for imshow
        img = samples[i].reshape(color_dim, *template_shape).transpose(1, 2, 0)
        # Normalize using global min/max for consistent colors
        img = (img - global_min) / (global_max - global_min)
        axes[i].imshow(img)
        axes[i].axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    return fig

# Plot ground truth samples
gt_fig = plot_colored_mnist_samples(X_ground_truth, "Ground Truth Samples")

# Plot DDIM samples using ground truth denoiser
ddim_gt_fig = plot_colored_mnist_samples(X_ddim_ground_truth, "DDIM Samples (Ground Truth Denoiser)")

# Plot DDIM samples using empirical denoiser
ddim_emp_fig = plot_colored_mnist_samples(X_ddim_empirical, "DDIM Samples (Empirical Denoiser)")

# Save the figures
# gt_fig.savefig("ground_truth_samples.png")
# ddim_gt_fig.savefig("ddim_ground_truth_samples.png")
# ddim_emp_fig.savefig("ddim_empirical_samples.png")

plt.show()
