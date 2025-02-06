import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax.tree_util import tree_map
from typing import Iterator, Tuple

from diffusionlab.dynamics import VariancePreservingProcess
from diffusionlab.schedulers import UniformScheduler
from diffusionlab.vector_fields import VectorFieldType

from diffusion_mem_gen.metrics.loss_fixed_t import LossEachT
from diffusion_mem_gen.metrics.memorization import RatioOfDistancesMetric
from diffusion_mem_gen.models.gmm import IsoHomGMMSplitParametersEstimator
from diffusion_mem_gen.trainer import DiffusionTrainer
from diffusion_mem_gen.models.t_conditioned_mlp import TimeConditionedMLP
from diffusionlab.samplers import DDMSampler
from diffusionlab.losses import DiffusionLoss
#jax.config.update("jax_disable_jit", True)

# Updated generator function to yield dummy JAX array for y
def simple_data_loader(data: jax.Array, batch_size: int) -> Iterator[Tuple[jax.Array, jax.Array]]:
    num_samples = data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    indices = jnp.arange(num_samples)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        x_batch = data[batch_indices]
        # Create a dummy y_batch with the same batch size as x_batch
        y_batch = jnp.zeros((x_batch.shape[0],), dtype=x_batch.dtype) # Simple dummy array
        yield x_batch, y_batch

def test_gaussian_training():
    key = jax.random.PRNGKey(42)
    data_dim = 2
    num_samples = 128
    batch_size = 32
    num_epochs = 4
    lr = 1e-4

    # 1. Generate Gaussian Data
    key, data_key = jax.random.split(key)
    jax_data = jax.random.normal(data_key, (num_samples, data_dim))

    # 2. Create DataLoaders using the simple generator
    # The generator now yields (Array, Array) matching the expected type
    train_dataloader = simple_data_loader(jax_data, batch_size)
    val_dataloader = simple_data_loader(jax_data, batch_size)

    # 3. Initialize Diffusion Process
    diffusion_process = VariancePreservingProcess()
    scheduler = UniformScheduler()

    # 4. Training Timesteps Configuration
    num_times = 101
    training_ts = scheduler.get_ts(t_min=0.01, t_max=0.99, num_steps=num_times-1)
    training_ts_probs = jnp.ones_like(training_ts) / num_times
    training_ts_weights = jnp.ones_like(training_ts)

    # 5. Initialize Model
    vf_type = VectorFieldType.EPS
    key, model_key = jax.random.split(key)
    model = TimeConditionedMLP(
        key=model_key,
        dim=data_dim,
        mlp_dim=64,
        num_layers=3,
        adaln_mlp_dim=64,
        adaln_num_layers=3,
    )
    initial_params, initial_static = eqx.partition(model, eqx.is_array)

    # 6: Initialize Loss
    num_noise_draws_per_sample = 1
    loss = DiffusionLoss(diffusion_process, vf_type, num_noise_draws_per_sample)

    # 6. Optimizer and LR Schedule
    optimizer_closure = optax.adam
    lr_schedule = optax.constant_schedule(lr)

    # 7. Instantiate Metrics
    use_stochastic_sampler = False
    memorization_metric = RatioOfDistancesMetric(5, jax_data, training_ts, lambda net: DDMSampler(diffusion_process, net, vf_type, use_stochastic_sampler))
    loss_fixed_t_metric = LossEachT(training_ts, loss)


    # 7. Instantiate Trainer
    trainer = DiffusionTrainer(
        diffusion_process=diffusion_process,
        vf_type=VectorFieldType.EPS,
        loss=loss,
        training_ts=training_ts,
        training_ts_probs=training_ts_probs,
        training_ts_weights=training_ts_weights,
        train_metrics={"train/loss_fixed_t": loss_fixed_t_metric},
        val_metrics={"val/loss_fixed_t": loss_fixed_t_metric},
        batchfree_metrics={"memorization": memorization_metric},
        num_epochs_train=num_epochs,
        num_epochs_per_metrics_log=1, # Log less frequently in a real scenario
        wandb_logger=None, # No logging for this test
        use_tqdm=True # Keep tqdm enabled for the test run
    )

    # 8. Run Training
    key, train_key = jax.random.split(key)
    trained_model = trainer.train(train_key, model, optimizer_closure=optimizer_closure, lr_schedule=lr_schedule, train_dataloader=train_dataloader, val_dataloader=val_dataloader)

    # 9. Basic Check
    trained_params, trained_static = eqx.partition(trained_model, eqx.is_array)

    # Check that parameters have changed
    assert not jax.tree_util.tree_all(
        tree_map(jnp.array_equal, initial_params, trained_params)
    ), "Model parameters did not change during training."

    # Optional: Check model type remains the same
    assert isinstance(trained_model, TimeConditionedMLP)
    assert initial_static == trained_static

    print("Gaussian training test completed successfully.")

# If run directly, execute the test
if __name__ == "__main__":
    test_gaussian_training() 