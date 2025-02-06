from typing import Callable, Dict, Iterable, Optional, Tuple, cast
from jax import Array, numpy as jnp, vmap
import equinox as eqx
import jax
import optax
from optax._src.linear_algebra import global_norm as optax_global_norm
from tqdm import tqdm
from wandb.sdk.wandb_run import Run as WandBLogger
from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.vector_fields import VectorFieldType
from diffusionlab.losses import DiffusionLoss


optax_grad_norm = jax.jit(optax_global_norm)


def _log_name(metric_name: str, submetric_name: str) -> str:
    connection = "/" if len(metric_name) > 0 and len(submetric_name) > 0 else ""
    return f"{metric_name}{connection}{submetric_name}"


class DiffusionTrainer:
    diffusion_process: DiffusionProcess
    vf_type: VectorFieldType

    training_ts: Array
    training_ts_probs: Array
    training_ts_weights: Array
    num_noise_draws_per_sample: int

    train_metrics: Dict[str, Callable[[Array, eqx.Module, Array, Array], Dict[str, Array]]]
    val_metrics: Dict[str, Callable[[Array, eqx.Module, Array, Array], Dict[str, Array]]]
    batchfree_metrics: Dict[str, Callable[[Array, eqx.Module], Dict[str, Array]]]

    num_epochs_train: int
    num_epochs_per_metrics_log: int

    wandb_logger: Optional[WandBLogger] = None
    use_tqdm: bool = True

    loss: DiffusionLoss

    def __init__(
        self,
        diffusion_process: DiffusionProcess,
        vf_type: VectorFieldType,
        loss: DiffusionLoss,
        training_ts: Array,
        training_ts_probs: Array,
        training_ts_weights: Array,
        train_metrics: Dict[str, Callable[[Array, eqx.Module, Array, Array], Dict[str, Array]]],
        val_metrics: Dict[str, Callable[[Array, eqx.Module, Array, Array], Dict[str, Array]]],
        batchfree_metrics: Dict[str, Callable[[Array, eqx.Module], Dict[str, Array]]],
        num_epochs_train: int,
        num_epochs_per_metrics_log: int,
        use_tqdm: bool = True,
    ):
        self.diffusion_process: DiffusionProcess = diffusion_process
        self.vf_type: VectorFieldType = vf_type
        self.loss: DiffusionLoss = loss
        
        self.training_ts: Array = training_ts  # (num_times, )
        self.training_ts_probs: Array = training_ts_probs  # (num_times, )
        self.training_ts_weights: Array = training_ts_weights  # (num_times, )

        self.train_metrics: Dict[str, Callable[[Array, eqx.Module, Array, Array], Dict[str, Array]]] = train_metrics
        self.val_metrics: Dict[str, Callable[[Array, eqx.Module, Array, Array], Dict[str, Array]]] = val_metrics
        self.batchfree_metrics: Dict[str, Callable[[Array, eqx.Module], Dict[str, Array]]] = batchfree_metrics

        self.num_epochs_train: int = num_epochs_train
        self.num_epochs_per_metrics_log: int = num_epochs_per_metrics_log
        
        self.use_tqdm: bool = use_tqdm


    def batched_loss(self, key: Array, model: eqx.Module, x: Array, ts: Array) -> Array:
        key, subkey = jax.random.split(key)
        num_times = self.training_ts.shape[0]
        batch_size = x.shape[0]
        ts_idx = jax.random.choice(
            subkey, a=num_times, shape=(batch_size,), p=self.training_ts_probs
        )
        ts = self.training_ts[ts_idx]
        ts_weights = self.training_ts_weights[ts_idx]
        subkeys = jax.random.split(subkey, batch_size)
        loss_values = vmap(self.loss.loss, in_axes=(0, None, 0, 0))(
            subkeys, cast(Callable, model), x, ts
        )
        loss = jnp.mean(loss_values * ts_weights)
        return loss

    def train(
        self, 
        key: Array, 
        model: eqx.Module,
        optimizer_closure: Callable[[optax.Schedule], optax.GradientTransformation],
        lr_schedule: optax.Schedule,
        train_dataloader: Iterable[Tuple[Array, Array]], 
        val_dataloader: Iterable[Tuple[Array, Array]],
        wandb_logger: Optional[WandBLogger] = None  
    ) -> eqx.Module:
        optimizer = optimizer_closure(lr_schedule)
        opt_state = optimizer.init(cast(optax.Params, model))

        @jax.jit
        def train_step(
            model: eqx.Module, opt_state: optax.OptState, key: Array, x: Array, y: Array
        ) -> Tuple[eqx.Module, optax.OptState, Array, Array]:
            loss, grads = jax.value_and_grad(self.batched_loss, argnums=1)(key, model, x, y)
            updates, opt_state = optimizer.update(grads, opt_state, cast(optax.Params, model))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, grads

        num_steps = 0

        for epoch in tqdm(range(self.num_epochs_train), desc="Epochs", disable=not self.use_tqdm):
            
            key, subkey = jax.random.split(key)
            if epoch % self.num_epochs_per_metrics_log == 0 and wandb_logger:
                metric_values = self.collect_metrics(subkey, model, train_dataloader, val_dataloader)
                wandb_logger.log(metric_values, step=num_steps)
            
            for batch in train_dataloader:
                x_train, y_train = batch
                key, subkey = jax.random.split(key)
                model, opt_state, loss, grads = train_step(
                    model, opt_state, subkey, x_train, y_train
                )

                if wandb_logger:
                    step_metric_values = {
                        "train/loss": loss,
                        "train/grad_norm": optax_grad_norm(grads),
                        "train/lr": lr_schedule(num_steps),
                    }
                    wandb_logger.log(step_metric_values, step=num_steps)
                num_steps += 1

        key, subkey = jax.random.split(key)
        if wandb_logger:
            metric_values = self.collect_metrics(subkey, model, train_dataloader, val_dataloader)
            wandb_logger.log(metric_values, step=num_steps)

        return model

    def collect_metrics(
        self,
        key: Array,
        model: eqx.Module,
        train_dataloader: Iterable[Tuple[Array, Array]],
        val_dataloader: Iterable[Tuple[Array, Array]],
    ) -> Dict[str, Array]:
        metric_values = dict()
        key, subkey = jax.random.split(key)
        train_loss = self.collect_loss(subkey, model, train_dataloader)
        metric_values["metric/train/loss"] = train_loss
        key, subkey = jax.random.split(key)
        val_loss = self.collect_loss(subkey, model, val_dataloader)
        metric_values["metric/val/loss"] = val_loss
        key, subkey = jax.random.split(key)
        batchwise_train_metric_values = self.collect_batchwise_metrics(subkey, model, train_dataloader, self.train_metrics)
        metric_values.update({"metric/train/" + k: v for k, v in batchwise_train_metric_values.items()})
        key, subkey = jax.random.split(key)
        batchwise_val_metric_values = self.collect_batchwise_metrics(subkey, model, val_dataloader, self.val_metrics)
        metric_values.update({"metric/val/" + k: v for k, v in batchwise_val_metric_values.items()})
        key, subkey = jax.random.split(key)
        batchfree_metric_values = self.collect_batchfree_metrics(subkey, model, self.batchfree_metrics)
        metric_values.update({"metric/" + k: v for k, v in batchfree_metric_values.items()})
        return metric_values
    
    def collect_loss(
        self,
        key: Array,
        model: eqx.Module,
        dataloader: Iterable[Tuple[Array, Array]],
    ) -> Array:
        loss_values = []
        for x, y in dataloader:
            key, subkey = jax.random.split(key)
            loss_values.append(self.batched_loss(subkey, model, x, y))
        return jnp.mean(jnp.stack(loss_values))
    
    def collect_batchwise_metrics(
        self,
        key: Array,
        model: eqx.Module,
        dataloader: Iterable[Tuple[Array, Array]],
        batchwise_metrics: Dict[str, Callable[[Array, eqx.Module, Array, Array], Dict[str, Array]]],
    ) -> Dict[str, Array]:
        metric_values = dict()
        for metric_name, metric_fn in batchwise_metrics.items():
            metric_values_this_metric_list = []
            for x, y in dataloader:
                key, subkey = jax.random.split(key)
                metric_values_this_metric_list.append(metric_fn(subkey, model, x, y))
            num_batches = len(metric_values_this_metric_list)
            if num_batches > 0:
                submetric_names = list(metric_values_this_metric_list[0].keys())
                submetric_values_this_metric = {
                    submetric_name: [metric_values_this_metric_list[i][submetric_name] for i in range(num_batches)]
                    for submetric_name in submetric_names
                }
                for submetric_name in submetric_names:
                    metric_values[_log_name(metric_name, submetric_name)] = sum(submetric_values_this_metric[submetric_name]) / num_batches
        return metric_values
    
    def collect_batchfree_metrics(
        self,
        key: Array,
        model: eqx.Module,
        batchfree_metrics: Dict[str, Callable[[Array, eqx.Module], Dict[str, Array]]],
    ) -> Dict[str, Array]:
        metric_values = dict()
        for metric_name, metric_fn in batchfree_metrics.items():
            key, subkey = jax.random.split(key)
            metric_value = metric_fn(subkey, model)
            for submetric_name in metric_value.keys():
                metric_values[_log_name(metric_name, submetric_name)] = metric_value[submetric_name]
        return metric_values