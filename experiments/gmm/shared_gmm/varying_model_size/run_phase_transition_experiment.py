from copy import deepcopy
from datetime import datetime
from pathlib import Path

import jax 
import dill as pickle
from diffusionlab.vector_fields import VectorFieldType
from diffusionlab.samplers import DDMSampler

from diffusion_mem_gen.models.gmm import IsoHomGMMInitStrategy
from common.orchestrate import aggregate_metrics, generate_base_config, generate_training_configs, generate_baseline_configs, generate_data, run_baseline_configs, run_training_configs
from common.log_plot import log_and_plot_aggregated_metrics

timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
experiment_name = f"data_gmm_model_shared_gmm_expt_varying_model_size_phase_transition_{timestamp}"

experiment_dir = Path(__file__).parent / "results" / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=True)

num_gpus = len(jax.devices())

# === Configuration ===
seed = 1
dim = 30
gt_num_components = 4
gt_means_scale = dim ** (1 / 2)
gt_var_scale = 1.0
num_samples_train = 100
num_samples_val = 100
num_samples_eval = 50
memorization_ratio_threshold = 1 / 3
memorization_phase_transition_delta = 0.1
t_min = 0.001
t_max = 0.999
num_times = 26
vf_type = VectorFieldType.EPS
num_noise_draws_per_sample = 100
num_epochs_train = 10 #100_000
num_epochs_per_log = num_epochs_train # // 10
batch_size = num_samples_train
init_lr = 0.0
peak_lr = 1e-3
final_lr = 1e-6
adam_b1 = 0.9
adam_b2 = 0.999
adam_eps = 1e-8
gradient_clipping = 1.0e9
ema = 0.99
init_strategy = IsoHomGMMInitStrategy.PMEM
init_means_scale = gt_means_scale
init_var_scale = 1e-6
init_means_noise_var = 0.0

assert num_samples_train % batch_size == 0

train_skip = num_samples_train // 10
eval_skip = num_samples_train // 100
num_models_coverage = 10

assert num_samples_train % train_skip == 0
assert num_samples_train % eval_skip == 0

num_components_to_train = [10, 50, 100] # list(range(train_skip, num_samples_train + 1, train_skip))
num_components_to_evaluate = [10, 50, 100] #  list(range(eval_skip, num_samples_train + 1, eval_skip))

sampler_class = DDMSampler  # For RatioOfDistancesMetric

num_steps_train = num_epochs_train * num_samples_train // batch_size
num_steps_warmup = num_steps_train // 20
num_steps_decay = num_steps_train - num_steps_warmup

key = jax.random.PRNGKey(seed)
overall_data_key, overall_train_key, overall_eval_key = jax.random.split(key, 3)

config = {
    "experiment_group_name": experiment_name,
    "experiment_name": experiment_name,
    "seed": seed,
    "overall_data_key": overall_data_key,
    "overall_train_key": overall_train_key,
    "overall_eval_key": overall_eval_key,
    "dim": dim,
    "gt_num_components": gt_num_components,
    "gt_means_scale": gt_means_scale,
    "gt_var_scale": gt_var_scale,
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
    "init_means_scale": init_means_scale,
    "init_var_scale": init_var_scale,
    "init_means_noise_var": init_means_noise_var,
    "train_skip": train_skip,
    "eval_skip": eval_skip,
    "num_models_coverage": num_models_coverage,
    "num_components_to_train": num_components_to_train,
    "num_components_to_evaluate": num_components_to_evaluate,
    "sampler_class": sampler_class,
}
base_config_file = generate_base_config(config, experiment_dir)
data_file = generate_data(base_config_file, experiment_dir)
training_configs_files = generate_training_configs(base_config_file, experiment_dir)
baseline_configs_files = generate_baseline_configs(base_config_file, experiment_dir)
training_results_files = run_training_configs(training_configs_files, data_file, experiment_dir, max_workers=num_gpus, num_gpus=num_gpus, use_wandb=True)
baseline_results_files = run_baseline_configs(baseline_configs_files, data_file, experiment_dir, max_workers=num_gpus, num_gpus=num_gpus, use_wandb=False)
metrics_file = aggregate_metrics(training_results_files, baseline_results_files, experiment_dir)

# === Find start and end of memorization phase transition ===
# ASSUMPTION: memorization ratio is monotonically increasing in num_components
with open(metrics_file, "rb") as f:
    metrics = pickle.load(f)

memorization_phase_transition_start = None 
memorization_phase_transition_end = None 

for idx in range(len(num_components_to_train) - 1):
    memorization_ratio_i = metrics["trained"][num_components_to_train[idx]]["memorization_ratio"]
    memorization_ratio_i_plus_1 = metrics["trained"][num_components_to_train[idx + 1]]["memorization_ratio"]
    if memorization_phase_transition_start is None and memorization_ratio_i < memorization_phase_transition_delta and memorization_ratio_i_plus_1 >= memorization_phase_transition_delta:
        assert memorization_phase_transition_end is None
        memorization_phase_transition_start = num_components_to_train[idx]
    if memorization_phase_transition_end is None and memorization_ratio_i < 1 - memorization_phase_transition_delta and memorization_ratio_i_plus_1 >= 1 - memorization_phase_transition_delta:
        assert memorization_phase_transition_start is not None
        memorization_phase_transition_end = num_components_to_train[idx + 1]

if memorization_phase_transition_start is None:
    memorization_phase_transition_start = num_components_to_train[0]
if memorization_phase_transition_end is None:
    memorization_phase_transition_end = num_components_to_train[-1]

num_components_transition_delta = max((memorization_phase_transition_end - memorization_phase_transition_start) // num_models_coverage, 1)
new_num_components_to_train = list(range(memorization_phase_transition_start, memorization_phase_transition_end + 1, num_components_transition_delta))
with open(base_config_file, "rb") as f:
    base_config = pickle.load(f)
base_config["memorization_phase_transition_start"] = memorization_phase_transition_start
base_config["memorization_phase_transition_end"] = memorization_phase_transition_end
base_config["num_components_transition_delta"] = num_components_transition_delta
base_config["initial_num_components_to_train"] = deepcopy(num_components_to_train)
base_config["new_num_components_to_train"] = new_num_components_to_train
base_config["num_components_to_train"] = sorted(list(set(num_components_to_train + new_num_components_to_train)))

base_config_file = generate_base_config(base_config, experiment_dir)
training_configs_files = generate_training_configs(base_config_file, experiment_dir)
training_results_files = run_training_configs(training_configs_files, data_file, experiment_dir, max_workers=num_gpus, num_gpus=num_gpus, use_wandb=True)
metrics_file = aggregate_metrics(training_results_files, baseline_results_files, experiment_dir)

log_and_plot_aggregated_metrics(base_config_file, data_file, metrics_file, experiment_dir, use_wandb=True)

