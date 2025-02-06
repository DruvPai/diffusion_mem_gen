import shutil
from typing import Any, Dict, List, Optional
from pathlib import Path
import dill as pickle

from jax import numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from wandb.sdk.wandb_run import Run

from diffusion_mem_gen.metrics.learned_means import LearnedMeans  # For accessing percentiles
from diffusion_mem_gen import constants


def log_and_plot_aggregated_metrics(
    base_config_file: Path,
    data_file: Path,
    metrics_file: Path,
    experiment_dir: Path,
    use_wandb: bool = False
) -> List[Path]:
    """
    Logs and plots metrics from the aggregated result dictionary produced by orchestrate.py.

    Args:
        base_config_file: Path to base config
        data_file: Path to data
        metrics_file: Path to metrics
        experiment_dir: Directory to save plots
        use_wandb: Whether to log to wandb

    Returns:
        List of paths to plot files
    """

    with open(base_config_file, "rb") as f:
        base_config: Dict[str, Any] = pickle.load(f)

    with open(data_file, "rb") as f:
        data_config: Dict[str, Any] = pickle.load(f)

    with open(metrics_file, "rb") as f:
        metrics: Dict[str, Any] = pickle.load(f)

    # Create plots directory
    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_files = []

    # Get base config for metadata
    memorization_ratio_threshold = base_config.get("memorization_ratio_threshold", 0.0)
    num_times = base_config.get("num_times", 1)
    ts_values = data_config.get("ts", jnp.array([]))

    # Extract metrics
    trained_models = metrics.get('trained', {})
    baseline_models = metrics.get('baselines', {})

    # Initialize wandb logger if requested
    summary_logger = None
    if use_wandb:
        summary_logger = wandb.init(
            entity=constants.WANDB_TEAM,
            project=constants.WANDB_PROJECT,
            group=base_config.get("experiment_group_name", base_config.get("experiment_name", None)),
            name=f"{base_config.get('experiment_name', 'experiment')}_summary",
            config=base_config,
        )
        summary_logger.save(metrics_file, metrics_file.parent)

    # Determine the LearnedMeans percentiles (try to extract from a model if available)
    percentiles = [0.1, 0.5, 0.9]  # Default if not found
    if trained_models:
        # Get the first model to extract percentiles
        first_model_key = next(iter(trained_models))
        first_model = trained_models[first_model_key]
        # Extract percentiles from metrics keys
        for key in first_model:
            if key.startswith("p") and "_distance_to_nearest" in key:
                try:
                    p_value = float(key.split("p")[1].split("_")[0])
                    percentiles.append(p_value)
                except (ValueError, IndexError):
                    pass
        percentiles = sorted(list(set(percentiles)))

    # === Plotting trained model metrics ===
    if trained_models:
        # Get all model sizes
        model_sizes = sorted(trained_models.keys())

        # Memorization Ratio
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="darkgrid")
        if all("memorization_ratio" in trained_models[size] for size in model_sizes):
            y_mem_ratio = [
                trained_models[size]["memorization_ratio"].item() for size in model_sizes
            ]
            plt.plot(model_sizes, y_mem_ratio, marker="o", label="Trained Models")
            plt.xlabel("Number of Means")
            plt.ylabel(f"Memorization Ratio (Threshold={memorization_ratio_threshold:.3f})")
            plt.title("Memorization Ratio vs Number of Means")
            plt.tight_layout()
            save_path = plot_dir / "trained_memorization_ratio.png"
            plt.savefig(save_path)
            plt.close()
            if summary_logger:
                summary_logger.log(
                    {"trained/memorization_ratio": wandb.Image(str(save_path.absolute()))},
                    step=0,
                )
            plot_files.append(save_path)

        # Variance
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="darkgrid")
        if all("variance" in trained_models[size] for size in model_sizes):
            y_var = [trained_models[size]["variance"].item() for size in model_sizes]
            plt.plot(model_sizes, y_var, marker="o", label="Trained Models")
            plt.xlabel("Number of Means")
            plt.ylabel("Learned Variance")
            plt.yscale("log")
            plt.title("Learned Variance vs Number of Means")
            plt.tight_layout()
            plt.legend()
            save_path = plot_dir / "trained_variance.png"
            plt.savefig(save_path)
            plt.close()
            if summary_logger:
                summary_logger.log(
                    {"trained/variance": wandb.Image(str(save_path.absolute()))}, step=0
                )
            plot_files.append(save_path)

        # Learned Mean Near True Mean
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="darkgrid")
        if all("num_learned_mean_near_true_mean" in trained_models[size] for size in model_sizes):
            y_lm_true = [
                trained_models[size]["num_learned_mean_near_true_mean"].item() for size in model_sizes
            ]
            plt.plot(model_sizes, y_lm_true, marker="o", label="Trained Models")
            plt.xlabel("Number of Means")
            plt.ylabel(
                f"Num Learned Means Near True Means (Threshold={memorization_ratio_threshold:.3f})"
            )
            plt.title("Num Learned Means Near True Means vs Number of Means")
            plt.tight_layout()
            save_path = plot_dir / "trained_learned_mean_near_true_mean.png"
            plt.savefig(save_path)
            plt.close()
            if summary_logger:
                summary_logger.log(
                    {
                        "trained/num_learned_mean_near_true_mean": wandb.Image(
                            str(save_path.absolute())
                        )
                    },
                    step=0,
                )
            plot_files.append(save_path)

        # Learned Mean Near Sample
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="darkgrid")
        if all("num_learned_mean_near_sample" in trained_models[size] for size in model_sizes):
            y_lm_sample = [
                trained_models[size]["num_learned_mean_near_sample"].item() for size in model_sizes
            ]
            plt.plot(model_sizes, y_lm_sample, marker="o", label="Trained Models")
            plt.xlabel("Number of Means")
            plt.ylabel(
                f"Num Learned Means Near Samples (Threshold={memorization_ratio_threshold:.3f})"
            )
            plt.title("Num Learned Means Near Samples vs Number of Means")
            plt.tight_layout()
            save_path = plot_dir / "trained_learned_mean_near_sample.png"
            plt.savefig(save_path)
            plt.close()
            if summary_logger:
                summary_logger.log(
                    {
                        "trained/num_learned_mean_near_sample": wandb.Image(
                            str(save_path.absolute())
                        )
                    },
                    step=0,
                )
            plot_files.append(save_path)

        # Average Distance to Nearest True Mean
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="darkgrid")
        if all("avg_distance_to_nearest_true_mean" in trained_models[size] for size in model_sizes):
            y_avg_dist_true = [
                trained_models[size]["avg_distance_to_nearest_true_mean"].item() for size in model_sizes
            ]
            y_avg_dist_sec_true = [
                trained_models[size]["avg_distance_to_second_nearest_true_mean"].item() for size in model_sizes
            ]
            plt.plot(
                model_sizes,
                y_avg_dist_true,
                marker="o",
                label="Avg Dist to Nearest True Mean",
            )
            plt.plot(
                model_sizes,
                y_avg_dist_sec_true,
                marker="o",
                label="Avg Dist to 2nd Nearest True Mean",
            )
            for p in percentiles:
                key = f"p{p}_distance_to_nearest_true_mean"
                if all(key in trained_models[size] for size in model_sizes):
                    y_p_dist_true = [
                        trained_models[size][key].item() for size in model_sizes
                    ]
                    plt.plot(
                        model_sizes,
                        y_p_dist_true,
                        marker="o",
                        label=f"{p}th Percentile Dist to Nearest True Mean",
                    )
            plt.xlabel("Number of Means")
            plt.ylabel("Distance")
            plt.title("Avg/Percentile Distance to Nearest True Mean vs Num Means (Trained)")
            plt.tight_layout()
            plt.legend()
            save_path = plot_dir / "trained_avg_dist_to_true_mean.png"
            plt.savefig(save_path)
            plt.close()
            if summary_logger:
                summary_logger.log(
                    {
                        "trained/dist_to_true_mean": wandb.Image(
                            str(save_path.absolute())
                        )
                    },
                    step=0,
                )
            plot_files.append(save_path)

        # Average Distance to Nearest Sample
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="darkgrid")
        if all("avg_distance_to_nearest_sample" in trained_models[size] for size in model_sizes):
            y_avg_dist_sample = [
                trained_models[size]["avg_distance_to_nearest_sample"].item() for size in model_sizes
            ]
            y_avg_dist_sec_sample = [
                trained_models[size]["avg_distance_to_second_nearest_sample"].item() for size in model_sizes
            ]
            plt.plot(
                model_sizes,
                y_avg_dist_sample,
                marker="o",
                label="Avg Dist to Nearest Sample",
            )
            plt.plot(
                model_sizes,
                y_avg_dist_sec_sample,
                marker="o",
                label="Avg Dist to 2nd Nearest Sample",
            )
            for p in percentiles:
                key = f"p{p}_distance_to_nearest_sample"
                if all(key in trained_models[size] for size in model_sizes):
                    y_p_dist_sample = [
                        trained_models[size][key].item() for size in model_sizes
                    ]
                    plt.plot(
                        model_sizes,
                        y_p_dist_sample,
                        marker="o",
                        label=f"{p}th Percentile Dist to Nearest Sample",
                    )
            plt.xlabel("Number of Means")
            plt.ylabel("Distance")
            plt.title("Avg/Percentile Distance to Nearest Sample vs Num Means (Trained)")
            plt.tight_layout()
            plt.legend()
            save_path = plot_dir / "trained_avg_dist_to_sample.png"
            plt.savefig(save_path)
            plt.close()
            if summary_logger:
                summary_logger.log(
                    {"trained/dist_to_sample": wandb.Image(str(save_path.absolute()))},
                    step=0,
                )
            plot_files.append(save_path)

    # === Plotting loss metrics ===
    # Define categories and styles
    loss_types = ["train_loss", "val_loss"]
    data_categories = [
        "ground_truth",
        "memorizing",
        "ground_truth_approx",
        "partial_mem",
        "partial_mem_approx",
        "trained",
    ]

    line_styles = {
        "ground_truth": (sns.color_palette()[0], "--"),
        "memorizing": ("black", "--"),
        "ground_truth_approx": (sns.color_palette()[0], "-."),
        "partial_mem": (sns.color_palette()[1], "-"),
        "partial_mem_approx": (sns.color_palette()[1], "-."),
        "trained": (sns.color_palette()[2], "-"),
    }

    # Plot loss for each time point
    for loss_type in loss_types:  # train_loss or val_loss
        # Per-t plots
        for t_idx in range(num_times):
            t_val = ts_values[t_idx] if t_idx < len(ts_values) else t_idx
            plt.figure(figsize=(14, 7))
            sns.set_theme(style="darkgrid")
            plot_made_for_t = False

            # Process trained model losses
            if trained_models:
                trained_sizes = sorted(trained_models.keys())
                for cat in ["trained"]:
                    valid_sizes = [
                        size for size in trained_sizes
                        if loss_type in trained_models[size] and
                        t_idx < len(trained_models[size][loss_type]) and
                        cat in trained_models[size][loss_type][t_idx]
                    ]

                    if valid_sizes:
                        y_coords = [trained_models[size][loss_type][t_idx][cat] for size in valid_sizes]
                        color, style = line_styles.get(cat, (sns.color_palette()[data_categories.index(cat) % len(sns.color_palette())], "-"))
                        plt.plot(
                            valid_sizes,
                            y_coords,
                            marker="o",
                            color=color,
                            linestyle=style,
                            label=cat.replace("_", " ").title(),
                        )
                        plot_made_for_t = True

            # Process baseline losses
            if baseline_models:
                baseline_sizes = sorted(baseline_models.keys())
                for cat in ["ground_truth", "memorizing", "ground_truth_approx", "partial_mem", "partial_mem_approx"]:
                    valid_sizes = [
                        size for size in baseline_sizes
                        if loss_type in baseline_models[size] and
                        t_idx < len(baseline_models[size][loss_type]) and
                        cat in baseline_models[size][loss_type][t_idx]
                    ]

                    if valid_sizes:
                        # For baselines like ground_truth and memorizing, they usually have the same value across all sizes
                        # For a given t_idx, so we'll draw a horizontal line
                        first_val = baseline_models[valid_sizes[0]][loss_type][t_idx][cat]
                        if cat in ["ground_truth", "memorizing", "ground_truth_approx"]:
                            if not (loss_type == "val_loss" and cat == "ground_truth_approx"):
                                color, style = line_styles.get(cat, (sns.color_palette()[data_categories.index(cat) % len(sns.color_palette())], "-"))
                                plt.axhline(
                                    y=first_val.item(),
                                    color=color,
                                    linestyle=style,
                                    label=cat.replace("_", " ").title(),
                                )
                                plot_made_for_t = True
                        else:
                            # Values vary by size, plot a line
                            y_coords = [baseline_models[size][loss_type][t_idx][cat] for size in valid_sizes]
                            color, style = line_styles.get(cat, (sns.color_palette()[data_categories.index(cat) % len(sns.color_palette())], "-"))
                            plt.plot(
                                valid_sizes,
                                y_coords,
                                marker="o",
                                color=color,
                                linestyle=style,
                                label=cat.replace("_", " ").title(),
                            )
                            plot_made_for_t = True

            if plot_made_for_t:
                plt.xlabel("Number of Means")
                plt.ylabel(loss_type.replace("_", " ").title())
                plt.yscale("log")
                plt.title(f"{loss_type.replace('_', ' ').title()} vs Num Means (t={t_val:.3f})")
                plt.legend()
                plt.tight_layout()
                plot_dir_t = plot_dir / f"t_{t_val:.3f}".replace(".", "_")
                plot_dir_t.mkdir(parents=True, exist_ok=True)
                save_path = plot_dir_t / f"{loss_type}.png"
                plt.savefig(save_path)
                plt.close()
                if summary_logger:
                    summary_logger.log(
                        {
                            f"{loss_type}/t={t_val:.3f}": wandb.Image(
                                str(save_path.absolute())
                            )
                        },
                        step=0,
                    )
                plot_files.append(save_path)
            else:
                plt.close()  # Close the figure if nothing was plotted

        # Averaged plots (over t)
        plt.figure(figsize=(14, 7))
        sns.set_theme(style="darkgrid")
        plot_made_avg = False

        # Process trained model average losses
        if trained_models:
            trained_sizes = sorted(trained_models.keys())
            for cat in ["trained"]:
                valid_sizes = []
                avg_losses = []

                for size in trained_sizes:
                    if loss_type in trained_models[size]:
                        vals_over_t = []
                        for t_idx in range(num_times):
                            if t_idx < len(trained_models[size][loss_type]) and cat in trained_models[size][loss_type][t_idx] and trained_models[size][loss_type][t_idx][cat] is not None:
                                vals_over_t.append(trained_models[size][loss_type][t_idx][cat].item())

                        if vals_over_t:
                            valid_sizes.append(size)
                            avg_losses.append(jnp.mean(jnp.array(vals_over_t)))

                if valid_sizes:
                    color, style = line_styles.get(cat, (sns.color_palette()[data_categories.index(cat) % len(sns.color_palette())], "-"))
                    plt.plot(
                        valid_sizes,
                        avg_losses,
                        marker="o",
                        color=color,
                        linestyle=style,
                        label=cat.replace("_", " ").title(),
                    )
                    plot_made_avg = True

        # Process baseline average losses
        if baseline_models:
            baseline_sizes = sorted(baseline_models.keys())
            for cat in ["ground_truth", "memorizing", "ground_truth_approx", "partial_mem", "partial_mem_approx"]:
                valid_sizes = []
                avg_losses = []

                for size in baseline_sizes:
                    if loss_type in baseline_models[size]:
                        vals_over_t = []
                        for t_idx in range(num_times):
                            if t_idx < len(baseline_models[size][loss_type]) and cat in baseline_models[size][loss_type][t_idx] and baseline_models[size][loss_type][t_idx][cat] is not None:
                                vals_over_t.append(baseline_models[size][loss_type][t_idx][cat].item())

                        if vals_over_t:
                            valid_sizes.append(size)
                            avg_losses.append(jnp.mean(jnp.array(vals_over_t)))

                if valid_sizes:
                    # Check if all sizes have the same average value (for ground_truth, memorizing)
                    if cat in ["ground_truth", "memorizing", "ground_truth_approx"]:
                        if not (loss_type == "val_loss" and cat == "ground_truth_approx"):
                            color, style = line_styles.get(cat, (sns.color_palette()[data_categories.index(cat) % len(sns.color_palette())], "-"))
                            plt.axhline(
                                y=avg_losses[0].item(),
                                color=color,
                                linestyle=style,
                                label=cat.replace("_", " ").title(),
                            )
                            plot_made_avg = True
                    else:
                        color, style = line_styles.get(cat, (sns.color_palette()[data_categories.index(cat) % len(sns.color_palette())], "-"))
                        plt.plot(
                            valid_sizes,
                            avg_losses,
                            marker="o",
                            color=color,
                            linestyle=style,
                            label=cat.replace("_", " ").title(),
                        )
                        plot_made_avg = True

        if plot_made_avg:
            plt.xlabel("Number of Means")
            plt.ylabel(f"Average {loss_type.replace('_', ' ').title()} (over t)")
            plt.yscale("log")
            plt.title(f"Average {loss_type.replace('_', ' ').title()} vs Num Means")
            plt.legend()
            plt.tight_layout()
            save_path = plot_dir / f"avg_{loss_type}.png"
            plt.savefig(save_path)
            plt.close()
            if summary_logger:
                summary_logger.log(
                    {f"{loss_type}/avg": wandb.Image(str(save_path.absolute()))}, step=0
                )
            plot_files.append(save_path)
        else:
            plt.close()

    # Close wandb logger if it was created
    if summary_logger:
        summary_logger_dir = Path(summary_logger.dir)
        summary_logger.finish()
        shutil.rmtree(summary_logger_dir.parent)

    return plot_files


def main():
    """
    Main function to load results and generate plots.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument("--base-config-file", type=str, required=True, help="Path to the base config file")
    parser.add_argument("--result-file", type=str, required=True, help="Path to the aggregated result file")
    parser.add_argument("--data-file", type=str, help="Path to the data file")
    parser.add_argument("--output-dir", type=str, help="Directory to save plots (defaults to same directory as result file)")
    parser.add_argument("--use-wandb", action="store_true", help="Whether to log results to wandb")

    args = parser.parse_args()

    # Generate plots
    log_and_plot_aggregated_metrics(Path(args.base_config_file), Path(args.data_file), Path(args.result_file), Path(args.output_dir), args.use_wandb)

    print(f"Plots generated and saved to {Path(args.output_dir) / 'plots'}")


if __name__ == "__main__":
    main()
