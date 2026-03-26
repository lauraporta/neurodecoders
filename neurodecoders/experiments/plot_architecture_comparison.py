#!/usr/bin/env python3
"""
Plot architecture comparison results from MLflow.

Fetches metrics from:
  - experiment 'architecture_comparison'  (exp_id=144): pixel + neural metrics
  - experiment 'cifar10_decoding_sweep_normalized' (exp_id=137): pure decoder baseline

Usage:
    python -m neurodecoders.experiments.plot_architecture_comparison
    python -m neurodecoders.experiments.plot_architecture_comparison --output /path/to/plot.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient

from neurodecoders.config import get_mlflow_tracking_uri


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARCH_COMP_EXP_NAME = "architecture_comparison"
PURE_DECODER_EXP_NAME = "cifar10_decoding_sweep_normalized"

# Display names for the architecture_comparison modes
MODE_LABELS = {
    "diffusion":      "Diffusion\n(DDPM)",
    "diffusion_ddim": "Diffusion\n(DDIM)",
    "guided":         "Encoder-Guided\nDecoder",
    "input_optim":    "Input\nOptim",
}

# Colours per method (consistent across all plots)
COLORS = {
    "diffusion":      "#2196F3",   # blue
    "diffusion_ddim": "#03A9F4",   # light blue
    "guided":         "#4CAF50",   # green
    "input_optim":    "#FF9800",   # orange
    "pure_decoder":   "#9C27B0",   # purple
}

PURE_DECODER_LABEL = "Pure Decoder\n(image-only)"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _get_latest_arch_runs(client: MlflowClient, exp_name: str) -> dict:
    """
    Return the most recent complete set of runs from architecture_comparison.
    Picks the timestamp suffix that appears the most times among FINISHED runs
    so we always get a consistent batch.
    """
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise ValueError(f"Experiment '{exp_name}' not found in MLflow.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
    )

    # Group by mode param; prefer the most recent run per mode
    by_mode: dict[str, mlflow.entities.Run] = {}
    for run in runs:
        mode = run.data.params.get("mode")
        if mode and mode not in by_mode:
            by_mode[mode] = run

    return by_mode


def _get_best_pure_decoder_run(client: MlflowClient, exp_name: str):
    """
    Return the best FINISHED run from cifar10_decoding_sweep_normalized
    (lowest final_test_loss), with model_type=diffusion.
    """
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise ValueError(f"Experiment '{exp_name}' not found in MLflow.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED' and params.model_type = 'diffusion'",
        order_by=["metrics.final_test_loss ASC"],
        max_results=1,
    )
    if not runs:
        return None
    return runs[0]


def fetch_all_metrics(client: MlflowClient) -> dict:
    """
    Fetch metrics for all methods and return as a nested dict:
        {method_key: {metric_name: value, ...}}
    """
    arch_runs = _get_latest_arch_runs(client, ARCH_COMP_EXP_NAME)
    pure_run = _get_best_pure_decoder_run(client, PURE_DECODER_EXP_NAME)

    data = {}

    for mode, run in arch_runs.items():
        m = run.data.metrics
        data[mode] = {
            "pixel_correlation_mean": m.get("pixel_correlation_mean"),
            "pixel_correlation_std":  m.get("pixel_correlation_std"),
            "ssim_mean":              m.get("ssim_mean"),
            "ssim_std":               m.get("ssim_std"),
            "neural_consistency_mean": m.get("neural_consistency_mean"),
            "neural_consistency_std":  m.get("neural_consistency_std"),
            "mse_mean":               m.get("mse_mean"),
            "mse_std":                m.get("mse_std"),
            "training_loss":          m.get("train_loss"),
            "run_name":               run.info.run_name,
        }

    if pure_run is not None:
        m = pure_run.data.metrics
        data["pure_decoder"] = {
            # pixel / neural metrics were not computed for this experiment
            "pixel_correlation_mean": None,
            "pixel_correlation_std":  None,
            "ssim_mean":              None,
            "ssim_std":               None,
            "neural_consistency_mean": None,
            "neural_consistency_std":  None,
            # use test_loss as best available pixel-loss proxy
            "mse_mean":               m.get("final_test_loss") or m.get("test_loss"),
            "mse_std":                None,
            "training_loss":          m.get("train_loss"),
            "run_name":               pure_run.info.run_name,
        }

    return data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _method_order(data: dict) -> list[str]:
    """Preferred display order."""
    order = ["diffusion", "diffusion_ddim", "guided", "input_optim", "pure_decoder"]
    return [k for k in order if k in data]


def _label(method: str) -> str:
    if method == "pure_decoder":
        return PURE_DECODER_LABEL
    return MODE_LABELS.get(method, method)


def _color(method: str) -> str:
    return COLORS.get(method, "#888888")


def _bar_group(
    ax,
    methods: list[str],
    values: list,
    errors: list,
    ylabel: str,
    title: str,
    ylim: tuple | None = None,
    hatch_missing: bool = True,
    log_scale: bool = False,
):
    """Draw a single group of bars on ax."""
    x = np.arange(len(methods))

    plot_values = [v if v is not None else 0 for v in values]
    plot_errors = [e if e is not None else 0 for e in errors]

    bars = ax.bar(
        x,
        plot_values,
        yerr=plot_errors,
        capsize=4,
        color=[_color(m) for m in methods],
        error_kw={"linewidth": 1.0, "ecolor": "black"},
        width=0.6,
    )

    # Hatch bars with no data
    if hatch_missing:
        for bar, v in zip(bars, values):
            if v is None:
                bar.set_hatch("///")
                bar.set_edgecolor("grey")
                bar.set_alpha(0.35)

    # Value annotations – position above bar (or at fixed y for log scale)
    for bar, v in zip(bars, values):
        if v is not None:
            y_pos = bar.get_height() * 1.08 if log_scale else bar.get_height() + 0.01
            fmt = f"{v:.2e}" if (abs(v) > 100 or (abs(v) < 0.001 and v != 0)) else f"{v:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                fmt,
                ha="center",
                va="bottom",
                fontsize=7,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.02,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=7,
                color="grey",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_label(m) for m in methods],
        fontsize=7.5,
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if log_scale:
        ax.set_yscale("log")
    elif ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not log_scale:
        ax.axhline(0, color="black", linewidth=0.5)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def create_comparison_plot(data: dict, output_path: str):
    """Create a 4-panel comparison figure."""
    methods = _method_order(data)

    pixel_corr  = [data[m]["pixel_correlation_mean"] for m in methods]
    pixel_std   = [data[m]["pixel_correlation_std"]  for m in methods]
    ssim_vals   = [data[m]["ssim_mean"]              for m in methods]
    ssim_std    = [data[m]["ssim_std"]               for m in methods]
    neural_vals = [data[m]["neural_consistency_mean"] for m in methods]
    neural_std  = [data[m]["neural_consistency_std"]  for m in methods]
    mse_vals    = [data[m]["mse_mean"]               for m in methods]
    mse_std     = [data[m]["mse_std"]                for m in methods]

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.5))
    fig.suptitle("Architecture Comparison", fontsize=13, fontweight="bold", y=1.02)

    _bar_group(
        axes[0], methods, pixel_corr, pixel_std,
        ylabel="Pearson r",
        title="Pixel Correlation",
        ylim=(-0.15, 1.0),
    )
    _bar_group(
        axes[1], methods, ssim_vals, ssim_std,
        ylabel="SSIM",
        title="Structural Similarity\n(SSIM)",
        ylim=(0, 0.55),
    )
    _bar_group(
        axes[2], methods, neural_vals, neural_std,
        ylabel="Pearson r",
        title="Neural Consistency\n(re-stimulation)",
        ylim=(0, 0.75),
    )
    _bar_group(
        axes[3], methods, mse_vals, mse_std,
        ylabel="MSE / Test Loss",
        title="Pixel Loss (MSE)\n[lower = better, log scale]",
        log_scale=True,
    )

    # Legend for pure decoder hatch
    legend_patch = mpatches.Patch(
        facecolor="lightgrey",
        hatch="///",
        edgecolor="grey",
        alpha=0.5,
        label="Not computed for this experiment",
    )
    fig.legend(
        handles=[legend_patch],
        loc="lower center",
        ncol=1,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot architecture comparison from MLflow")
    parser.add_argument(
        "--output", type=str,
        default="architecture_comparison_plot.png",
        help="Output path for the comparison plot",
    )
    parser.add_argument(
        "--tracking-uri", type=str, default=None,
        help="MLflow tracking URI (defaults to env/config value)",
    )
    args = parser.parse_args()

    tracking_uri = args.tracking_uri or get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    print(f"Fetching data from MLflow at: {tracking_uri}")

    data = fetch_all_metrics(client)

    print("\nFetched methods:")
    for method, metrics in data.items():
        print(f"  {method} (run: {metrics['run_name']})")
        for k, v in metrics.items():
            if k != "run_name" and v is not None:
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    create_comparison_plot(data, args.output)


if __name__ == "__main__":
    main()
