"""Helper functions for the Encode Fellowship Demo notebook.

Keeps the notebook cells clean by moving data loading, MLflow queries,
model loading, inference, and plotting into reusable functions.
"""

import glob
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dotenv import load_dotenv

from neurodecoders.data.loading import (
    DatasetConfig,
    compute_and_apply_normalization,
    load_synthetic_split_data,
)
from neurodecoders.input_optim.optimizer import ImageOptimizer, OptimConfig
from neurodecoders.paths import get_synthetic_data_path

# ── Colour palette (consistent across all plots) ───────────────────────────
COLORS = {
    "Input Optim": "#d9534f",
    "Guided Decoder": "#2e8b57",
    "DDIM Diffusion": "#1e90ff",
    "Pure Decoder": "#f0ad4e",
}


# ── Project root (.env lives next to config.yaml) ──────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Setup ───────────────────────────────────────────────────────────────────
def setup_environment():
    """Connect to MLflow and configure plotting style. Returns tracking URI."""
    plt.style.use("dark_background")
    sns.set_context("talk")

    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI not set in .env")
    mlflow.set_tracking_uri(tracking_uri)


# ── Data loading ────────────────────────────────────────────────────────────
def get_validation_dataset_path(n_neurons=10, n_images=80):
    """Find the small synthetic dataset used for validation visualisation.

    Searches under the configured synthetic data path (from config.yaml).
    """
    synth_dir = os.path.join(get_synthetic_data_path(), "train")
    pattern = os.path.join(
        synth_dir,
        f"synthdata_dataset-cifar10_sta-gabor*_n_neurons-{n_neurons}_n_images-{n_images}_*.npz",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No validation dataset found matching {pattern}"
        )
    return matches[-1]  # most recent by filename


def load_demo_data(n_neurons=5000, n_train=8000, n_test=2000):
    """Load and normalize synthetic CIFAR-10 test data.

    Returns (test_images_norm, test_firing_norm).
    """
    config = DatasetConfig(
        n_neurons=n_neurons,
        n_train_images=n_train,
        n_test_images=n_test,
        dataset_type="cifar10",
        sta_type="gabor,11,11",
    )
    train_img, train_fire, _, _ = load_synthetic_split_data(config, split="train")
    test_img, test_fire, _, _ = load_synthetic_split_data(config, split="test")
    _, _, test_img_norm, test_fire_norm, _ = compute_and_apply_normalization(
        train_img, train_fire, test_img, test_fire
    )
    return test_img_norm, test_fire_norm


# ── Run selection ───────────────────────────────────────────────────────────
def select_best_runs(
    arch_experiment="architecture_comparison",
    exclude_transformer_guided=True,
):
    """Pick the best MLflow run per architecture by neural similarity.

    Parameters
    ----------
    exclude_transformer_guided : bool
        If True, guided decoder runs that used a transformer backbone are
        excluded so only CNN-based guided decoders are considered.

    Returns dict  {display_name: pandas.Series | None}.
    """
    runs = mlflow.search_runs(experiment_names=[arch_experiment])
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{arch_experiment}'")
    df = runs[runs["params.mode"].notna()].copy()

    def _best(mask, metric="metrics.neural_consistency_mean", ascending=False):
        sub = df.loc[mask].dropna(subset=[metric])
        return sub.sort_values(metric, ascending=ascending).iloc[0] if len(sub) else None

    input_optim = _best(df["params.mode"] == "input_optim")

    guided_mask = df["params.mode"] == "guided"
    if exclude_transformer_guided and "params.guided_decoder_type" in df.columns:
        guided_mask = guided_mask & (
            df["params.guided_decoder_type"].fillna("") != "transformer"
        )
    guided = _best(guided_mask)

    diffusion = _best(df["params.mode"].str.contains("diffusion", na=False))
    pure = _best(df["params.mode"].fillna("") == "pure_decoder")

    selected = {
        "Input Optim": input_optim,
        "Guided Decoder": guided,
        "DDIM Diffusion": diffusion,
        "Pure Decoder": pure,
    }
    return selected


def print_selected_runs(selected):
    """Print a compact summary table of the selected runs."""
    for name, run in selected.items():
        if run is None:
            print(f"  {name}: not found")
            continue
        nc = run.get("metrics.neural_consistency_mean", float("nan"))
        print(f"  {name}: neural_sim={nc:.4f}")


# ── Time formatting ─────────────────────────────────────────────────────────
def format_time(seconds):
    """Format seconds into the most natural unit (hours / minutes / seconds)."""
    if seconds is None or np.isnan(seconds):
        return "N/A"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}min"
    return f"{seconds:.1f}s"


# ── Model loading ───────────────────────────────────────────────────────────
def load_models(selected_runs, device=None):
    """Load model artifacts from MLflow for each selected run.

    Returns (models_dict, encoder) where encoder is extracted from the
    guided decoder model (needed for input optimisation).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for name, run in selected_runs.items():
        if run is None:
            continue
        run_id = run.get("run_id")
        if not run_id:
            continue
        try:
            model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
            model.to(device).eval()
            models[name] = model
        except Exception:
            pass

    # Extract the encoder from the guided model for input optimisation
    encoder = None
    guided = models.get("Guided Decoder")
    if guided is not None and hasattr(guided, "encoder"):
        encoder = guided.encoder
        encoder.eval()

    return models, encoder


# ── Reconstruction ──────────────────────────────────────────────────────────
def reconstruct_samples(
    models,
    encoder,
    test_firing_norm,
    test_images_norm,
    sample_indices,
    input_optim_steps=50,
    diffusion_steps=20,
    device=None,
):
    """Run live inference for all loaded architectures on selected samples.

    Returns (results, originals) where results is
    {architecture_name: [img_array, ...]} and originals is a list of arrays.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    originals = []
    results = {name: [] for name in models}

    for i, idx in enumerate(sample_indices):
        originals.append(test_images_norm[idx].squeeze())
        target = test_firing_norm[idx]
        t_fire = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            if "Guided Decoder" in models:
                img = models["Guided Decoder"](t_fire, return_images=True)
                results["Guided Decoder"].append(img.cpu().squeeze().numpy())

            if "DDIM Diffusion" in models:
                img = models["DDIM Diffusion"].sample(
                    t_fire, num_inference_steps=diffusion_steps, sampler="ddim"
                )
                results["DDIM Diffusion"].append(img.cpu().squeeze().numpy())

            if "Pure Decoder" in models:
                m = models["Pure Decoder"]
                base = m.model if hasattr(m, "model") else m
                img = base(t_fire).detach()
                results["Pure Decoder"].append(img.cpu().squeeze().numpy())

        # Input optimisation requires gradients
        if encoder is not None:
            cfg = OptimConfig(
                image_size=32,
                channels=1,
                steps=input_optim_steps,
                lr=0.1,
                log_every=10000,
                loss="poisson_mean",
            )
            opt = ImageOptimizer(
                encoder=encoder,
                target_rates=target.astype(np.float32),
                config=cfg,
            )
            with torch.enable_grad():
                img, _ = opt.optimize()
            results.setdefault("Input Optim", []).append(img.squeeze())

    return results, originals


# ── Plotting: validation environment ────────────────────────────────────────
def plot_validation_environment(dataset_path, neuron_id=0, image_idx=1):
    """Plot stimulus with receptive-field box and gabor at matched size."""
    data = np.load(dataset_path)
    image = data["images"][image_idx][0, :, :]
    sta = data["stas"][neuron_id]
    rf_coords = data["rf_coords"][neuron_id]
    sta_shape = sta.shape[0]
    img_size = image.shape[0]

    # Make the gabor subplot the same physical width as the RF box
    ratio = img_size / sta_shape
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7, 3),
        gridspec_kw={"width_ratios": [ratio, 1], "wspace": 0.4},
    )

    # Left: stimulus + RF bounding box
    ax = axes[0]
    ax.imshow(image, cmap="gray")
    rect = plt.Rectangle(
        (rf_coords[0], rf_coords[1]),
        sta_shape,
        sta_shape,
        linewidth=1.5,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        rf_coords[0] + sta_shape / 2,
        rf_coords[1] - 1,
        "Receptive Field",
        color="red",
        fontsize=8,
        ha="center",
        va="bottom",
    )
    ax.set_title("Environmental Stimulus (CIFAR-10)", fontsize=10)
    ax.axis("off")

    # Right: gabor filter
    ax = axes[1]
    ax.imshow(sta, cmap="gray")
    ax.set_title("Simulated Neuron\nResponsiveness Function", fontsize=10)
    ax.axis("off")

    fig.suptitle(
        "Validation Environment: Single Neuron Analysis", fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.show()


# ── Plotting: firing rate traces ────────────────────────────────────────────
def plot_firing_rates(dataset_path, n_neurons=20, n_frames=200):
    """Plot stacked firing-rate traces for multiple neurons."""
    data = np.load(dataset_path)
    n_neurons = min(n_neurons, data["responses"].shape[1])
    n_frames = min(n_frames, data["responses"].shape[0])
    rates = data["responses"][:n_frames, :n_neurons]

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = plt.cm.tab20(np.linspace(0, 1, n_neurons))
    shift = np.max(rates) - np.min(rates) + 1

    for i in range(n_neurons):
        ax.plot(rates[:, i] + i * shift, color=palette[i], linewidth=0.8, alpha=0.85)

    ax.set_title(
        f"Simulated Firing Rates: {n_neurons} Neurons \u00d7 {n_frames} Frames",
        fontsize=11,
    )
    ax.set_xlabel("Frame", fontsize=9)
    ax.set_ylabel("Neuron (stacked)", fontsize=9)
    ax.set_yticks([i * shift for i in range(n_neurons)])
    ax.set_yticklabels([f"N{i}" for i in range(n_neurons)], fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


# ── Plotting: inference time ─────────────────────────────────────────────────
def plot_inference_time(selected_runs):
    """Inference time bar chart with human-readable time labels."""
    names, vals, cols = [], [], []
    for name, run in selected_runs.items():
        if run is None:
            continue
        v = run.get("metrics.inference_time_seconds")
        if pd.notna(v):
            names.append(name)
            vals.append(float(v))
            cols.append(COLORS.get(name, "#888"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, vals, color=cols, width=0.6)
    ax.set_yscale("log")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            format_time(v),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Seconds (log scale)", fontsize=9)
    ax.set_title("Inference Time (2000 images)", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    plt.show()


# ── Plotting: reconstructions grid ──────────────────────────────────────────
def plot_reconstructions(originals, results, sample_indices):
    """Side-by-side grid: original vs each architecture's reconstruction."""
    arch_order = ["Input Optim", "Guided Decoder", "DDIM Diffusion", "Pure Decoder"]
    available = [a for a in arch_order if a in results and results[a]]
    titles = ["Original"] + available

    n_rows = len(sample_indices)
    n_cols = len(titles)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axs = axs[np.newaxis, :]

    for i in range(n_rows):
        for j, title in enumerate(titles):
            ax = axs[i, j]
            if title == "Original":
                img = originals[i]
            else:
                img = results[title][i]

            img = np.squeeze(img)
            if img.ndim != 2:
                img = np.zeros((32, 32))

            ax.imshow(img, cmap="gray")
            if i == 0:
                ax.set_title(title, fontsize=11, pad=10)
            ax.axis("off")

    fig.suptitle(
        "Neural Decoding: Test Set Reconstructions",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.show()


# ── GIF export ──────────────────────────────────────────────────────────────
def save_reconstruction_gifs(
    models,
    encoder,
    test_firing_norm,
    test_images_norm,
    n_images=10,
    output_dir=".",
    duration_ms=500,
    input_optim_steps=50,
    diffusion_steps=20,
    device=None,
):
    """Generate two GIFs: one of original stimuli, one of reconstructions.

    Each frame is shown for *duration_ms* milliseconds (default 500 ms).
    Only the first available decoder architecture is used for the
    reconstruction GIF, preferring Guided > DDIM > Pure > Input Optim.

    Returns the two file paths (originals_path, reconstructions_path).
    """
    from PIL import Image

    indices = list(range(n_images))
    results, originals = reconstruct_samples(
        models,
        encoder,
        test_firing_norm,
        test_images_norm,
        indices,
        input_optim_steps=input_optim_steps,
        diffusion_steps=diffusion_steps,
        device=device,
    )

    # Pick the first available architecture for the reconstruction GIF
    arch_priority = ["Guided Decoder", "DDIM Diffusion", "Pure Decoder", "Input Optim"]
    chosen_arch = None
    for arch in arch_priority:
        if arch in results and results[arch]:
            chosen_arch = arch
            break
    if chosen_arch is None:
        raise RuntimeError("No reconstructions available – are any models loaded?")

    def _to_pil_frames(arrays):
        """Convert list of 2-D numpy arrays to 8-bit PIL Image frames."""
        frames = []
        for arr in arrays:
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                arr = np.zeros((32, 32))
            lo, hi = arr.min(), arr.max()
            if hi - lo > 0:
                arr = (arr - lo) / (hi - lo) * 255
            else:
                arr = np.zeros_like(arr)
            frames.append(Image.fromarray(arr.astype(np.uint8), mode="L"))
        return frames

    os.makedirs(output_dir, exist_ok=True)

    orig_frames = _to_pil_frames(originals)
    recon_frames = _to_pil_frames(results[chosen_arch])

    orig_path = os.path.join(output_dir, "originals.gif")
    recon_path = os.path.join(output_dir, "reconstructions.gif")

    orig_frames[0].save(
        orig_path,
        save_all=True,
        append_images=orig_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    recon_frames[0].save(
        recon_path,
        save_all=True,
        append_images=recon_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return orig_path, recon_path
