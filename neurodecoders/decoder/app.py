import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from skimage.transform import resize

# Add the decoder directory to the path so we can import from it
sys.path.append(os.path.dirname(__file__))

# Import the decoder functionality
from decoder import (
    DecoderLightningModule,
    save_predictions,
    train_model_lightning,
)

# Configure Streamlit page
st.set_page_config(
    page_title="Neural Decoder Dashboard",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create workspace directories if they don't exist
os.makedirs("workspace/datasets/synthetic", exist_ok=True)
os.makedirs("workspace/models", exist_ok=True)
os.makedirs("workspace/predictions", exist_ok=True)


def load_data():
    """Load neural data file selected by user from dropdown"""
    try:
        # Find all synthdata files
        files = glob.glob("workspace/datasets/synthetic/synthdata_dataset-*.npz")
        if not files:
            st.error(
                "No neural data files found in workspace/datasets/synthetic/ directory. Please generate data first using the synthetic dashboard."
            )
            return None, None, None

        # Create a mapping of display names to file paths
        file_options = {}
        for file_path in files:
            # Extract meaningful info from filename for display
            filename = os.path.basename(file_path)
            # Remove the synthdata_dataset- prefix and .npz suffix
            display_name = filename.replace("synthdata_dataset-", "").replace(
                ".npz", ""
            )
            # Replace underscores with spaces for better readability
            display_name = display_name.replace("_", " ")
            file_options[display_name] = file_path

        # Sort by creation time (newest first) for the dropdown
        sorted_files = sorted(
            file_options.items(),
            key=lambda x: os.path.getctime(x[1]),
            reverse=True,
        )

        # Create dropdown
        selected_display_name = st.selectbox(
            "Select Dataset:",
            options=[name for name, _ in sorted_files],
            index=0,  # Default to newest file
            help="Choose a synthetic dataset to load",
        )

        # Get the selected file path
        selected_file = file_options[selected_display_name]

        data = np.load(selected_file)
        images = data["images"]
        firing_rates = data["responses"]

        return images, firing_rates, selected_file
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None


def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train Loss", linewidth=2)
    ax.plot(val_losses, label="Validation Loss", linewidth=2)

    # Add horizontal dashed gray line at poor/good threshold (MSE = 0.01)
    ax.axhline(
        y=0.01,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Poor/Good Threshold",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    min_loss = (
        min(min(train_losses), min(val_losses))
        if train_losses and val_losses
        else 0.001
    )
    max_loss = (
        max(max(train_losses), max(val_losses))
        if train_losses and val_losses
        else 0.05
    )
    ax.set_ylim(
        max(min_loss * 0.5, 0.0001), max_loss * 1.5
    )  # Ensure threshold line is visible

    plt.tight_layout()
    return fig


def plot_psnr_curves(train_psnr, val_psnr):
    """Plot training and validation PSNR curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_psnr, label="Train PSNR", linewidth=2)
    ax.plot(val_psnr, label="Validation PSNR", linewidth=2)

    # Add horizontal dashed gray line at poor/good threshold (PSNR = 25 dB)
    ax.axhline(
        y=25,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Poor/Good Threshold",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Training and Validation PSNR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis to log scale and show full spectrum from 1 to 100 dB
    ax.set_yscale("log")
    ax.set_ylim(1, 100)

    plt.tight_layout()
    return fig


def plot_correlation_curves(train_correlation, val_correlation):
    """Plot training and validation correlation curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_correlation, label="Train Correlation", linewidth=2)
    ax.plot(val_correlation, label="Validation Correlation", linewidth=2)

    # Add horizontal dashed gray line at poor/good threshold (correlation = 0.5)
    ax.axhline(
        y=0.5,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Poor/Good Threshold",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation")
    ax.set_title("Training and Validation Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis to show full spectrum from -1 to 1
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    return fig


def plot_firing_rate_distribution(firing_rates):
    """Plot firing rate distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    ax1.hist(
        firing_rates.flatten(),
        bins=50,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    ax1.set_title("Firing Rate Distribution")
    ax1.set_xlabel("Firing Rate (Hz)")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)

    # Heatmap
    im = ax2.imshow(firing_rates[:100].T, aspect="auto", cmap="viridis")
    ax2.set_title("Firing Rates for First 100 Images")
    ax2.set_xlabel("Image Index")
    ax2.set_ylabel("Neuron Index")
    plt.colorbar(im, ax=ax2, label="Firing Rate (Hz)")

    plt.tight_layout()
    return fig


def plot_reconstructions(original_images, reconstructed_images, n_samples=10):
    """Plot original vs reconstructed images"""
    n_samples = min(n_samples, len(original_images))
    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 4))

    for i in range(n_samples):
        # Original image
        np.max(original_images[i])
        np.min(original_images[i])
        axes[0, i].imshow(original_images[i], cmap="gray", vmin=-1, vmax=1)
        axes[0, i].set_title(f"Original {i + 1}")
        axes[0, i].axis("off")

        # Reconstructed image
        np.max(reconstructed_images[i])
        np.min(reconstructed_images[i])
        axes[1, i].imshow(
            reconstructed_images[i], cmap="gray", vmin=-1, vmax=1
        )
        axes[1, i].set_title(f"Reconstructed {i + 1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    return fig


def load_trained_model(model_path):
    """Load a trained model and return it for inference"""
    try:
        # Load the saved state dict first to inspect the model architecture
        state_dict = torch.load(model_path, map_location=device)

        # Determine the number of input neurons from the saved model
        # Look for the first layer weights (fc.0.weight)
        if "fc.0.weight" in state_dict:
            saved_in_neurons = state_dict["fc.0.weight"].shape[1]
        elif "model.fc.0.weight" in state_dict:
            saved_in_neurons = state_dict["model.fc.0.weight"].shape[1]
        else:
            raise ValueError(
                "Could not determine model architecture from saved weights"
            )

        # Try to determine image size from the model filename or use default
        model_name = os.path.basename(model_path)
        image_size = 64  # Default size

        # Extract dataset info from model filename to determine image size
        if "synthdata_dataset-" in model_name:
            dataset_name = model_name.replace("decoder_", "").replace(
                ".pth", ""
            )
            dataset_file = f"data/{dataset_name}.npz"

            if os.path.exists(dataset_file):
                # Load a sample to determine the correct image size
                sample_data = np.load(dataset_file)
                sample_images = sample_data["images"]
                if sample_images.ndim == 4:
                    image_size = sample_images.shape[2]  # Height
                elif sample_images.ndim == 3:
                    image_size = sample_images.shape[1]  # Height

        st.info(
            f"Loading model with {saved_in_neurons} input neurons and {image_size}x{image_size} output size"
        )

        # Initialize model with the correct parameters from the saved model
        model = DecoderLightningModule(
            in_neurons=saved_in_neurons, image_size=image_size
        )

        # Handle state dict key mapping
        # The saved model might have direct keys (fc.0.weight) or prefixed keys (model.fc.0.weight)
        # The current DecoderLightningModule expects prefixed keys (model.fc.0.weight)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                # Already has the correct prefix
                new_state_dict[key] = value
            else:
                # Add the model prefix
                new_state_dict[f"model.{key}"] = value

        # Load trained weights
        model.load_state_dict(new_state_dict)
        model.eval()
        model.to(device)

        # Verify model is on correct device
        st.info(f"Model loaded on device: {next(model.parameters()).device}")

        return model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def main():
    st.title("🧠 Neural Decoder Dashboard")
    st.markdown(
        "Train a decoder to reconstruct images from neural responses using PyTorch Lightning"
    )

    # Display detailed device information
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Device in use: {device.type.upper()}")
    with col2:
        if device.type == "cuda":
            st.info(f"GPU: {torch.cuda.get_device_name(0)}")
        elif device.type == "mps":
            st.info("GPU: Apple Silicon")
        else:
            st.info("GPU: CPU (no GPU available)")
    # Load data first - moved to beginning of function
    st.sidebar.header("Training configuration")

    # Load data
    images, firing_rates, input_file = load_data()

    if images is None or firing_rates is None:
        st.stop()

    # Display basic data info in sidebar
    st.sidebar.subheader("📊 Data Info")
    st.sidebar.write(f"**Images:** {len(images)}")
    st.sidebar.write(f"**Neurons:** {firing_rates.shape[1]}")
    st.sidebar.write(f"**Image Size:** {images.shape[-2]}x{images.shape[-1]}")

    # Training parameters in sidebar
    st.sidebar.subheader("🚀 Training Parameters")

    train_split = st.sidebar.slider("Training split", 0.5, 0.9, 0.7, 0.05)
    val_split = st.sidebar.slider("Validation split", 0.1, 0.3, 0.15, 0.05)
    batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64, 128], index=1)
    learning_rate = st.sidebar.selectbox(
        "Learning rate",
        [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e-0, 5e-0],
        index=0,
    )
    epochs = st.sidebar.slider("Epochs", 10, 100, 30)

    # Training section
    st.header("🚀 Train Decoder with PyTorch Lightning")

    # Train button
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training decoder model..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create metrics display
            col1, col2, col3 = st.columns(3)
            train_metric = col1.metric("Train Loss", "0.0000")
            val_metric = col2.metric("Val Loss", "0.0000")
            best_metric = col3.metric("Best Val Loss", "0.0000")

            # Create placeholder for loss plot
            loss_plot_placeholder = st.empty()

            # Create placeholders for PSNR and correlation plots
            psnr_plot_placeholder = st.empty()
            correlation_plot_placeholder = st.empty()

            def update_progress(progress, status):
                progress_bar.progress(progress)
                status_text.text(status)

            def update_metrics(train_loss, val_loss, best_val_loss):
                train_metric.metric("Train Loss", f"{train_loss:.4f}")
                val_metric.metric("Val Loss", f"{val_loss:.4f}")
                best_metric.metric("Best Val Loss", f"{best_val_loss:.4f}")

            def update_plot(train_losses, val_losses):
                if (
                    len(train_losses) > 1
                ):  # Only plot if we have at least 2 points
                    fig = plot_training_curves(train_losses, val_losses)
                    loss_plot_placeholder.pyplot(fig)
                    plt.close(fig)  # Close to prevent memory issues

            def update_psnr_plot(train_psnr, val_psnr):
                if (
                    len(train_psnr) > 1
                ):  # Only plot if we have at least 2 points
                    fig = plot_psnr_curves(train_psnr, val_psnr)
                    psnr_plot_placeholder.pyplot(fig)
                    plt.close(fig)  # Close to prevent memory issues

            def update_correlation_plot(train_correlation, val_correlation):
                if (
                    len(train_correlation) > 1
                ):  # Only plot if we have at least 2 points
                    fig = plot_correlation_curves(
                        train_correlation, val_correlation
                    )
                    correlation_plot_placeholder.pyplot(fig)
                    plt.close(fig)  # Close to prevent memory issues

            # Train the model using the functionality from decoder.py
            try:
                # Handle 4D image input if present
                if images.ndim == 4:
                    N, _, H, W = images.shape
                    images_processed = images[:, 0, :, :]  # Take first channel
                elif images.ndim == 3:
                    N, H, W = images.shape
                    images_processed = images
                else:
                    raise ValueError(f"Unexpected image shape: {images.shape}")

                # Train using the function from decoder.py
                trainer, model, data_module = train_model_lightning(
                    firing_rates=firing_rates,
                    images=images_processed,
                    train_split=train_split,
                    val_split=val_split,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    enable_progress_bar=False,
                    progress_callback=update_progress,
                    metrics_callback=update_metrics,
                    plot_callback=update_plot,
                    psnr_callback=update_psnr_plot,
                    correlation_callback=update_correlation_plot,
                )

                # Training completed
                st.success("✅ Training completed successfully!")

                # Display final metrics
                st.subheader("📊 Final Results")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Final Train Loss", f"{model.train_losses[-1]:.4f}"
                    )
                with col2:
                    st.metric("Final Val Loss", f"{model.val_losses[-1]:.4f}")
                with col3:
                    st.metric("Best Val Loss", f"{min(model.val_losses):.4f}")
                with col4:
                    st.metric("Training Epochs", len(model.train_losses))

                # Save the model
                dataset_path = Path(input_file)
                model_path = f"workspace/models/decoder_{dataset_path.stem}.pth"
                torch.save(model.state_dict(), model_path)
                st.success(f"✅ Model saved to: {model_path}")

                # Save predictions
                save_predictions(
                    model,
                    firing_rates,
                    images,
                    input_file,
                    dataset_to_load=dataset_path,
                )

                # Get test predictions for display
                model.eval()
                test_predictions = []
                test_actuals = []

                with torch.no_grad():
                    for batch in data_module.test_dataloader():
                        x, y = batch
                        # Ensure tensors are on the correct device
                        if x.device != device:
                            x = x.to(device)
                        if y.device != device:
                            y = y.to(device)
                        pred = model(x)
                        test_predictions.append(pred.cpu().numpy())
                        test_actuals.append(y.cpu().numpy())

                test_predictions = np.concatenate(test_predictions, axis=0)
                test_actuals = np.concatenate(test_actuals, axis=0)

                # Display some reconstructions
                st.subheader("🖼️ Sample Reconstructions")

                # Ensure proper shapes for visualization
                if test_actuals.ndim == 4:
                    test_actuals = test_actuals.squeeze(1)
                if test_predictions.ndim == 4:
                    test_predictions = test_predictions.squeeze(1)

                fig = plot_reconstructions(
                    test_actuals[:10], test_predictions[:10]
                )
                st.pyplot(fig)

                # Calculate reconstruction quality metrics
                st.subheader("📊 Reconstruction Quality")
                col1, col2, col3 = st.columns(3)

                # MSE between original and reconstructed
                mse = np.mean((test_actuals - test_predictions) ** 2)
                with col1:
                    st.metric("MSE", f"{mse:.4f}")

                # Calculate PSNR and correlation per image
                n_images = test_actuals.shape[0]
                psnr_values = []
                correlation_values = []

                for i in range(n_images):
                    pred_img = test_predictions[i]
                    target_img = test_actuals[i]

                    # Calculate MSE for this image
                    img_mse = np.mean((target_img - pred_img) ** 2)

                    # Calculate PSNR for this image
                    max_val = np.max(target_img)
                    if img_mse > 0:
                        psnr = 20 * np.log10(max_val / np.sqrt(img_mse))
                    else:
                        psnr = float("inf")  # Perfect reconstruction
                    psnr_values.append(psnr)

                    # Calculate correlation for this image
                    correlation = np.corrcoef(
                        target_img.flatten(), pred_img.flatten()
                    )[0, 1]
                    correlation_values.append(correlation)

                # Average the metrics across all images
                mean_psnr = np.mean(psnr_values)
                mean_correlation = np.mean(correlation_values)

                with col2:
                    st.metric("PSNR (dB)", f"{mean_psnr:.2f}")

                with col3:
                    st.metric("Correlation", f"{mean_correlation:.3f}")

            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)

    # Main content area
    st.header("📁 Load Trained Model")

    # Check for available model files - both old and new Lightning models
    model_files = glob.glob("workspace/models/decoder_*.pth")
    model_files = sorted(list(set(model_files)))  # Remove duplicates

    if model_files:
        col1, col2 = st.columns([2, 1])

        with col1:
            selected_model = st.selectbox(
                "Select trained decoder model:",
                ["None"] + [os.path.basename(f) for f in model_files],
                index=0,
            )

        with col2:
            load_button = st.button(
                "🔄 Load Model & Visualize", type="primary"
            )

        if selected_model != "None" and load_button:
            model_path = os.path.join("workspace/models", selected_model)
            with st.spinner("Loading model..."):
                model = load_trained_model(model_path)

                if model is not None:
                    st.success(
                        f"✅ Model loaded successfully: {selected_model}"
                    )

                    # Display reconstructions
                    st.header("🖼️ Image Reconstructions")

                    # Load the original training data that was used for this model
                    # Extract dataset info from model filename
                    model_name = os.path.basename(selected_model)
                    if "synthdata_dataset-" in model_name:
                        # Extract the dataset name from the model filename
                        dataset_name = model_name.replace(
                            "decoder_", ""
                        ).replace(".pth", "")
                        dataset_file = f"workspace/datasets/synthetic/{dataset_name}.npz"

                        if os.path.exists(dataset_file):
                            # Load the original training data
                            training_data = np.load(dataset_file)
                            training_images = training_data["images"]
                            training_firing_rates = training_data["responses"]

                            st.info(
                                f"Using original training data: {dataset_name}"
                            )

                            # Get some test images for visualization
                            if training_images.ndim == 4:
                                test_images = (
                                    training_images[:10, 0]
                                    if training_images.shape[1] == 1
                                    else training_images[:10, 0]
                                )
                            else:
                                test_images = training_images[:10]

                            # Generate reconstructions using the original training data
                            with torch.no_grad():
                                test_firing_rates = torch.tensor(
                                    training_firing_rates[:10],
                                    dtype=torch.float32,
                                ).to(device)
                                test_reconstructions = (
                                    model(test_firing_rates).cpu().numpy()
                                )

                                # Ensure proper shape - remove extra dimensions if present
                                if test_reconstructions.ndim == 4:
                                    test_reconstructions = (
                                        test_reconstructions.squeeze(1)
                                    )  # Remove channel dimension if present
                                elif test_reconstructions.ndim == 2:
                                    # If it's 2D, reshape to (N, H, W)
                                    test_reconstructions = (
                                        test_reconstructions.reshape(
                                            len(test_reconstructions),
                                            test_images.shape[1],
                                            test_images.shape[2],
                                        )
                                    )

                                # Resize reconstructions to match original image dimensions if needed
                                if (
                                    test_reconstructions.shape[1:]
                                    != test_images.shape[1:]
                                ):
                                    resized_reconstructions = []
                                    for recon in test_reconstructions:
                                        resized = resize(
                                            recon,
                                            test_images.shape[1:],
                                            preserve_range=True,
                                        )
                                        resized_reconstructions.append(resized)
                                    test_reconstructions = np.array(
                                        resized_reconstructions
                                    )
                        else:
                            st.error(
                                f"Original training data not found: {dataset_file}"
                            )
                            st.stop()
                    else:
                        st.error(
                            "Could not determine original training data from model filename"
                        )
                        st.stop()

                    fig = plot_reconstructions(
                        test_images, test_reconstructions
                    )
                    st.pyplot(fig)

                    # Calculate reconstruction quality metrics
                    st.subheader("📊 Reconstruction Quality")
                    col1, col2, col3 = st.columns(3)

                    # MSE between original and reconstructed
                    mse = np.mean((test_images - test_reconstructions) ** 2)
                    with col1:
                        st.metric("MSE", f"{mse:.4f}")

                    # Calculate PSNR and correlation per image
                    n_images = test_images.shape[0]
                    psnr_values = []
                    correlation_values = []

                    for i in range(n_images):
                        pred_img = test_reconstructions[i]
                        target_img = test_images[i]

                        # Calculate MSE for this image
                        img_mse = np.mean((target_img - pred_img) ** 2)

                        # Calculate PSNR for this image
                        max_val = np.max(target_img)
                        if img_mse > 0:
                            psnr = 20 * np.log10(max_val / np.sqrt(img_mse))
                        else:
                            psnr = float("inf")  # Perfect reconstruction
                        psnr_values.append(psnr)

                        # Calculate correlation for this image
                        correlation = np.corrcoef(
                            target_img.flatten(), pred_img.flatten()
                        )[0, 1]
                        correlation_values.append(correlation)

                    # Average the metrics across all images
                    mean_psnr = np.mean(psnr_values)
                    mean_correlation = np.mean(correlation_values)

                    with col2:
                        st.metric("PSNR (dB)", f"{mean_psnr:.2f}")

                    with col3:
                        st.metric("Correlation", f"{mean_correlation:.3f}")
                else:
                    st.error("❌ Failed to load model")
    else:
        st.info("No trained decoder models found in workspace/models/ directory")


if __name__ == "__main__":
    main()
