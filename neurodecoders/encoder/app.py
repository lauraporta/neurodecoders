import datetime
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from pytorch_lightning.callbacks import Callback

# Add the encoder directory to the path so we can import from it
sys.path.append(os.path.dirname(__file__))

# Import the refactored modules
from neurodecoders.encoder.models import SimpleEncoder
from neurodecoders.encoder.training import train_encoder
from neurodecoders.encoder.utils import (
    NeuralDataModule,
    load_latest_data,
    plot_firing_rate_distribution,
    plot_predictions_vs_actual,
    preprocess_data,
    save_predictions,
)

# Configure Streamlit page
st.set_page_config(
    page_title="Neural Encoder Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create workspace directories if they don't exist
os.makedirs("workspace/datasets/synthetic", exist_ok=True)
os.makedirs("workspace/models", exist_ok=True)
os.makedirs("workspace/predictions", exist_ok=True)


class StreamlitProgressCallback(Callback):
    """Custom Lightning callback to update Streamlit progress during
    training"""

    def __init__(
        self, progress_callback=None, metrics_callback=None, total_epochs=30
    ):
        super().__init__()
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback
        self.total_epochs = total_epochs
        self.best_val_loss = float("inf")

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch"""
        current_epoch = trainer.current_epoch + 1
        progress = current_epoch / self.total_epochs

        # Get current losses with fallback to stored losses in module
        train_loss = trainer.callback_metrics.get("train_loss_epoch", 0)
        val_loss = trainer.callback_metrics.get("val_loss", 0)

        # Fallback to module's stored losses if callback metrics are empty
        if (
            (train_loss == 0 or val_loss == 0)
            and hasattr(pl_module, "train_losses")
            and hasattr(pl_module, "val_losses")
        ):
            if pl_module.train_losses:
                train_loss = pl_module.train_losses[-1]
            if pl_module.val_losses:
                val_loss = pl_module.val_losses[-1]

        # Convert tensors to floats
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.item()
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()

        # Update best validation loss
        if val_loss > 0 and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        # Update Streamlit progress
        if self.progress_callback:
            status = f"Epoch {current_epoch}/{self.total_epochs}"
            if train_loss > 0:
                status += f" - Train Loss: {train_loss:.4f}"
            if val_loss > 0:
                status += f", Val Loss: {val_loss:.4f}"
            self.progress_callback(progress, status)

        # Update metrics
        if self.metrics_callback:
            self.metrics_callback(train_loss, val_loss, self.best_val_loss)


def load_data():
    """Load neural data file selected by user from dropdown"""
    try:
        # Find all synthdata files
        files = glob.glob(
            "workspace/datasets/synthetic/synthdata_dataset-*.npz"
        )
        if not files:
            st.error(
                "No neural data files found in workspace/datasets/synthetic/ "
                "directory. Please generate data first using the synthetic "
                "dashboard."
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

        # Use the utility function to load data
        images, firing_rates, _ = load_latest_data(selected_file)

        return images, firing_rates, selected_file
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None


# Visualization functions now imported from utils.py


def train_encoder_lightning(
    images,
    firing_rates,
    train_split=0.7,
    val_split=0.15,
    batch_size=32,
    learning_rate=1e-3,
    epochs=30,
    progress_callback=None,
    metrics_callback=None,
):
    """Train the encoder model using PyTorch Lightning with real-time
    updates"""

    # Preprocess data using utility function
    images, firing_rates = preprocess_data(images, firing_rates)

    # Create data module
    data_module = NeuralDataModule(
        images=images,
        firing_rates=firing_rates,
        train_split=train_split,
        val_split=val_split,
        batch_size=batch_size,
    )

    # Create model
    model = SimpleEncoder(out_neurons=firing_rates.shape[1])

    # Create custom callback for Streamlit updates
    streamlit_callback = StreamlitProgressCallback(
        progress_callback=progress_callback,
        metrics_callback=metrics_callback,
        total_epochs=epochs,
    )

    # Train using the new training pipeline
    trainer, lightning_model, data_module = train_encoder(
        model=model,
        data_module=data_module,
        learning_rate=learning_rate,
        epochs=epochs,
        callbacks=[streamlit_callback],
        enable_progress_bar=False,  # Disable Lightning's progress bar since
        # we have Streamlit
        logger_name="streamlit_encoder",
    )

    # Update progress to 100% when training is complete
    if progress_callback:
        progress_callback(1.0, "Training completed!")

    # Get test predictions
    lightning_model.eval()
    test_predictions = []
    test_actuals = []

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            x, y = batch
            pred = lightning_model(x)
            test_predictions.append(pred.cpu().numpy())
            test_actuals.append(y.cpu().numpy())

    test_predictions = np.concatenate(test_predictions, axis=0)
    test_actuals = np.concatenate(test_actuals, axis=0)

    # Calculate test loss
    test_loss = np.mean((test_predictions - test_actuals) ** 2)

    return {
        "model": lightning_model,
        "train_losses": lightning_model.train_losses,
        "val_losses": lightning_model.val_losses,
        "test_loss": test_loss,
        "predictions": test_predictions,
        "actuals": test_actuals,
        "epochs_trained": len(lightning_model.train_losses),
    }


def main():
    st.title("🧠 Neural Encoder Dashboard")
    st.markdown(
        "Train a neural encoder to map images to firing rates using "
        "PyTorch Lightning"
    )
    st.info(f"Device in use: {device.type.upper()}")

    # Sidebar controls
    st.sidebar.header("📊 Data & Model Settings")

    # Load data
    images, firing_rates, data_file = load_data()

    if images is None:
        st.stop()

    # Display data info
    st.sidebar.info(f"""
    **Data Info:**
    - Images: {images.shape}
    - Neurons: {firing_rates.shape[1]}
    - Mean firing rate: {firing_rates.mean():.2f} Hz
    - Max firing rate: {firing_rates.max():.2f} Hz
    """)

    # Training parameters
    st.sidebar.header("⚙️ Training Parameters")

    # Data split
    train_split = st.sidebar.slider("Training Split", 0.5, 0.9, 0.7, 0.05)
    val_split = st.sidebar.slider("Validation Split", 0.05, 0.3, 0.15, 0.05)
    test_split = 1.0 - train_split - val_split

    if test_split < 0:
        st.sidebar.error("Invalid split: train + val > 1.0")
        st.stop()

    st.sidebar.info(f"Test split: {test_split:.2f}")

    # Model parameters
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    learning_rate = st.sidebar.selectbox(
        "Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2
    )
    epochs = st.sidebar.slider("Max Epochs", 10, 100, 30, 5)

    # Main content
    st.header("📈 Data Visualization")

    # Show firing rate distribution
    if st.button("Show Firing Rate Distribution"):
        fig = plot_firing_rate_distribution(firing_rates)
        st.pyplot(fig)

    st.header("🚀 Training")

    # Add helpful information
    st.info("""
    **Training Information:**
    - Training will show real-time progress updates
    - Loss curves will update after each epoch
    - Progress bar shows overall training completion
    - You can see live metrics during training
    """)

    if st.button("Start Training", type="primary"):
        # Training progress
        st.subheader("Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create placeholders for metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        # Create placeholders for real-time metrics
        train_loss_placeholder = metric_col1.empty()
        val_loss_placeholder = metric_col2.empty()
        best_val_placeholder = metric_col3.empty()

        # Create placeholder for real-time loss plot
        st.subheader("📈 Live Training Curves")
        loss_plot_placeholder = st.empty()

        # Lists to store losses for real-time plotting
        live_train_losses = []
        live_val_losses = []

        def update_progress(progress, status):
            progress_bar.progress(progress)
            status_text.text(status)

        def update_metrics(train_loss, val_loss, best_val_loss):
            train_loss_placeholder.metric("Train Loss", f"{train_loss:.4f}")
            val_loss_placeholder.metric("Val Loss", f"{val_loss:.4f}")
            best_val_placeholder.metric(
                "Best Val Loss", f"{best_val_loss:.4f}"
            )

            # Update real-time loss plot
            if train_loss > 0:  # Only plot when we have valid losses
                live_train_losses.append(train_loss)
                live_val_losses.append(val_loss if val_loss > 0 else 0)

                # Create real-time plot (limit to reasonable update frequency)
                if len(live_train_losses) % 1 == 0:  # Update every epoch
                    try:
                        plt.ioff()  # Turn off interactive mode
                        fig, ax = plt.subplots(figsize=(10, 6))
                        epochs_so_far = list(
                            range(1, len(live_train_losses) + 1)
                        )
                        ax.plot(
                            epochs_so_far,
                            live_train_losses,
                            label="Train Loss",
                            linewidth=2,
                            color="blue",
                        )
                        if len(live_val_losses) > 0 and any(
                            v > 0 for v in live_val_losses
                        ):
                            val_to_plot = [
                                v if v > 0 else None for v in live_val_losses
                            ]
                            ax.plot(
                                epochs_so_far,
                                val_to_plot,
                                label="Validation Loss",
                                linewidth=2,
                                color="orange",
                            )
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("MSE Loss")
                        ax.set_title("Training Progress (Live Update)")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()

                        # Update the plot
                        loss_plot_placeholder.pyplot(fig)
                        plt.close(fig)  # Close to prevent memory leaks
                        plt.ion()  # Turn interactive mode back on
                    except Exception:
                        # If plotting fails, continue without breaking training
                        pass

        try:
            # Initialize progress
            update_progress(0.0, "Initializing training...")

            # Train the model with Lightning
            with st.spinner("Setting up model and data..."):
                results = train_encoder_lightning(
                    images=images,
                    firing_rates=firing_rates,
                    train_split=train_split,
                    val_split=val_split,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    progress_callback=update_progress,
                    metrics_callback=update_metrics,
                )

            progress_bar.progress(1.0)
            status_text.text("Training completed!")

            # Display results
            st.subheader("📊 Final Evaluation")
            st.success(f"Test Loss: {results['test_loss']:.4f}")

            # Show final training summary
            if results["train_losses"] and results["val_losses"]:
                final_train_loss = results["train_losses"][-1]
                final_val_loss = results["val_losses"][-1]
                min_val_loss = (
                    min(results["val_losses"])
                    if results["val_losses"]
                    else float("inf")
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("Final Train Loss", f"{final_train_loss:.4f}")
                col2.metric("Final Val Loss", f"{final_val_loss:.4f}")
                col3.metric("Best Val Loss", f"{min_val_loss:.4f}")

            # Plot predictions vs actual
            st.subheader("🎯 Predictions vs Actual")
            fig = plot_predictions_vs_actual(
                results["predictions"], results["actuals"]
            )
            st.pyplot(fig)

            # Save training info to data folder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_info = {
                "test_loss": results["test_loss"],
                "train_losses": results["train_losses"],
                "val_losses": results["val_losses"],
                "epochs_trained": results["epochs_trained"],
                "data_file": data_file,
                "timestamp": timestamp,
            }

            training_info_path = (
                f"workspace/models/encoder_training_info_{timestamp}.npy"
            )
            np.save(training_info_path, model_info)
            st.info(f"Training info saved to: {training_info_path}")

            # Save model
            model_path = f"workspace/models/encoder_model_\
                {os.path.splitext(os.path.basename(data_file))[0]}.pth"
            torch.save(results["model"].state_dict(), model_path)
            st.info(f"Model saved to: {model_path}")

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.exception(e)

    # Model loading and inference section
    st.header("🔍 Model Inference")

    # Find all possible encoder model files
    model_files = []
    model_files.extend(glob.glob("workspace/models/best_encoder_model.pth"))
    model_files.extend(glob.glob("workspace/models/encoder_model_*.pth"))
    model_files.extend(
        glob.glob("workspace/models/lightning_encoder_model_*.pth")
    )

    if model_files:
        st.success(f"Found {len(model_files)} trained model(s)!")

        # Create a mapping of display names to file paths
        model_options = {}
        for file_path in model_files:
            # Extract meaningful info from filename for display
            filename = os.path.basename(file_path)
            # Remove the .pth suffix
            display_name = filename.replace(".pth", "")
            # Replace underscores with spaces for better readability
            display_name = display_name.replace("_", " ")

            # Try to extract neuron count from filename for better display
            neuron_match = re.search(r"n_neurons-(\d+)", filename)
            if neuron_match:
                neuron_count = neuron_match.group(1)
                display_name = f"{display_name} ({neuron_count} neurons)"

            model_options[display_name] = file_path

        # Sort by creation time (newest first) for the dropdown
        sorted_models = sorted(
            model_options.items(),
            key=lambda x: os.path.getctime(x[1]),
            reverse=True,
        )

        # Create dropdown
        selected_model_name = st.selectbox(
            "Select Model:",
            options=[name for name, _ in sorted_models],
            index=0,  # Default to newest model
            help="Choose a trained encoder model to load",
        )

        # Get the selected model path
        selected_model_path = model_options[selected_model_name]

        if st.button("Load Model and Run Inference"):
            try:
                # Load state dict first to determine the model architecture
                state_dict = torch.load(selected_model_path)

                # Handle state dicts that have "model." prefix (from Lightning
                # modules)
                if any(key.startswith("model.") for key in state_dict.keys()):
                    # Strip the "model." prefix from all keys
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith("model."):
                            new_key = key[6:]  # Remove "model." prefix
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    state_dict = new_state_dict

                # Determine the number of output neurons from the saved model
                # Look for the final layer weights (fc.6.weight)
                if "fc.6.weight" in state_dict:
                    out_neurons = state_dict["fc.6.weight"].shape[0]
                elif "model.fc.6.weight" in torch.load(selected_model_path):
                    # If we still have the original state dict with model.
                    # prefix
                    out_neurons = torch.load(selected_model_path)[
                        "model.fc.6.weight"
                    ].shape[0]
                else:
                    # Fallback to current dataset size
                    out_neurons = firing_rates.shape[1]

                st.info(f"Loading model with {out_neurons} output neurons")

                # Create model with the correct number of output neurons
                model = SimpleEncoder(out_neurons=out_neurons).to(device)
                model.load_state_dict(state_dict)
                model.eval()

                # Check if we have data loaded for inference
                if images is None or firing_rates is None:
                    st.warning(
                        "⚠️ No dataset loaded. Please load a dataset first to "
                        "run inference."
                    )
                    st.stop()

                # Run inference on a few samples
                with torch.no_grad():
                    # Handle image dimensions properly
                    if images.ndim == 4:
                        # Images are already [N, C, H, W], just take first 5
                        sample_images = torch.tensor(
                            images[:5], dtype=torch.float32
                        ).to(device)
                    else:
                        # Images are [N, H, W], add channel dimension
                        sample_images = torch.tensor(
                            images[:5, None, :, :], dtype=torch.float32
                        ).to(device)

                    predictions = model(sample_images).cpu().numpy()
                    actuals = firing_rates[:5]

                # Save predictions for all data
                dataset_path = Path(data_file)
                predictions_file = save_predictions(
                    model,
                    images,
                    firing_rates,
                    data_file,
                    dataset_to_load=dataset_path,
                )
                st.success(f"Predictions saved to: {predictions_file}")

                # Display results
                st.subheader("Sample Predictions")
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))

                # Determine how many neurons to plot (minimum of model output
                # and dataset)
                n_neurons_to_plot = min(predictions.shape[1], actuals.shape[1])

                for i in range(5):
                    # Show image - handle different dimensions
                    if images.ndim == 4:
                        img_display = images[i, 0]  # Take first channel if 4D
                    else:
                        img_display = images[i]

                    axes[0, i].imshow(img_display, cmap="gray")
                    axes[0, i].set_title(f"Sample {i + 1}")
                    axes[0, i].axis("off")

                    # Show predictions vs actual (only for neurons that exist
                    # in both)
                    if n_neurons_to_plot > 0:
                        axes[1, i].scatter(
                            actuals[i, :n_neurons_to_plot],
                            predictions[i, :n_neurons_to_plot],
                            alpha=0.6,
                        )
                        max_val = max(
                            actuals[i, :n_neurons_to_plot].max(),
                            predictions[i, :n_neurons_to_plot].max(),
                        )
                        axes[1, i].plot([0, max_val], [0, max_val], "r--")
                        axes[1, i].set_xlabel("Actual")
                        axes[1, i].set_ylabel("Predicted")
                        axes[1, i].set_title(f"Neurons 1-{n_neurons_to_plot}")
                    else:
                        axes[1, i].text(
                            0.5,
                            0.5,
                            "No compatible neurons",
                            ha="center",
                            va="center",
                            transform=axes[1, i].transAxes,
                        )
                        axes[1, i].set_title("No Data")

                    axes[1, i].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Inference failed: {str(e)}")
                st.exception(e)
    else:
        st.info(
            "No trained model found. Train a model first to enable inference."
        )


if __name__ == "__main__":
    main()
