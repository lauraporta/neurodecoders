import datetime
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "encoder"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "synthetic"))

# Import MEI functionality
from mei import (
    MEIOptimizer,
    generate_sta_patterns,
    load_encoder_model,
    load_synthetic_data,
    plot_mei_optimization,
    save_mei_results,
)

# Configure Streamlit page
st.set_page_config(
    page_title="MEI (Maximal Exciting Image) Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)


def load_available_models():
    """Load available trained encoder models"""
    try:
        # Find all encoder model files
        files = glob.glob("data/encoder_model_*.pth")
        if not files:
            st.error(
                "No encoder model files found in data/ directory. Please train an encoder first."
            )
            return {}

        # Create a mapping of display names to file paths
        file_options = {}
        for file_path in files:
            # Extract meaningful info from filename for display
            filename = os.path.basename(file_path)
            # Remove the encoder_model_ prefix and .pth suffix
            display_name = filename.replace("encoder_model_", "").replace(
                ".pth", ""
            )
            # Replace underscores with spaces for better readability
            display_name = display_name.replace("_", " ")
            file_options[display_name] = file_path

        # Sort by creation time (newest first)
        sorted_files = sorted(
            file_options.items(),
            key=lambda x: os.path.getctime(x[1]),
            reverse=True,
        )

        return dict(sorted_files)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}


def infer_dataset_from_model(model_path):
    """Automatically infer the appropriate dataset path from the encoder model path"""
    try:
        # Extract dataset info from model filename
        model_filename = os.path.basename(model_path)

        if "synthdata_dataset-" in model_filename:
            # Parse the dataset identifier from the model filename
            # Example: encoder_synthdata_dataset-gaussian_noise_n_neurons_100_epochs_50.pth
            # Extract: gaussian_noise_n_neurons_100_epochs_50
            dataset_identifier = model_filename.split("synthdata_dataset-")[
                1
            ].split(".pth")[0]

            # Construct the expected dataset filename
            dataset_filename = f"synthdata_dataset-{dataset_identifier}.npz"

            # Look for the dataset in the data directory
            data_dir = "data"
            dataset_path = os.path.join(data_dir, dataset_filename)

            if os.path.exists(dataset_path):
                return dataset_path
            else:
                # Try alternative locations
                alt_paths = [
                    os.path.join("neurodecoders", "data", dataset_filename),
                    os.path.join("..", "data", dataset_filename),
                    dataset_filename,  # Try current directory
                ]

                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        return alt_path

                # If not found, return None
                return None
        else:
            # For models without clear dataset identifier, try to find any dataset
            data_dir = "data"
            if os.path.exists(data_dir):
                dataset_files = [
                    f
                    for f in os.listdir(data_dir)
                    if f.endswith(".npz") and "synthdata" in f
                ]
                if dataset_files:
                    # Use the first available dataset
                    return os.path.join(data_dir, dataset_files[0])

            return None

    except Exception as e:
        print(f"Error inferring dataset from model: {e}")
        return None


def get_dataset_info(dataset_path):
    """Get information about a synthetic dataset"""
    try:
        # Get file info
        file_size = os.path.getsize(dataset_path) / (1024 * 1024)  # MB
        mod_time = datetime.datetime.fromtimestamp(
            os.path.getmtime(dataset_path)
        )

        # Extract info from filename
        filename = os.path.basename(dataset_path)
        if "synthdata_dataset-" in filename:
            # Parse dataset info from filename
            dataset_info = filename.split("synthdata_dataset-")[1].split(
                ".npz"
            )[0]
            dataset_info = dataset_info.replace("_", " ")
        else:
            dataset_info = "Unknown dataset"

        return {
            "file_size_mb": file_size,
            "modified": mod_time,
            "dataset_info": dataset_info,
            "filename": filename,
        }
    except Exception:
        return {
            "file_size_mb": 0,
            "modified": datetime.datetime.now(),
            "dataset_info": "Error loading dataset",
            "filename": os.path.basename(dataset_path),
        }


def get_model_info(model_path):
    """Get information about a trained encoder model"""
    try:
        # Load the model to get basic info
        device = torch.device("cpu")
        encoder_model = load_encoder_model(model_path, device)

        # Get number of output neurons
        n_neurons = None
        for layer in reversed(encoder_model.fc):
            if isinstance(layer, nn.Linear):
                n_neurons = layer.out_features
                break

        # Get file info
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        mod_time = datetime.datetime.fromtimestamp(
            os.path.getmtime(model_path)
        )

        # Extract info from filename
        filename = os.path.basename(model_path)
        if "synthdata_dataset-" in filename:
            # Parse dataset info from filename
            dataset_info = filename.split("synthdata_dataset-")[1].split(
                "_n_neurons"
            )[0]
            dataset_info = dataset_info.replace("_", " ")
        else:
            dataset_info = "Unknown dataset"

        return {
            "n_neurons": n_neurons,
            "file_size_mb": file_size,
            "modified": mod_time,
            "dataset_info": dataset_info,
            "filename": filename,
        }
    except Exception:
        return {
            "n_neurons": "Unknown",
            "file_size_mb": 0,
            "modified": datetime.datetime.now(),
            "dataset_info": "Error loading model",
            "filename": os.path.basename(model_path),
        }


def main():
    st.title("🎯 MEI (Maximal Exciting Image) Dashboard")
    st.info(f"Device in use: {device.type.upper()}")

    # Sidebar controls
    st.sidebar.header("Model Selection")

    # Load available models
    available_models = load_available_models()
    if not available_models:
        st.stop()

    # Model selection with dropdown
    selected_model_display = st.sidebar.selectbox(
        "Select Encoder Model:",
        options=list(available_models.keys()),
        index=0,
        help="Choose a trained encoder model to use for MEI optimization",
    )

    selected_model_path = available_models[selected_model_display]

    # Show model info
    st.sidebar.info(f"**Selected Model:** {selected_model_display}")

    # Display model information
    model_info = get_model_info(selected_model_path)
    with st.sidebar.expander("Model Details", expanded=False):
        st.write(f"**Filename:** {model_info['filename']}")
        st.write(f"**Dataset:** {model_info['dataset_info']}")
        st.write(f"**Neurons:** {model_info['n_neurons']}")
        st.write(f"**File Size:** {model_info['file_size_mb']:.1f} MB")
        st.write(
            f"**Modified:** {model_info['modified'].strftime('%Y-%m-%d %H:%M')}"
        )

    # Automatically infer the appropriate dataset from the model
    selected_dataset_path = infer_dataset_from_model(selected_model_path)

    if selected_dataset_path is None:
        st.error(
            "Could not automatically find the appropriate dataset for the selected model. Please ensure the dataset file exists."
        )
        st.stop()

    # Show dataset info
    dataset_info = get_dataset_info(selected_dataset_path)
    st.sidebar.info(
        f"**Auto-selected Dataset:** {dataset_info['dataset_info']}"
    )

    # Display dataset information
    with st.sidebar.expander("Dataset Details", expanded=False):
        st.write(f"**Filename:** {dataset_info['filename']}")
        st.write(f"**Dataset:** {dataset_info['dataset_info']}")
        st.write(f"**File Size:** {dataset_info['file_size_mb']:.1f} MB")
        st.write(
            f"**Modified:** {dataset_info['modified'].strftime('%Y-%m-%d %H:%M')}"
        )

    # MEI Optimization Parameters
    st.sidebar.header("MEI Optimization Parameters")

    # Target neuron selection
    # First, try to load the model to get number of neurons
    try:
        encoder_model = load_encoder_model(selected_model_path, device)
        # Find the last nn.Linear layer in the fc sequence
        n_neurons = None
        for layer in reversed(encoder_model.fc):
            if isinstance(layer, nn.Linear):
                n_neurons = layer.out_features
                break
        if n_neurons is None:
            raise ValueError(
                "Could not determine number of output neurons from model structure."
            )
        st.sidebar.info(f"Model has {n_neurons} output neurons")

        target_neuron = st.sidebar.slider(
            "Target Neuron Index",
            min_value=0,
            max_value=n_neurons - 1,
            value=0,
            help="Select which neuron to optimize for",
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # Optimization parameters
    learning_rate_exp = st.sidebar.slider(
        "Learning Rate (10^)",
        min_value=-4,
        max_value=-1,
        value=-2,
        step=1,
        help="Learning rate in exponential notation (e.g., -2 means 10^-2 = 0.01)",
    )
    learning_rate = 10**learning_rate_exp

    num_iterations = st.sidebar.slider(
        "Number of Iterations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Number of optimization iterations",
    )

    regularization_weight_exp = st.sidebar.slider(
        "Regularization Weight (10^)",
        min_value=-4,
        max_value=-2,
        value=-3,
        step=1,
        help="L2 regularization weight in exponential notation",
    )
    regularization_weight = 10**regularization_weight_exp

    # Learning rate scheduler parameters
    st.sidebar.header("Learning Rate Scheduler")

    patience = st.sidebar.slider(
        "Patience",
        min_value=50,
        max_value=200,
        value=100,
        step=10,
        help="Number of iterations to wait before reducing LR on plateau",
    )

    factor = st.sidebar.slider(
        "Reduction Factor",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Factor by which to reduce learning rate (0.5 = halve the LR)",
    )

    min_lr_exp = st.sidebar.slider(
        "Minimum LR (10^)",
        min_value=-6,
        max_value=-4,
        value=-6,
        step=1,
        help="Minimum learning rate threshold",
    )
    min_lr = 10**min_lr_exp

    noise_std = st.sidebar.slider(
        "Initial Noise Standard Deviation",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Standard deviation of initial random noise",
    )

    # Run MEI Optimization
    if st.sidebar.button("🚀 Run MEI Optimization"):
        with st.spinner("Running MEI optimization..."):
            try:
                # Load the encoder model
                encoder_model = load_encoder_model(selected_model_path, device)

                # Load synthetic data to get STA patterns
                images, firing_rates, sta_info = load_synthetic_data(
                    selected_dataset_path
                )

                # Generate STA patterns for comparison and simulation
                if sta_info:
                    sta_patterns = generate_sta_patterns(sta_info, n_neurons)
                    sta_pattern = sta_patterns[target_neuron]

                    # Generate RF coordinates for the neurons (same as in synthetic simulation)
                    rf_size = sta_patterns.shape[1]  # Get patch size from STA
                    rf_coords = np.random.randint(
                        0, 224 - rf_size, size=(n_neurons, 2)
                    )
                else:
                    # Create dummy data if info not available
                    sta_patterns = np.random.randn(n_neurons, 11, 11) * 0.1
                    sta_patterns = np.tanh(sta_patterns)
                    sta_pattern = sta_patterns[target_neuron]
                    rf_coords = np.random.randint(
                        0, 224 - 11, size=(n_neurons, 2)
                    )

                # Create MEI optimizer with STA data
                mei_optimizer = MEIOptimizer(
                    device,
                    encoder_model,
                    target_neuron,
                    sta_patterns=sta_patterns,
                    rf_coords=rf_coords,
                )

                # Get STA patch size for initialization
                sta_patch_size = sta_patterns.shape[
                    1
                ]  # Should be 11x11 or similar

                # Initialize random patch (same size as STA)
                initial_patch = mei_optimizer.initialize_random_image(
                    sta_size=sta_patch_size, noise_std=noise_std
                )

                # Create placeholders for real-time updates
                st.subheader("🔄 Real-time MEI Optimization")
                image_placeholder = st.empty()
                progress_placeholder = st.empty()
                loss_plot_placeholder = st.empty()

                # Create progress bar
                progress_bar = st.progress(0)

                # Create metrics columns once and store references
                col1, col2, col3, col4, col5 = st.columns(5)
                loss_metric = col1.empty()
                pred_fr_metric = col2.empty()
                exp_fr_metric = col3.empty()
                lr_metric = col4.empty()
                progress_metric = col5.empty()

                # Initialize loss history for plotting
                loss_history_realtime = []

                # Callback function for real-time updates
                def update_visualization(
                    iteration,
                    current_patch,
                    loss,
                    predicted_fr,
                    expected_fr,
                    current_lr,
                ):
                    # Update progress bar
                    progress = (iteration + 1) / num_iterations
                    progress_bar.progress(progress)

                    # Update progress text
                    progress_placeholder.text(
                        f"Step {iteration + 1}/{num_iterations} - Loss: {loss:.4f} - Predicted FR: {predicted_fr:.2f} Hz - Expected FR: {expected_fr:.2f} Hz - LR: {current_lr:.2e}"
                    )

                    # Update patch - clear previous and show new
                    image_placeholder.empty()
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    im = ax.imshow(
                        current_patch.squeeze().cpu().numpy(),
                        cmap="gray",
                        vmin=-1,
                        vmax=1,
                    )
                    ax.set_title(f"MEI Patch Evolution - Step {iteration + 1}")
                    ax.axis("off")
                    plt.colorbar(im, ax=ax)
                    image_placeholder.pyplot(fig)
                    plt.close(fig)

                    # Update metrics with latest values only
                    loss_metric.metric("Current Loss", f"{loss:.4f}")
                    pred_fr_metric.metric(
                        "Predicted FR", f"{predicted_fr:.2f} Hz"
                    )
                    exp_fr_metric.metric(
                        "Expected FR", f"{expected_fr:.2f} Hz"
                    )
                    lr_metric.metric("Learning Rate", f"{current_lr:.2e}")
                    progress_metric.metric(
                        "Progress", f"{progress * 100:.1f}%"
                    )

                    # Update loss plot
                    loss_history_realtime.append(loss)
                    loss_plot_placeholder.empty()
                    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(10, 4))
                    ax_loss.plot(
                        loss_history_realtime, linewidth=2, color="blue"
                    )
                    ax_loss.set_title("Loss During Optimization")
                    ax_loss.set_xlabel("Iteration")
                    ax_loss.set_ylabel("Loss")
                    ax_loss.grid(True, alpha=0.3)
                    ax_loss.set_yscale(
                        "log"
                    )  # Use log scale for better visualization
                    loss_plot_placeholder.pyplot(fig_loss)
                    plt.close(fig_loss)

                # Run optimization with real-time updates
                (
                    optimized_patch,
                    loss_history,
                    firing_rate_history,
                    expected_firing_rate_history,
                    lr_history,
                ) = mei_optimizer.optimize_image(
                    initial_image=initial_patch,
                    learning_rate=learning_rate,
                    num_iterations=num_iterations,
                    regularization_weight=regularization_weight,
                    update_interval=50,  # Update every 50 iterations
                    callback=update_visualization,
                    patience=patience,
                    factor=factor,
                    min_lr=min_lr,
                )

                # Clear only the image placeholder and progress elements
                image_placeholder.empty()
                progress_placeholder.empty()
                progress_bar.empty()
                loss_plot_placeholder.empty()

                # Clear metrics
                loss_metric.empty()
                pred_fr_metric.empty()
                exp_fr_metric.empty()
                lr_metric.empty()
                progress_metric.empty()

                # Compare MEI with STA
                comparison_metrics = mei_optimizer.compare_with_sta(
                    optimized_patch, sta_pattern, target_neuron
                )

                # Save results
                image_path, history_path = save_mei_results(
                    optimized_patch,
                    loss_history,
                    firing_rate_history,
                    comparison_metrics,
                    selected_model_path,
                    target_neuron,
                )

                # Create and display results
                st.subheader("🎯 MEI Optimization Results")

                # Display comparison plots
                fig_results = plot_mei_optimization(
                    optimized_patch,
                    sta_pattern,
                    loss_history,
                    firing_rate_history,
                    comparison_metrics,
                    target_neuron,
                )
                st.pyplot(fig_results)

                # Display optimization progress
                st.subheader("📈 Optimization Progress")

                col1, col2 = st.columns(2)
                with col1:
                    fig_loss = plt.figure(figsize=(8, 6))
                    plt.plot(loss_history)
                    plt.title("Loss During Optimization")
                    plt.xlabel("Iteration")
                    plt.ylabel("Loss")
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig_loss)

                with col2:
                    fig_rate = plt.figure(figsize=(8, 6))
                    plt.plot(
                        firing_rate_history, label="Predicted FR", linewidth=2
                    )
                    plt.plot(
                        expected_firing_rate_history,
                        label="Expected FR",
                        linewidth=2,
                        linestyle="--",
                    )
                    plt.title("Firing Rate Evolution")
                    plt.xlabel("Iteration")
                    plt.ylabel("Firing Rate (Hz)")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig_rate)

                # Learning rate evolution
                st.subheader("📉 Learning Rate Evolution")
                fig_lr = plt.figure(figsize=(10, 6))
                plt.semilogy(lr_history, linewidth=2, color="red")
                plt.title("Learning Rate Schedule")
                plt.xlabel("Iteration")
                plt.ylabel("Learning Rate")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig_lr)

                # Display final metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric(
                        "Correlation",
                        f"{comparison_metrics['correlation']:.3f}",
                    )
                with col2:
                    st.metric(
                        "Cosine Similarity",
                        f"{comparison_metrics['cosine_similarity']:.3f}",
                    )
                with col3:
                    st.metric("SSIM", f"{comparison_metrics['ssim']:.3f}")
                with col4:
                    st.metric(
                        "Final Predicted FR",
                        f"{firing_rate_history[-1]:.2f} Hz",
                    )
                with col5:
                    st.metric(
                        "Final Expected FR",
                        f"{expected_firing_rate_history[-1]:.2f} Hz",
                    )

                st.success(
                    f"✅ MEI optimization completed! Results saved to {image_path}"
                )

            except Exception as e:
                st.error(f"Error during MEI optimization: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
