import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import datetime
import os
from image_datasets import ImageDataset
from sta import STA
from simulate_response import SimulateResponse
from create_simulated_neural_responses import plot_sta_and_spikes, plot_response_heatmaps_all, plot_response_histograms, plot_neural_correlations, save_output

# Configure Streamlit page to use wide mode
st.set_page_config(layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    st.title("Neural Response Visualization")
    st.info(f"Device in use: {device.type.upper()}")
    
    # Sidebar controls
    st.sidebar.header("Parameters")
    
    # Dataset selection
    dataset_type = st.sidebar.selectbox(
        "Dataset",
        ["mnist", "cifar10"],
        index=0,
        help="Choose the dataset to use for generating neural responses"
    )
    
    # STA type selection with patch size
    sta_base_type = st.sidebar.selectbox(
        "STA Type",
        [
            "perlin_noise_patterns",
            "binary_patterns",
            "periodic_patterns",
            "from_model"
        ],
        index=0,
        help="Choose the type of Spatiotemporal Average (STA) to use"
    )
    
    # Add model selection and layer selection for model-based STAs
    if sta_base_type == "from_model":
        model_type = st.sidebar.selectbox(
            "Model",
            ["Alexnet", "ResNet", "VGG"],
            index=0,
            help="Choose the pretrained model to use"
        )
        
        # Define meaningful layers for each model
        model_layers = {
            "Alexnet": {
                "layers": [0, 3, 6, 8, 10],  # Conv layers in AlexNet
                "names": ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"]
            },
            "ResNet": {
                "layers": [0, 3, 7, 10, 13, 16],  # First conv layer of each block in ResNet18
                "names": ["Conv1", "Block1", "Block2", "Block3", "Block4", "Block5"]
            },
            "VGG": {
                "layers": [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28],  # Conv layers in VGG16
                "names": ["Conv1_1", "Conv1_2", "Conv2_1", "Conv2_2", "Conv3_1", "Conv3_2", 
                         "Conv3_3", "Conv4_1", "Conv4_2", "Conv4_3", "Conv5_1", "Conv5_2", "Conv5_3"]
            }
        }
        
        # Create layer selection with meaningful names
        layer_idx = st.sidebar.selectbox(
            "Layer",
            range(len(model_layers[model_type]["layers"])),
            format_func=lambda x: model_layers[model_type]["names"][x],
            help=f"Choose which layer to use from {model_type}"
        )
        
        # Get the actual layer index
        layer = model_layers[model_type]["layers"][layer_idx]
        sta_type = f"from_model:{model_type},{layer}"
    # Add patch size input for pattern-based STAs
    elif "patterns" in sta_base_type:
        patch_size = st.sidebar.number_input(
            "Patch Size",
            min_value=3,
            max_value=63,
            value=11,
            step=2,
            help="Size of the STA patch (must be odd number)"
        )
        # Ensure patch size is odd
        if patch_size % 2 == 0:
            patch_size += 1
            st.sidebar.info(f"Patch size adjusted to {patch_size} (must be odd)")
        sta_type = f"{sta_base_type},{patch_size},{patch_size}"
    else:
        sta_type = sta_base_type
    
    # Visualization parameters
    st.sidebar.header("Visualization Parameters")
    
    # Number of neurons and images to generate
    n_neurons = st.sidebar.slider(
        "Total Number of Neurons",
        min_value=5,
        max_value=50,
        value=30,
        help="Total number of neurons to generate"
    )
    
    n_images = st.sidebar.slider(
        "Total Number of Images",
        min_value=5,
        max_value=50,
        value=30,
        help="Total number of images to generate"
    )
    
    # Number of neurons and images to visualize
    n_plot_neurons = st.sidebar.slider(
        "Number of Neurons to Visualize",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of top neurons to show in the visualization"
    )
    
    n_plot_images = st.sidebar.slider(
        "Number of Images to Visualize",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of images to show in each panel"
    )
    
    # Add Generate New Data button above large dataset section
    if st.sidebar.button("Generate New Data"):
        with st.spinner("Generating neural responses..."):
            # Load data and model
            images = ImageDataset().get_data(dataset_type, n_images)
            stas = STA().get_simulated_sta(sta_type)
            
            # Generate responses
            simulator = SimulateResponse(device, images, stas, n_neurons)
            firing_rates, dot_products, adaptation_states = simulator.simulate_neural_responses()
            
            # Save the data
            os.makedirs("output", exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_output(images, firing_rates, simulator.selected_stas, simulator.rf_coords, adaptation_states,
                        dataset_type, sta_type, n_neurons, n_images)
                       
            
            # Create and display the plots
            fig1, fig2 = plot_sta_and_spikes(
                images=images,
                responses=firing_rates,
                dot_products=dot_products,
                adaptation_states=adaptation_states,
                stas=simulator.selected_stas,
                coords=simulator.rf_coords,
                n_plot_images=n_plot_images,
                n_top_neurons=n_plot_neurons
            )
            st.pyplot(fig1)
            st.pyplot(fig2)

            # Create and display the all-neuron/image heatmap
            fig4 = plot_response_heatmaps_all(firing_rates, dot_products, adaptation_states)
            st.pyplot(fig4)

            # Create and display the histogram plot
            fig5 = plot_response_histograms(firing_rates, dot_products, adaptation_states)
            st.pyplot(fig5)

            # Create and display the neural correlation plot
            fig6 = plot_neural_correlations(firing_rates)
            st.pyplot(fig6)

            st.success("Data generated and visualized successfully!")
    
    # Large dataset generation parameters
    st.sidebar.header("Large Dataset Generation")
    
    large_n_images = st.sidebar.number_input(
        "Large Dataset - Number of Images",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Number of images for large dataset generation"
    )
    
    large_n_neurons = st.sidebar.number_input(
        "Large Dataset - Number of Neurons", 
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Number of neurons for large dataset generation"
    )
    
    # Add Generate Large Dataset button below the large dataset section
    if st.sidebar.button("Generate Large Dataset", type="primary"):
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Loading data and model
            status_text.text("Step 1/4: Loading data and model...")
            progress_bar.progress(25)
            
            images = ImageDataset().get_data(dataset_type, large_n_images)
            stas = STA().get_simulated_sta(sta_type)
            
            # Step 2: Generating responses
            status_text.text("Step 2/4: Generating neural responses...")
            progress_bar.progress(50)
            
            simulator = SimulateResponse(device, images, stas, large_n_neurons)
            firing_rates, dot_products, adaptation_states = simulator.simulate_neural_responses()
            
            # Step 3: Creating visualizations
            status_text.text("Step 3/4: Creating visualizations...")
            progress_bar.progress(75)
            
            # Create output directory
            os.makedirs("output", exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save the dataset
            save_output(images, firing_rates, simulator.selected_stas, simulator.rf_coords, adaptation_states,
                        dataset_type, sta_type, large_n_neurons, large_n_images)
            
            # Create and save visualizations
            fig1 = plot_response_heatmaps_all(firing_rates, dot_products, adaptation_states)
            fig1.savefig(f"output/heatmaps_all_{timestamp}.png", dpi=300, bbox_inches='tight')
            
            fig2 = plot_response_histograms(firing_rates, dot_products, adaptation_states)
            fig2.savefig(f"output/response_histograms_{timestamp}.png", dpi=300, bbox_inches='tight')
            
            fig3 = plot_neural_correlations(firing_rates)
            fig3.savefig(f"output/neural_correlations_{timestamp}.png", dpi=300, bbox_inches='tight')
            
            # Step 4: Complete
            status_text.text("Step 4/4: Complete!")
            progress_bar.progress(100)
            
            # Display results
            st.success(f"Large dataset generated successfully!")
            st.info(f"""
            **Generated Files:**
            - Dataset: `data/synthdata_dataset-{dataset_type}_sta-{sta_type}_n_neurons-{large_n_neurons}_n_images-{large_n_images}_datetime-{timestamp}.npz`
            - Heatmaps: `output/heatmaps_all_{timestamp}.png`
            - Histograms: `output/response_histograms_{timestamp}.png`
            - Correlations: `output/neural_correlations_{timestamp}.png`
            
            **Dataset Statistics:**
            - Images: {large_n_images}
            - Neurons: {large_n_neurons}
            - Total responses: {large_n_images * large_n_neurons:,}
            - Mean firing rate: {firing_rates.mean():.2f} Hz
            - Max firing rate: {firing_rates.max():.2f} Hz
            """)
            
            # Show sample visualizations
            st.subheader("Sample Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(f"output/heatmaps_all_{timestamp}.png", caption="Response Heatmaps")
            
            with col2:
                st.image(f"output/response_histograms_{timestamp}.png", caption="Response Distributions")
            
            st.image(f"output/neural_correlations_{timestamp}.png", caption="Neural Correlations")
            
        except Exception as e:
            st.error(f"Error generating dataset: {str(e)}")
            st.exception(e)
    
    # Add some information about the visualization
    st.markdown("""
    ### About the Visualization
    
    This app visualizes simulated neural responses to images. The visualization shows:
    
    1. **Left Panel**: First set of images with their corresponding neural responses
    2. **Right Panel**: Additional set of images with their neural responses
    3. **Bottom**: Spatiotemporal Averages (STAs) for the top neurons
    
    Each neuron's response is shown as both firing rate (dots) and dot product (x markers).
    The colored rectangles on the images show the receptive fields of the top neurons.
    
    ### Available Options
    
    - **Datasets**: MNIST (handwritten digits) and CIFAR10 (color images)
    - **STA Types**:
        - Perlin Noise Patterns: Natural-looking noise patterns
        - Binary Patterns: Black and white patterns
        - Periodic Patterns: Sine wave-based patterns
        - Model-based STAs:
            - AlexNet: 5 convolutional layers
            - ResNet18: 6 convolutional blocks
            - VGG16: 13 convolutional layers
    
    For pattern-based STAs, you can adjust the patch size (must be an odd number).
    For model-based STAs, you can choose which layer's features to use as STAs.
    
    ### Large Dataset Generation
    
    Use the "Generate Large Dataset" button to create datasets with many neurons and images for training neural decoders.
    The generated .npz files contain:
    - `images`: Input images (N, 1, H, W)
    - `responses`: Neural firing rates (N, C)
    - `stas`: Spatiotemporal averages (C, H, W)
    - `rf_coords`: Receptive field coordinates (C, 2)
    - `adaptation_states`: Adaptation states (N, C)
    
    These files can be loaded directly into your training scripts.
    """)

if __name__ == "__main__":
    main()
