"""
DEPRECATED: This file maintains backwards compatibility.
For new code, use the modular structure:
- neurodecoders.encoder.models for model architectures
- neurodecoders.encoder.training for training pipelines
- neurodecoders.encoder.utils for utilities
"""

import argparse
from pathlib import Path

# Import from new modular structure for backwards compatibility
from .models import SimpleEncoder
from .training import EncoderLightningModule, train_encoder
from .utils import (
    NeuralDataModule,
    NeuralDataset,
    load_latest_data,
    plot_training_results,
    preprocess_data,
    save_model_with_metadata,
    save_predictions,
    visualize_data,
)

# Re-export commonly used items for backwards compatibility
__all__ = [
    "SimpleEncoder",
    "EncoderLightningModule",
    "NeuralDataset",
    "NeuralDataModule",
    "load_latest_data",
    "preprocess_data",
    "visualize_data",
    "train_model_lightning",  # Legacy function
    "plot_training_results",
    "save_predictions",
]


# Legacy wrapper function for backwards compatibility
def train_model_lightning(
    images,
    firing_rates,
    train_split=0.7,
    val_split=0.15,
    batch_size=32,
    learning_rate=1e-3,
    epochs=30,
    enable_progress_bar=True,
    log_every_n_steps=50,
    callbacks=None,
):
    """
    DEPRECATED: Legacy wrapper for backwards compatibility.
    Use train_simple_encoder() or train_resnet_encoder() from training module.
    """
    print("WARNING: train_model_lightning is deprecated. Use new functions.")

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

    # Train using new training pipeline
    trainer, lightning_model, data_module = train_encoder(
        model=model,
        data_module=data_module,
        learning_rate=learning_rate,
        epochs=epochs,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        logger_name="legacy_encoder",
    )

    # Return in old format for compatibility
    return trainer, lightning_model, data_module


def main(dataset_to_load, epochs=30, learning_rate=1e-3):
    """Main function to run encoder training - DEPRECATED: Use new modules"""
    print("=== Neural Encoder Training with PyTorch Lightning ===")
    print(
        "WARNING: This function is deprecated. Consider using the new modules."
    )

    # Import from new modules
    from .models import SimpleEncoder
    from .training import train_encoder
    from .utils import (
        NeuralDataModule,
        load_latest_data,
        plot_training_results,
        preprocess_data,
        save_predictions,
        visualize_data,
    )

    # Load data
    print("Loading data...")
    images, firing_rates, data_file = load_latest_data(dataset_to_load)

    # Preprocess data
    images, firing_rates = preprocess_data(images, firing_rates)

    # Visualize data
    visualize_data(firing_rates)

    # Create data module
    data_module = NeuralDataModule(
        images=images,
        firing_rates=firing_rates,
        train_split=0.7,
        val_split=0.15,
        batch_size=32,
    )

    # Create model
    model = SimpleEncoder(out_neurons=firing_rates.shape[1])

    # Train with new training pipeline
    trainer, lightning_model, data_module = train_encoder(
        model=model,
        data_module=data_module,
        learning_rate=learning_rate,
        epochs=epochs,
        enable_progress_bar=True,
        logger_name="simple_encoder",
    )

    # Plot training results
    plot_training_results(
        lightning_model.train_losses, lightning_model.val_losses
    )

    # Save predictions
    save_predictions(
        lightning_model.model,
        images,
        firing_rates,
        data_file,
        "workspace/predictions/encoder",
        dataset_to_load,
    )

    # Save model with metadata
    save_model_with_metadata(
        lightning_model.model,
        data_file,
        model_type="simple_encoder",
        additional_info={"epochs": epochs, "lr": learning_rate},
    )

    print("=== Lightning Training Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural encoder model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="workspace/datasets/synthetic/synthdata_dataset-cifar10_sta-perlin_noise_patterns,11,11_n_neurons-1000_n_images-1000_datetime-20250703_162151.npz",
        help="Path to the dataset file (.npz format)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for training (default: 1e-3)",
    )

    args = parser.parse_args()
    dataset_to_load = Path(args.dataset)

    # Check if dataset file exists
    if not dataset_to_load.exists():
        print(f"Error: Dataset file '{dataset_to_load}' not found!")
        print("Available datasets in workspace/datasets/synthetic/ directory:")
        data_dir = Path("workspace/datasets/synthetic")
        if data_dir.exists():
            for file in data_dir.glob("*.npz"):
                print(f"  {file}")
        exit(1)

    print(f"Using dataset: {dataset_to_load}")
    main(dataset_to_load, epochs=args.epochs, learning_rate=args.learning_rate)
