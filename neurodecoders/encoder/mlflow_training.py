#!/usr/bin/env python3
"""
Generic MLflow training script for neural encoders.

This script provides a flexible interface for training different encoder models
with various configurations, datasets, and hyperparameters while tracking
experiments with MLflow.
"""

import os
import sys
from typing import Any, Dict

# Add the encoder directory to the path
sys.path.append(os.path.dirname(__file__))

import torch

from neurodecoders.data import NeuralDataModule
from neurodecoders.data.loading import (
    load_synthetic_split_data,
)
from neurodecoders.encoder.models import (
    ResNetConvOnly,
    ResNetConv_2layerHead,
    ResNetEncoder,
    ResNetFromScratch,
    Simple3LayerEncoder,
    SimpleEncoder,
    SimpleEncoderWithSkipConnection,
)
from neurodecoders.encoder.training import train_encoder
from neurodecoders.mlflow_utils.argument_parsers import (
    create_encoder_parser,
    parse_encoder_args,
)


# get_model remains encoder-specific
def get_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Create model based on configuration.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        model: PyTorch model
    """
    model_type = config["model_type"]
    out_neurons = config.get("out_neurons")

    if out_neurons is None:
        raise ValueError(
            "out_neurons must be specified or inferred from dataset before "
            "calling get_model"
        )

    if model_type == "simple":
        return SimpleEncoder(out_neurons=out_neurons)
    elif model_type == "simple3layer":
        return Simple3LayerEncoder(out_neurons=out_neurons)
    elif model_type == "skip":
        return SimpleEncoderWithSkipConnection(out_neurons=out_neurons)
    elif model_type == "resnet":
        freeze_backbone = config["freeze_backbone"]
        return ResNetEncoder(
            out_neurons=out_neurons,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "resnet_scratch":
        freeze_backbone = config["freeze_backbone"]
        return ResNetFromScratch(
            out_neurons=out_neurons,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "resnet_conv_only":
        freeze_backbone = config["freeze_backbone"]
        return ResNetConvOnly(
            out_neurons=out_neurons,
            freeze_backbone=freeze_backbone,
        )
    elif model_type == "resnet_conv_2layer":
        freeze_backbone = config["freeze_backbone"]
        return ResNetConv_2layerHead(
            out_neurons=out_neurons,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_with_config(config: Dict[str, Any]):
    """
    Train encoder with given configuration.

    Args:
        config: Configuration dictionary

    Returns:
        trainer, model, data_module: Training results
    """
    print("=== ENCODER TRAINING WITH CONFIG ===")
    print(f"Model type: {config['model_type']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Loss function: {config['loss_function']}")
    print(f"Scheduler: {config['scheduler']}")
    print(f"Dataset: {config['dataset_type']}")
    print(f"STA type: {config['sta_type']}")
    print(f"Neurons: {config['n_neurons']}")
    print(f"Images: {config['n_images']}")
    print(f"Cross-validation folds: {config['cv_folds']}")
    print(f"Mixed precision: {config['enable_mixed_precision']}")
    print(f"Early stopping: {config['enable_early_stopping']}")
    print(f"Checkpointing: {config['enable_checkpointing']}")

    # Load data
    images, firing_rates, labels, metadata = load_synthetic_split_data(config)

    # Infer out_neurons from dataset if not specified
    if config.get("out_neurons") is None:
        config["out_neurons"] = firing_rates.shape[1]
        print(f"Inferred output neurons from dataset: {config['out_neurons']}")
    else:
        print(f"Output neurons: {config['out_neurons']}")

    # Create data module
    data_module = NeuralDataModule(
        images=images,
        firing_rates=firing_rates,
        labels=labels,
        batch_size=config["batch_size"],
        dataset_metadata=metadata,
        use_memory_mapping=config["use_memory_mapping"],
        chunk_size=config["chunk_size"],
        prefetch_factor=config["prefetch_factor"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    # Create optimizer and scheduler configurations
    optimizer_config = {
        "type": config["optimizer"],
    }

    scheduler_config = {
        "type": config["scheduler"],
        "step_size": config["scheduler_step_size"],
        "gamma": config["scheduler_gamma"],
    }

    # Create model
    model = get_model(config)

    # Train the model using the main training function
    fold_results = train_encoder(
        model=model,
        data_module=data_module,
        model_name=f"{config['model_type']}_encoder",
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        optimizer_config=optimizer_config,
        loss_fn=config["loss_function"],
        scheduler_config=scheduler_config,
        enable_mlflow=config["enable_mlflow"],
        mlflow_experiment_name=config["mlflow_experiment_name"],
        mlflow_run_name=config["mlflow_run_name"],
        n_folds=config["cv_folds"],
        enable_mixed_precision=config["enable_mixed_precision"],
        enable_early_stopping=config["enable_early_stopping"],
        early_stopping_patience=config["early_stopping_patience"],
        enable_checkpointing=config["enable_checkpointing"],
    )

    # Return the first fold result for backward compatibility
    if fold_results:
        return fold_results[0]
    else:
        return None, None, data_module


def main():
    """Main function for command-line training."""
    parser = create_encoder_parser()
    args = parser.parse_args()

    # Use shared encoder args parser to build config
    config = parse_encoder_args(args)

    trainer, model, _ = train_with_config(config)

    print("\nTraining completed!")
    if model.train_losses:
        print(f"Final train loss: {model.train_losses[-1]:.4f}")
    if model.val_losses:
        print(f"Final validation loss: {model.val_losses[-1]:.4f}")

    print("\nTo view MLflow experiments, run:")
    print("mlflow ui")
    print("Then open http://localhost:5000 in your browser.")


if __name__ == "__main__":
    main()
