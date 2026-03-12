"""
Shared argument parsers for neurodecoders.

This module provides common argument parsing functionality for both
encoder and decoder training, maintaining backward compatibility
with existing command line interfaces.
"""

import argparse
from typing import Any, Dict


def create_common_parser() -> argparse.ArgumentParser:
    """
    Create a parser with common arguments shared between encoder and decoder.

    Returns:
        ArgumentParser with common arguments
    """
    parser = argparse.ArgumentParser(
        description="Neurodecoders training with MLflow tracking"
    )

    # MLflow configuration
    parser.add_argument(
        "--experiment-name",
        default="neurodecoders",
        help="MLflow experiment name",
    )
    parser.add_argument("--run-name", help="MLflow run name")
    parser.add_argument("--tracking-uri", help="MLflow tracking server URI")

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd", "adamw"],
        default="adam",
        help="Optimizer type",
    )

    return parser


def create_encoder_parser() -> argparse.ArgumentParser:
    """
    Create a parser with encoder-specific arguments.

    Returns:
        ArgumentParser with encoder arguments
    """
    parser = create_common_parser()
    parser.description = "Train neural encoder with MLflow tracking"

    # Model configuration
    parser.add_argument(
        "--model-type",
        choices=[
            "simple",
            "simple3layer",
            "skip",
            "resnet",
            "resnet_scratch",
            "resnet_conv_only",
            "resnet_conv_2layer",
        ],
        default="simple3layer",
        help="Type of encoder model (simple3layer recommended for reconstruction)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="Freeze ResNet18 backbone (only for resnet models)",
    )
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Unfreeze ResNet backbone (overrides --freeze-backbone)",
    )

    # Loss and scheduler configuration
    parser.add_argument(
        "--loss-function",
        choices=["mse", "l1", "smooth_l1", "huber"],
        default="mse",
        help="Loss function",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "step", "cosine", "plateau"],
        default="none",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=30,
        help="Step size for step scheduler",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="Gamma for step scheduler (multiplies LR by this factor)",
    )

    # Data configuration
    parser.add_argument(
        "--dataset-type",
        default="cifar10",
        help="Dataset type (cifar10, mnist, etc.)",
    )
    parser.add_argument(
        "--sta-type",
        default="periodic_patterns,70,70",
        help="STA pattern type",
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=100,
        help="Number of neurons in synthetic data",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=None,
        help="Number of images in synthetic data (deprecated: use --n-train-images and --n-test-images)",
    )
    parser.add_argument(
        "--n-train-images",
        type=int,
        default=None,
        help="Number of training images in synthetic data",
    )
    parser.add_argument(
        "--n-test-images",
        type=int,
        default=None,
        help="Number of test images in synthetic data",
    )

    # Data loading configuration
    parser.add_argument(
        "--use-memory-mapping",
        action="store_true",
        help="Use memory mapping for large datasets",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Chunk size for data loading",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches to prefetch in background (0=disable)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading (0=main process)",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Pin memory for faster GPU transfer",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pin memory (overrides --pin-memory)",
    )

    # Cross-validation configuration
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help="Number of cross-validation folds (1=no CV)",
    )

    # Enhanced training options
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Enable mixed precision training (16-bit)",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Enable early stopping",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=100,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--checkpointing",
        action="store_true",
        default=True,
        help="Enable model checkpointing",
    )
    parser.add_argument(
        "--no-checkpointing",
        action="store_true",
        help="Disable model checkpointing",
    )

    return parser


def create_decoder_parser() -> argparse.ArgumentParser:
    """
    Create a parser with decoder-specific arguments.

    Returns:
        ArgumentParser with decoder arguments
    """
    parser = create_common_parser()
    parser.description = "Train neural decoder with MLflow tracking"

    # Dataset configuration - match encoder arguments
    parser.add_argument(
        "--dataset-type",
        default="cifar10",
        help="Dataset type (cifar10, mnist, etc.)",
    )
    parser.add_argument(
        "--sta-type",
        default="periodic_patterns,70,70",
        help="STA pattern type",
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=100,
        help="Number of neurons in synthetic data",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=10000,
        help="Number of images in synthetic data (deprecated: use --n-train-images and --n-test-images)",
    )
    parser.add_argument(
        "--n-train-images",
        type=int,
        default=None,
        help="Number of training images in synthetic data",
    )
    parser.add_argument(
        "--n-test-images",
        type=int,
        default=None,
        help="Number of test images in synthetic data",
    )

    # Model configuration
    parser.add_argument(
        "--model-type",
        choices=["simple", "transformer", "diffusion"],
        default="simple",
        help="Type of decoder model",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Output image size (default: 32 for CIFAR-10 native resolution)",
    )
    
    # Transformer-specific arguments
    parser.add_argument(
        "--patch-size",
        type=int,
        default=4,
        help="Patch size for transformer decoder (default: 4)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        help="Embedding dimension for transformer decoder (default: 256)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads for transformer decoder (default: 8)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers (default: 6)",
    )
    parser.add_argument(
        "--mlp-ratio",
        type=float,
        default=4.0,
        help="MLP hidden dim ratio for transformer (default: 4.0)",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.1,
        help="Dropout rate for transformer (default: 0.1)",
    )
    
    # Diffusion-specific arguments
    parser.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="Base channels for diffusion U-Net (default: 64)",
    )
    parser.add_argument(
        "--channel-mults",
        type=str,
        default="1,2,4",
        help="Channel multipliers for diffusion U-Net, comma-separated (default: '1,2,4')",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps (default: 1000)",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=50,
        help="Number of inference steps for diffusion sampling (default: 50)",
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=1e-4,
        help="Starting beta for diffusion noise schedule (default: 1e-4)",
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=0.02,
        help="Ending beta for diffusion noise schedule (default: 0.02)",
    )

    # Training configuration
    parser.add_argument(
        "--loss-function",
        choices=["mse", "l1", "smooth_l1"],
        default="mse",
        help="Loss function",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "step", "cosine", "plateau"],
        default="none",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=30,
        help="Step size for step scheduler",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="Gamma for step scheduler",
    )

    # Data loading configuration
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading (0=main process)",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Pin memory for faster GPU transfer",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pin memory (overrides --pin-memory)",
    )

    # Enhanced training options
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Enable mixed precision training (16-bit)",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Enable early stopping",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=50,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--checkpointing",
        action="store_true",
        default=True,
        help="Enable model checkpointing",
    )
    parser.add_argument(
        "--no-checkpointing",
        action="store_true",
        help="Disable model checkpointing",
    )

    return parser


def parse_encoder_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Parse encoder arguments into a config dictionary.

    Args:
        args: Parsed arguments

    Returns:
        Dictionary with encoder configuration
    """
    # Handle boolean logic
    freeze_backbone = args.freeze_backbone and not args.unfreeze_backbone
    pin_memory = args.pin_memory and not args.no_pin_memory
    enable_mixed_precision = (
        args.mixed_precision and not args.no_mixed_precision
    )
    enable_early_stopping = args.early_stopping and not args.no_early_stopping
    enable_checkpointing = args.checkpointing and not args.no_checkpointing

    config = {
        "model_type": args.model_type,
        "out_neurons": None,  # Will be inferred from dataset
        "freeze_backbone": freeze_backbone,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "loss_function": args.loss_function,
        "scheduler": args.scheduler,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "dataset_type": args.dataset_type,
        "sta_type": args.sta_type,
        "n_neurons": args.n_neurons,
        "n_images": args.n_images,
        "use_memory_mapping": args.use_memory_mapping,
        "chunk_size": args.chunk_size,
        "prefetch_factor": args.prefetch_factor,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "cv_folds": args.cv_folds,
        "mlflow_experiment_name": args.experiment_name,
        "mlflow_run_name": args.run_name,
        "enable_mixed_precision": enable_mixed_precision,
        "enable_early_stopping": enable_early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "enable_checkpointing": enable_checkpointing,
        "enable_mlflow": True,
    }

    return config


def parse_decoder_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Parse decoder arguments into a config dictionary.

    Args:
        args: Parsed arguments

    Returns:
        Dictionary with decoder configuration
    """
    # Handle boolean logic
    pin_memory = args.pin_memory and not args.no_pin_memory
    enable_mixed_precision = (
        args.mixed_precision and not args.no_mixed_precision
    )
    enable_early_stopping = args.early_stopping and not args.no_early_stopping
    enable_checkpointing = args.checkpointing and not args.no_checkpointing

    # Handle train/test split defaults (match encoder behavior)
    n_train_images = args.n_train_images if args.n_train_images is not None else args.n_images
    n_test_images = args.n_test_images if args.n_test_images is not None else args.n_images
    
    # Parse channel multipliers for diffusion model
    channel_mults = tuple(int(x) for x in args.channel_mults.split(","))
    
    config = {
        "dataset_type": args.dataset_type,
        "sta_type": args.sta_type,
        "n_neurons": args.n_neurons,
        "n_images": args.n_images,
        "n_train_images": n_train_images,
        "n_test_images": n_test_images,
        "model_type": args.model_type,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "loss_function": args.loss_function,
        "scheduler": args.scheduler,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "enable_mixed_precision": enable_mixed_precision,
        "enable_early_stopping": enable_early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "enable_checkpointing": enable_checkpointing,
        "mlflow_experiment_name": args.experiment_name,
        "mlflow_run_name": args.run_name,
        "enable_mlflow": True,
        "tracking_uri": getattr(args, "tracking_uri", None),
        # Transformer-specific parameters
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "mlp_ratio": args.mlp_ratio,
        "transformer_dropout": args.transformer_dropout,
        # Diffusion-specific parameters
        "base_channels": args.base_channels,
        "channel_mults": channel_mults,
        "timesteps": args.timesteps,
        "inference_steps": args.inference_steps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
    }

    return config
