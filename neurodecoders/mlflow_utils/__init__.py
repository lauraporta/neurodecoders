"""
Shared MLflow utilities for neurodecoders.

This package provides common MLflow functionality for both encoder and decoder
training, including experiment tracking, argument parsing, and utilities.
"""

from .experiment_tracker import MLflowExperimentTracker
from .argument_parsers import (
    create_encoder_parser,
    create_decoder_parser,
    create_common_parser,
    parse_encoder_args,
    parse_decoder_args,
)
from .utils import (
    setup_mlflow_experiment,
    log_training_config,
    log_model_artifacts,
    log_training_metrics,
    log_validation_metrics,
    log_test_metrics,
    log_encoder_verification_metrics,
    log_cross_validation_metrics,
    get_experiment_comparison,
)

__all__ = [
    "MLflowExperimentTracker",
    "create_encoder_parser",
    "create_decoder_parser",
    "create_common_parser",
    "parse_encoder_args",
    "parse_decoder_args",
    "setup_mlflow_experiment",
    "log_training_config",
    "log_model_artifacts",
    "log_training_metrics",
    "log_validation_metrics",
    "log_test_metrics",
    "log_encoder_verification_metrics",
    "log_cross_validation_metrics",
    "get_experiment_comparison",
]
