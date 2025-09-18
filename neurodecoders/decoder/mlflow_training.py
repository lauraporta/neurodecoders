#!/usr/bin/env python3
"""
Generic MLflow training script for neural decoders.
"""

import os
from typing import Any, Dict

import mlflow

from neurodecoders.data.loading import (
    normalize_images_and_rates,
)
from neurodecoders.decoder.training import train_decoder
from neurodecoders.mlflow_utils.argument_parsers import (
    create_decoder_parser,
    parse_decoder_args,
)
from neurodecoders.mlflow_utils.utils import (
    log_dataset_input_and_params,
    log_model_artifacts,
    log_training_config,
    log_training_metrics,
    setup_mlflow_experiment,
)
from neurodecoders.paths import get_path


def main(config: Dict[str, Any]):
    print("=== Neural Decoder Training (MLflow) ===")
    setup_mlflow_experiment(
        config["mlflow_experiment_name"], config.get("tracking_uri")
    )

    with mlflow.start_run(
        run_name=config["mlflow_run_name"], log_system_metrics=True
    ):
        log_training_config(config)

        # Load synthetic data using the same method as encoder
        from neurodecoders.data.loading import load_synthetic_split_data

        # Create config for synthetic data loading
        synthetic_config = {
            "dataset_type": config["dataset_type"],
            "sta_type": config["sta_type"],
            "n_neurons": config["n_neurons"],
            "n_images": config["n_images"],
        }

        images, firing, labels, metadata = load_synthetic_split_data(
            synthetic_config
        )
        images, firing, H, W = normalize_images_and_rates(images, firing)

        # Create data module for dataset logging
        from neurodecoders.data import NeuralDataModule

        data_module_for_logging = NeuralDataModule(
            images=images,
            firing_rates=firing,
            labels=labels,
            batch_size=config["batch_size"],
            dataset_metadata=metadata,
            use_memory_mapping=False,
            chunk_size=100,
            prefetch_factor=2,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )

        # Log dataset metadata
        log_dataset_input_and_params(data_module_for_logging)

        trainer, model, data_module = train_decoder(
            images=images,
            firing_rates=firing,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            optimizer=config["optimizer"],
            loss_fn=config["loss_function"],
            model_type=config["model_type"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            enable_mixed_precision=config["enable_mixed_precision"],
            enable_early_stopping=config["enable_early_stopping"],
            early_stopping_patience=config["early_stopping_patience"],
            enable_checkpointing=True,
            mlflow_experiment_name=config["mlflow_experiment_name"],
            mlflow_run_name=None,
        )

        # Evaluate on test split to get final test loss
        test_result = trainer.test(
            model, datamodule=data_module, verbose=False
        )
        final_test_loss = None
        if test_result and isinstance(test_result, list) and test_result[0]:
            final_test_loss = test_result[0].get("test_loss")
            if final_test_loss is not None:
                log_training_metrics({"test_loss": float(final_test_loss)})

        # Save final model weights path as artifact and metadata
        model_dir = get_path("workspace/models/decoders")
        os.makedirs(model_dir, exist_ok=True)
        dataset_name = (
            f"{config['dataset_type']}"
            f"_{config['sta_type'].replace(',', '_')}"
            f"_{config['n_neurons']}n_{config['n_images']}i"
        )
        model_path = f"{model_dir}/decoder_{dataset_name}.pth"
        import torch

        torch.save(model.state_dict(), model_path)

        # Log final losses as params/metrics
        final_train = (
            float(model.train_losses[-1]) if model.train_losses else None
        )
        final_val = float(model.val_losses[-1]) if model.val_losses else None
        if final_train is not None:
            log_training_metrics(
                {"train_loss": final_train}, step=config["epochs"]
            )
        if final_val is not None:
            log_training_metrics(
                {"val_loss": final_val}, step=config["epochs"]
            )
        if final_test_loss is not None:
            log_training_metrics(
                {"test_loss": float(final_test_loss)}, step=config["epochs"]
            )

        log_model_artifacts(
            model=model,
            model_name="decoder_model",
            model_type="decoder",
            dataset_info={
                "dataset_type": config["dataset_type"],
                "sta_type": config["sta_type"],
                "n_images": len(images),
                "n_neurons": firing.shape[1],
                "image_shape": images.shape[1:],
            },
            training_info={
                "epochs": config["epochs"],
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "train_loss": final_train,
                "val_loss": final_val,
                "test_loss": float(final_test_loss)
                if final_test_loss is not None
                else None,
            },
        )

    print("Training complete. View runs with: mlflow ui")


if __name__ == "__main__":
    parser = create_decoder_parser()
    args = parser.parse_args()
    config = parse_decoder_args(args)
    main(config)
