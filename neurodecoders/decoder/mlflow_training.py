#!/usr/bin/env python3
"""
Generic MLflow training script for neural decoders.
"""

import os
from typing import Any, Dict

import mlflow
import numpy as np

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
    
    # Set up MLflow with proper artifact location
    from neurodecoders.config import get_base_path
    artifact_location = f"file://{get_base_path()}/mlruns"
    
    setup_mlflow_experiment(
        config["mlflow_experiment_name"], 
        config.get("tracking_uri"),
        artifact_location=artifact_location
    )
    print(f"Using MLflow artifact location: {artifact_location}")

    with mlflow.start_run(
        run_name=config["mlflow_run_name"], log_system_metrics=True
    ):
        log_training_config(config)

        # Load existing dataset directly
        from neurodecoders.data.loading import load_npz_dataset
        
        # Use existing decoder test dataset
        dataset_path = config 
        images, firing = load_npz_dataset(dataset_path)
        images, firing, H, W = normalize_images_and_rates(images, firing)
        
        print(f"Loaded dataset: {dataset_path}")
        print(f"Images shape: {images.shape}")
        print(f"Firing rates shape: {firing.shape}")

        # Create data module for dataset logging
        from neurodecoders.data import NeuralDataModule

        data_module_for_logging = NeuralDataModule(
            images=images,
            firing_rates=firing,
            labels=None,  # No labels for decoder training
            batch_size=config["batch_size"],
            dataset_metadata={},  # Empty metadata for now
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

        # Generate and log decoder comparison plots
        try:
            from neurodecoders.decoder.generate_images import (
                create_decoder_comparison_plots,
                generate_decoder_images,
            )
            import matplotlib.pyplot as plt
            
            # Create output directory for plots
            plot_dir = get_path("workspace/plots/decoder_training")
            os.makedirs(plot_dir, exist_ok=True)
            
            # Select a few test samples for visualization
            n_vis_samples = min(5, len(images))
            vis_indices = np.random.choice(len(images), n_vis_samples, replace=False)
            vis_images = images[vis_indices]
            vis_firing = firing[vis_indices]
            
            print(f"Selected {n_vis_samples} samples for visualization")
            print(f"Original images shape: {vis_images.shape}")
            print(f"Firing rates shape: {vis_firing.shape}")
            print(f"Original images range: [{vis_images.min():.3f}, {vis_images.max():.3f}]")
            print(f"Firing rates range: [{vis_firing.min():.3f}, {vis_firing.max():.3f}]")
            
            # Generate images using the trained decoder
            device = torch.device("cuda" if torch.cuda.is_available() 
                                else "mps" if torch.backends.mps.is_available() 
                                else "cpu")
            
            # Extract the underlying PyTorch model from Lightning module
            pytorch_model = model.model
            pytorch_model.to(device)
            pytorch_model.eval()  # Ensure eval mode
            
            generated_images = generate_decoder_images(pytorch_model, vis_firing, device)
            
            print(f"Generated images shape: {generated_images.shape}")
            print(f"Generated images range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")
            
            # Create comparison plot
            comparison_path = os.path.join(plot_dir, "decoder_training_comparison.png")
            create_decoder_comparison_plots(
                original_images=list(vis_images),
                generated_images=list(generated_images),
                image_ids=vis_indices.tolist(),
                output_path=comparison_path,
            )
            
            # Log the comparison plot
            mlflow.log_artifact(comparison_path, "decoder_training_comparison.png")
            print(f"✅ Decoder training comparison plot logged: {comparison_path}")
            
            # Create and log training loss plot
            if model.train_losses and model.val_losses:
                loss_plot_path = os.path.join(plot_dir, "training_losses.png")
                plt.figure(figsize=(10, 6))
                plt.plot(model.train_losses, label='Training Loss', alpha=0.8)
                plt.plot(model.val_losses, label='Validation Loss', alpha=0.8)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Decoder Training Progress')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Log the loss plot
                mlflow.log_artifact(loss_plot_path, "training_losses.png")
                print(f"✅ Training loss plot logged: {loss_plot_path}")
            
        except Exception as e:
            print(f"⚠️  Could not generate decoder training plots: {e}")
            import traceback
            traceback.print_exc()

    print("Training complete. View runs with: mlflow ui")


if __name__ == "__main__":
    parser = create_decoder_parser()
    args = parser.parse_args()
    config = parse_decoder_args(args)
    main(config)
