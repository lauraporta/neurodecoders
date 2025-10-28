import mlflow
from mlflow.tracking import MlflowClient
import shutil
from pathlib import Path
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables only.")

# Set up source (file system) and destination (database) clients
source_uri = "file:///ceph/margrie/laura/neurodecodersmlruns"

# Construct destination URI from environment variables
pg_user = os.getenv('POSTGRES_USER')
pg_password = os.getenv('POSTGRES_PASSWORD')
pg_host = os.getenv('POSTGRES_HOST')
pg_port = os.getenv('POSTGRES_PORT')
pg_db = os.getenv('POSTGRES_DB')
dest_uri = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"

source_client = MlflowClient(tracking_uri=source_uri)
dest_client = MlflowClient(tracking_uri=dest_uri)

# Migrate each experiment
for exp in source_client.search_experiments():
    print(f"\nMigrating experiment: {exp.name} (ID: {exp.experiment_id})")
    
    # Create experiment in destination
    try:
        dest_exp_id = dest_client.create_experiment(
            name=exp.name,
            artifact_location=exp.artifact_location,
            tags=exp.tags
        )
        print(f"  Created experiment with ID: {dest_exp_id}")
    except mlflow.exceptions.MlflowException:
        # Experiment may already exist
        dest_exp = dest_client.get_experiment_by_name(exp.name)
        dest_exp_id = dest_exp.experiment_id
        print(f"  Experiment already exists with ID: {dest_exp_id}")
    
    # Migrate runs for this experiment
    runs = source_client.search_runs(experiment_ids=[exp.experiment_id])
    print(f"  Found {len(runs)} runs to migrate")
    
    for run in runs:
        print(f"  Migrating run {run.info.run_id}...")
        run_data = source_client.get_run(run.info.run_id)
        
        # Create the run in destination
        dest_run = dest_client.create_run(
            experiment_id=dest_exp_id,
            start_time=run_data.info.start_time,
            tags=run_data.data.tags
        )
        
        # Log all parameters
        for key, value in run_data.data.params.items():
            dest_client.log_param(dest_run.info.run_id, key, value)
        
        # Log all metrics
        for key, value in run_data.data.metrics.items():
            # Get metric history for this key
            metric_history = source_client.get_metric_history(run.info.run_id, key)
            for metric in metric_history:
                dest_client.log_metric(
                    dest_run.info.run_id, 
                    key, 
                    metric.value, 
                    timestamp=metric.timestamp,
                    step=metric.step
                )
        
        # Set run status
        dest_client.set_terminated(
            dest_run.info.run_id,
            status=run_data.info.status,
            end_time=run_data.info.end_time
        )
        
        # Copy artifacts if they exist
        source_artifact_path = Path(run_data.info.artifact_uri.replace("file://", ""))
        if source_artifact_path.exists():
            dest_artifact_path = Path(dest_run.info.artifact_uri.replace("file://", ""))
            dest_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            if source_artifact_path.is_dir():
                shutil.copytree(source_artifact_path, dest_artifact_path, dirs_exist_ok=True)
        
        print(f"    ✓ Migrated run {run.info.run_id} -> {dest_run.info.run_id}")

print("\n✓ Migration complete!")