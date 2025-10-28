"""
Configuration management for neurodecoders.

This module provides utilities for loading and accessing configuration
from the config.yaml file and environment variables.
"""

import os
from pathlib import Path
from typing import Optional

import yaml

try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, will use environment variables only
    pass


def get_config_path() -> Path:
    """Get the path to the config.yaml file."""
    # Try to find config.yaml in the project root
    current_dir = Path(__file__).parent.parent
    config_path = current_dir / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "Please ensure config.yaml exists in the project root."
        )
    
    return config_path


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = get_config_path()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_mlflow_tracking_uri() -> Optional[str]:
    """
    Get the MLflow tracking URI from environment variables or config.
    
    Priority:
    1. MLFLOW_TRACKING_URI environment variable (if set explicitly)
    2. Construct from POSTGRES_* environment variables (from .env)
    3. Fall back to config.yaml
    4. Fall back to file system
    
    Returns:
        str: MLflow tracking URI, or None if not configured
    """
    # Check for explicit MLFLOW_TRACKING_URI
    env_uri = os.getenv('MLFLOW_TRACKING_URI')
    if env_uri:
        return env_uri
    
    # Try to construct from POSTGRES_* environment variables
    pg_user = os.getenv('POSTGRES_USER')
    pg_password = os.getenv('POSTGRES_PASSWORD')
    pg_host = os.getenv('POSTGRES_HOST')
    pg_port = os.getenv('POSTGRES_PORT')
    pg_db = os.getenv('POSTGRES_DB')
    
    if pg_user and pg_password and pg_db and pg_host and pg_port:
        return f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"



def get_base_path() -> str:
    """Get the base path from config."""
    config = load_config()
    return config.get('base_path', '')
