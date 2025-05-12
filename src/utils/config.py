import yaml
import os

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_project_root():
    """Get the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) 