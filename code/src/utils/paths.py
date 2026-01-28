"""Path management utilities."""
import os
from pathlib import Path


def get_project_root():
    """Get the project root directory (code/)."""
    return Path(__file__).parent.parent.parent


def resolve_path(path_str, base_dir=None):
    """
    Resolve a path string relative to base_dir or project root.
    
    Args:
        path_str: Path string (can be relative or absolute)
        base_dir: Base directory for relative paths (default: project root)
        
    Returns:
        Absolute Path object
    """
    if base_dir is None:
        base_dir = get_project_root()
    
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_config_path(config_type, name):
    """
    Get path to a configuration file.
    
    Args:
        config_type: 'models' or 'languages'
        name: Config name (with or without .yaml extension)
        
    Returns:
        Path to config file
    """
    root = get_project_root()
    if not name.endswith('.yaml'):
        name = f"{name}.yaml"
    return root / 'configs' / config_type / name
