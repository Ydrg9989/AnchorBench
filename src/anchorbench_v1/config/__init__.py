"""Configuration loader for AnchorBench v1."""

from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG_DIR = Path(__file__).parent


def load_models_config() -> Dict[str, Any]:
    """Load the models.yaml configuration file."""
    with open(_CONFIG_DIR / "models.yaml") as f:
        return yaml.safe_load(f)


def get_role_config(role: str) -> Dict[str, Any]:
    """Return the model configuration for a given role name."""
    cfg = load_models_config()
    roles = cfg.get("roles", {})
    if role not in roles:
        raise KeyError(f"Unknown role {role!r}. Available: {list(roles)}")
    return roles[role]
