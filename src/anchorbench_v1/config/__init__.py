"""Configuration loader for AnchorBench v1."""

from pathlib import Path
import yaml

_CONFIG_DIR = Path(__file__).parent


def load_models_config() -> dict:
    with open(_CONFIG_DIR / "models.yaml") as f:
        return yaml.safe_load(f)


def get_role_config(role: str) -> dict:
    cfg = load_models_config()
    roles = cfg.get("roles", {})
    if role not in roles:
        raise KeyError(f"Unknown role {role!r}. Available: {list(roles)}")
    return roles[role]
