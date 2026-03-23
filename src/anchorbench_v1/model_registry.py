"""Verify that configured model IDs are live on OpenRouter.

Usage:
    PYTHONPATH=src python -m anchorbench_v1.model_registry --check
"""

from __future__ import annotations

import argparse
import sys

from .config import load_models_config


def check_models(live_ids: set[str] | None = None) -> list[str]:
    """Return list of error messages (empty = all OK)."""
    cfg = load_models_config()
    roles = cfg.get("roles", {})

    if live_ids is None:
        from .openrouter_client import OpenRouterClient
        client = OpenRouterClient()
        live_ids = set(client.list_models())

    errors = []
    for role, rcfg in roles.items():
        mid = rcfg["model_id"]
        if mid not in live_ids:
            errors.append(f"[{role}] model {mid!r} not found on OpenRouter")
        else:
            print(f"  OK  {role:20s} → {mid}")
    return errors


def main() -> None:
    """CLI entry point for model registry validation."""
    parser = argparse.ArgumentParser(description="AnchorBench model registry")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Query OpenRouter /models and verify configured IDs.",
    )
    args = parser.parse_args()

    if not args.check:
        parser.print_help()
        return

    print("Checking model IDs against OpenRouter...")
    errors = check_models()
    if errors:
        print(f"\n{'='*50}")
        for e in errors:
            print(f"  ERROR: {e}")
        print(f"\nFix model IDs in config/models.yaml")
        sys.exit(1)
    else:
        print("\nAll model IDs verified.")


if __name__ == "__main__":
    main()
