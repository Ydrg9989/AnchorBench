"""Top-level dispatcher for the ``anchorbench`` console script.

We do not use Hydra at the top level because Hydra wants to own ``sys.argv``
and that doesn't compose well with classic subcommands. Instead, we peel
off the subcommand and re-exec into the subcommand's own entrypoint
(which itself is Hydra-decorated for ``eval`` / ``experiment``).
"""

from __future__ import annotations

import sys

SUBCOMMANDS = ("eval", "experiment", "tables", "generate", "add-model", "verify")


def _print_help() -> int:
    print(
        "Usage: anchorbench <subcommand> [options]\n"
        "\n"
        "Subcommands:\n"
        "  eval         Run one (model, data) cell.\n"
        "  experiment   Run a named recipe (e.g. experiment=paper_main).\n"
        "  tables       Regenerate paper figures and LaTeX tables.\n"
        "  generate     Generate a dataset for a given suite.\n"
        "  verify       Verify every numeric paper claim against unified data.\n"
        "  add-model    Append a new model to conf/model/*.yaml.\n"
        "\n"
        "Examples:\n"
        "  anchorbench eval data=external model=qwen_7b\n"
        "  anchorbench experiment=paper_main\n"
        "  anchorbench tables --paper\n"
        "  anchorbench verify --quick\n"
    )
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        return _print_help()

    cmd = argv[0]
    sys.argv = [f"anchorbench {cmd}", *argv[1:]]

    if cmd == "eval":
        from .eval import main as run
        return run()
    if cmd == "experiment":
        from .experiment import main as run
        return run()
    if cmd == "tables":
        from .tables import main as run
        return run()
    if cmd == "generate":
        from .generate import main as run
        return run()
    if cmd == "verify":
        from anchorbench.paper import verify as v
        return v.main()
    if cmd in ("add-model", "add_model"):
        from .add_model import main as run
        return run()

    print(f"Unknown subcommand: {cmd!r}\n", file=sys.stderr)
    return _print_help() or 2


if __name__ == "__main__":
    sys.exit(main())
