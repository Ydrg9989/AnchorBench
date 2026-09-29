#!/usr/bin/env bash
# Sourced by the scripts that call the anchorbench CLI. Uses the installed
# console script when present; otherwise runs the package straight from src/
# so a checkout without `pip install -e .` still works.
#
#   source "$(dirname "$0")/_env.sh"
#   "${ANCHORBENCH[@]}" eval data=external model=qwen_7b
#
# Also puts src/ on PYTHONPATH so `python -m anchorbench.<module>` works,
# and names the six committed dataset suites once (conf/panel.yaml lists the
# five paper suites and the icl_dist variant; the shell scripts need both).
CORE_SUITES=(external history icl icl_dist rag tool)
_ANCHORBENCH_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${_ANCHORBENCH_REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
if command -v anchorbench >/dev/null 2>&1; then
    ANCHORBENCH=(anchorbench)
else
    ANCHORBENCH=(python -m anchorbench.cli.main)
fi
