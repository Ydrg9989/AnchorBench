"""The CLI commands exit non-zero when the work they launched failed.

`anchorbench eval` and `experiment` run the runner in a subprocess, and
`tables` runs each generator in one; a lost return code means a crashed
run, an OOM or a missing dataset leaves a partial results tree behind a
green exit status. hydra.main returns None whatever the task function
returns, so the Hydra apps signal failure with SystemExit instead.
"""

from __future__ import annotations

import sys

import pytest

from anchorbench.cli import eval as eval_cli
from anchorbench.cli import experiment as experiment_cli
from anchorbench.cli import tables as tables_cli


class _Completed:
    def __init__(self, returncode: int):
        self.returncode = returncode


def _fake_run(rc_for_cmd):
    """A subprocess.run stand-in; rc_for_cmd maps the command tokens to a code."""
    def run(cmd, **_kwargs):
        return _Completed(rc_for_cmd(cmd))
    return run


def _argv(monkeypatch, tmp_path, *overrides: str) -> None:
    monkeypatch.setattr(sys, "argv", [
        "anchorbench", *overrides,
        f"out_dir={tmp_path}", f"+api_out_dir={tmp_path}",
        f"hydra.run.dir={tmp_path}/hydra",
    ])


def test_eval_exits_with_the_runner_code(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_cli.subprocess, "run", _fake_run(lambda cmd: 3))
    _argv(monkeypatch, tmp_path, "model=qwen_7b", "data=external")
    with pytest.raises(SystemExit) as exc:
        eval_cli.main()
    assert exc.value.code == 3


def test_eval_returns_zero_when_the_runner_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_cli.subprocess, "run", _fake_run(lambda cmd: 0))
    _argv(monkeypatch, tmp_path, "model=qwen_7b", "data=external")
    assert eval_cli.main() == 0


def test_experiment_exits_nonzero_when_one_cell_fails(monkeypatch, tmp_path):
    seen: list[list[str]] = []

    def rc(cmd):
        seen.append(cmd)
        return 1 if "anchorbench.runners.rag" in cmd else 0

    monkeypatch.setattr(experiment_cli.subprocess, "run", _fake_run(rc))
    _argv(monkeypatch, tmp_path, "+models=[qwen_7b]", "+suites=[external,rag]")
    with pytest.raises(SystemExit) as exc:
        experiment_cli.main()
    assert exc.value.code == 1
    assert len(seen) == 2, "the failing cell does not stop the plan"


def test_experiment_returns_zero_when_every_cell_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment_cli.subprocess, "run", _fake_run(lambda cmd: 0))
    _argv(monkeypatch, tmp_path, "+models=[qwen_7b]", "+suites=[external]")
    assert experiment_cli.main() == 0


def test_experiment_refuses_an_empty_plan(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(experiment_cli.subprocess, "run", _fake_run(lambda cmd: ran.append(cmd) or 0))
    _argv(monkeypatch, tmp_path)          # no recipe, no models, no suites
    with pytest.raises(SystemExit) as exc:
        experiment_cli.main()
    assert "no cells" in str(exc.value)
    assert ran == []


def test_tables_propagates_an_extension_analysis_failure(monkeypatch):
    def rc(cmd):
        return 5 if "anchorbench.analysis.gold_shift" in cmd else 0

    monkeypatch.setattr(tables_cli.subprocess, "run", _fake_run(rc))
    monkeypatch.setattr(sys, "argv", ["anchorbench tables"])
    assert tables_cli.main() != 0


def test_tables_returns_zero_when_every_generator_succeeds(monkeypatch):
    monkeypatch.setattr(tables_cli.subprocess, "run", _fake_run(lambda cmd: 0))
    monkeypatch.setattr(sys, "argv", ["anchorbench tables"])
    assert tables_cli.main() == 0
