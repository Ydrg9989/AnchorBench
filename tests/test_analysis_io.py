"""The shared analysis helpers reproduce the per-module variants they replaced."""

from __future__ import annotations

import csv
import json
import math

from anchorbench.analysis._io import fmt, fmt_latex, fmt_pct, write_csv, write_json


def _old_latex_signed(v, prec=2):
    """The formatter five appendix modules used to carry."""
    if v is None:
        return "---"
    s = f"{v:+.{prec}f}"
    if s.startswith("-"):
        return f"$-${s[1:]}"
    return s[1:] if s.startswith("+") else s


def test_fmt_variants():
    assert fmt(None) == "---" and fmt(float("nan")) == "---" and fmt(math.inf) == "---"
    assert fmt(0.1234) == "0.12" and fmt(0.1234, 3) == "0.123"
    assert fmt(0.1234, 3, signed=True) == "+0.123" and fmt(-0.5, 1, signed=True) == "-0.5"
    assert fmt(None, dash="n/a") == "n/a"


def test_fmt_latex_matches_the_old_signed_then_stripped_formatter():
    for v in (0.0, -0.0, 0.004, -0.004, 1.2345, -1.2345, 12, -3, None,
              float("nan"), float("inf"), -float("inf")):
        assert fmt_latex(v) == _old_latex_signed(v), v


def test_fmt_pct():
    assert fmt_pct(None) == "---" and fmt_pct(0.956) == "96\\%" and fmt_pct(0.004) == "0\\%"


def test_write_json_is_indent_two_without_trailing_newline(tmp_path):
    p = tmp_path / "sub" / "x.json"
    write_json([{"a": 1}], p)
    assert p.read_text() == json.dumps([{"a": 1}], indent=2)


def test_write_csv_default_columns_and_ignored_extras(tmp_path):
    p = tmp_path / "x.csv"
    write_csv([{"a": 1, "b": 2}, {"a": 3, "b": 4, "c": "extra"}], p)
    rows = list(csv.DictReader(p.open()))
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_write_csv_explicit_fieldnames_writes_a_header_even_when_empty(tmp_path):
    p = tmp_path / "x.csv"
    write_csv([], p, fieldnames=["m", "s"])
    assert p.read_bytes() == b"m,s\r\n"   # csv line terminator, read raw
    q = tmp_path / "y.csv"
    write_csv([], q)
    assert not q.exists(), "no columns to name, so nothing is written"
