# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""Smoke tests for the `sketchsort` console script installed via pyproject."""

import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "dat", "sample.txt")


def _resolve_script():
    # Prefer the installed entry point if present, else fall back to invoking
    # the module so the test still works in editable installs without PATH set.
    path = shutil.which("sketchsort")
    if path:
        return [path]
    return [sys.executable, "-m", "sketchsort.cli"]


def test_cli_runs_on_sample(tmp_path):
    out_path = tmp_path / "out.txt"
    cmd = _resolve_script() + [
        "-cosdist", "0.01",
        "-seed", "42",
        SAMPLE,
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"stderr was: {result.stderr}"
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    with out_path.open() as fh:
        first = fh.readline().split()
    assert len(first) == 3
    int(first[0])    # id1
    int(first[1])    # id2
    float(first[2])  # cos_dist


def test_cli_reports_missing_input(tmp_path):
    out_path = tmp_path / "out.txt"
    cmd = _resolve_script() + [
        "-cosdist", "0.01",
        str(tmp_path / "does_not_exist.txt"),
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "sketchsort:" in result.stderr
