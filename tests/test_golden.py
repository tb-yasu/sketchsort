# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""Byte-exact comparison: Python run_from_file() vs golden file.

Both paths flow through the same C++ output formatter in SketchSort::run(),
so they must produce identical bytes when seed and parameters match.

If the golden was regenerated against a different build (e.g. the Makefile
binary uses -ffast-math while the .so does not), the assertion can fail.
The golden in this repo was generated with the CMake Release build, which is
the same configuration scikit-build-core uses for the wheel.
"""

import filecmp
import os

import sketchsort

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "dat", "sample.txt")
GOLDEN = os.path.join(ROOT, "golden", "sample.cos001.seed42.txt")


def test_run_from_file_matches_golden(tmp_path):
    out_path = tmp_path / "py_run_from_file.txt"
    sketchsort.run_from_file(
        input_path=SAMPLE,
        output_path=str(out_path),
        cos_dist=0.01,
        seed=42,
    )
    assert filecmp.cmp(out_path, GOLDEN, shallow=False), (
        "Python run_from_file() output diverged from golden — "
        "rebuild the .so and re-run; if the divergence persists the golden "
        "may need regeneration with the current CMake build."
    )
