# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""C++ CLI compatibility test — runs only when SKETCHSORT_CPP_CLI_BIN is set.

This is intentionally excluded from cibuildwheel's CIBW_TEST_COMMAND because
the wheel does not ship the C++ binary. To run it, build the CLI separately:

    cmake -B build_cli \\
          -DSKETCHSORT_BUILD_CLI=ON \\
          -DSKETCHSORT_BUILD_PYTHON=OFF \\
          -DCMAKE_BUILD_TYPE=Release
    cmake --build build_cli --target sketchsort_cli

then:

    SKETCHSORT_CPP_CLI_BIN=$PWD/build_cli/sketchsort pytest tests/test_cpp_cli.py

For the byte-exact comparison to hold, the CLI must be built with the same
CMake configuration as the wheel — in particular, Release mode without
-ffast-math. The legacy Makefile under src/ uses -ffast-math and will NOT
match the .so byte-for-byte (the pair set is the same but cos_dist text
representation differs).
"""

import os
import filecmp
import subprocess

import pytest

CPP_BIN = os.environ.get("SKETCHSORT_CPP_CLI_BIN")
needs_cpp = pytest.mark.skipif(
    not CPP_BIN or not os.path.isfile(CPP_BIN) or not os.access(CPP_BIN, os.X_OK),
    reason="set SKETCHSORT_CPP_CLI_BIN to point at a CMake-built sketchsort_cli",
)

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "dat", "sample.txt")


@needs_cpp
def test_cpp_cli_matches_golden(tmp_path):
    out = tmp_path / "cpp_out.txt"
    subprocess.run(
        [CPP_BIN, "-cosdist", "0.01", "-seed", "42", SAMPLE, str(out)],
        check=True, capture_output=True,
    )
    golden = os.path.join(ROOT, "golden", "sample.cos001.seed42.txt")
    assert filecmp.cmp(out, golden, shallow=False)


@needs_cpp
def test_cpp_cli_matches_python_run_from_file(tmp_path):
    import sketchsort
    py_out  = tmp_path / "py.txt"
    cpp_out = tmp_path / "cpp.txt"

    sketchsort.run_from_file(SAMPLE, str(py_out), cos_dist=0.01, seed=42)
    subprocess.run(
        [CPP_BIN, "-cosdist", "0.01", "-seed", "42", SAMPLE, str(cpp_out)],
        check=True, capture_output=True,
    )
    assert filecmp.cmp(py_out, cpp_out, shallow=False)
