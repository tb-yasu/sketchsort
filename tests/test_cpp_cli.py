# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""C++ CLI compatibility tests — run only when SKETCHSORT_CPP_CLI_BIN is set.

This is intentionally excluded from cibuildwheel's CIBW_TEST_COMMAND because
the wheel does not ship the C++ binary. To run it, build the CLI separately:

    cmake -B build_cli \
          -DSKETCHSORT_BUILD_CLI=ON \
          -DSKETCHSORT_BUILD_PYTHON=OFF \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build build_cli --target sketchsort_cli

then:

    SKETCHSORT_CPP_CLI_BIN=$PWD/build_cli/sketchsort pytest tests/test_cpp_cli.py

Two complementary checks:

1. test_cpp_cli_pair_set_close_to_golden
   Cross-platform: id-pair set matches the golden within a small symmetric
   difference (boundary pairs can differ across OS due to libm/FMA effects).
   cos_dist on the intersection compared with float32 tolerance.

2. test_cpp_cli_matches_python_run_from_file
   Byte-exact: when the CLI and the .so are built in the same env (same
   compiler, same flags, same libm), their text output must match.
"""

import os
import filecmp
import subprocess
import numpy as np

import pytest

CPP_BIN = os.environ.get("SKETCHSORT_CPP_CLI_BIN")
needs_cpp = pytest.mark.skipif(
    not CPP_BIN or not os.path.isfile(CPP_BIN) or not os.access(CPP_BIN, os.X_OK),
    reason="set SKETCHSORT_CPP_CLI_BIN to point at a CMake-built sketchsort_cli",
)

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "dat", "sample.txt")
GOLDEN = os.path.join(ROOT, "golden", "sample.cos001.seed42.txt")

PAIR_DTYPE = np.dtype([("id1", "<u4"), ("id2", "<u4"), ("cos_dist", "<f4")])


def _load_as_dict(path: str) -> dict:
    arr = np.loadtxt(path, dtype=PAIR_DTYPE)
    return {(int(r["id1"]), int(r["id2"])): float(r["cos_dist"]) for r in arr}


@needs_cpp
def test_cpp_cli_pair_set_close_to_golden(tmp_path):
    out = tmp_path / "cpp_out.txt"
    subprocess.run(
        [CPP_BIN, "-cosdist", "0.01", "-seed", "42",
         "-auto", "-missingratio", "0.0001",
         SAMPLE, str(out)],
        check=True, capture_output=True,
    )
    got_d    = _load_as_dict(str(out))
    golden_d = _load_as_dict(GOLDEN)

    sym_diff = set(got_d) ^ set(golden_d)
    union    = set(got_d) | set(golden_d)
    allowed  = max(100, int(len(union) * 0.001))
    assert len(sym_diff) <= allowed, (
        f"C++ CLI vs golden diverged by {len(sym_diff)} pairs out of {len(union)} "
        f"(allowed <= {allowed}). sample: {list(sym_diff)[:5]}"
    )

    common = sorted(set(got_d) & set(golden_d))
    got_arr    = np.array([got_d[k]    for k in common], dtype=np.float64)
    golden_arr = np.array([golden_d[k] for k in common], dtype=np.float64)
    np.testing.assert_allclose(got_arr, golden_arr, rtol=1e-3, atol=1e-5)


@needs_cpp
def test_cpp_cli_matches_python_run_from_file(tmp_path):
    """Byte-exact: same machine, same compiler env → identical text output."""
    import sketchsort
    py_out  = tmp_path / "py.txt"
    cpp_out = tmp_path / "cpp.txt"

    sketchsort.run_from_file(SAMPLE, str(py_out), cos_dist=0.01, seed=42)
    subprocess.run(
        [CPP_BIN, "-cosdist", "0.01", "-seed", "42",
         "-auto", "-missingratio", "0.0001",
         SAMPLE, str(cpp_out)],
        check=True, capture_output=True,
    )
    assert filecmp.cmp(py_out, cpp_out, shallow=False)


@needs_cpp
def test_cpp_cli_missing_outfile_exits_cleanly():
    """OUTFILE omitted used to segfault (exit 139); it must now exit 1."""
    result = subprocess.run(
        [CPP_BIN, "-cosdist", "0.01", "-seed", "42", SAMPLE],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert result.stderr.strip() != ""


@needs_cpp
def test_cpp_cli_trailing_flag_without_value_exits_cleanly():
    """A value-taking flag with nothing after it used to segfault via atoi(NULL)."""
    result = subprocess.run(
        [CPP_BIN, "-seed", "42", "-cosdist"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1


@needs_cpp
def test_cpp_cli_non_numeric_flag_value_exits_cleanly(tmp_path):
    out = tmp_path / "out.txt"
    result = subprocess.run(
        [CPP_BIN, "-hamdist", "abc", "-seed", "42", SAMPLE, str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1


@needs_cpp
def test_cpp_cli_hamdist_not_less_than_numblocks_exits_cleanly(tmp_path):
    """ham_dist >= num_blocks used to silently drop every pair via unsigned underflow."""
    out = tmp_path / "out.txt"
    result = subprocess.run(
        [CPP_BIN, "-hamdist", "5", "-numblocks", "4", "-seed", "42", SAMPLE, str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1


@needs_cpp
def test_cpp_cli_negative_hamdist_exits_cleanly(tmp_path):
    out = tmp_path / "out.txt"
    result = subprocess.run(
        [CPP_BIN, "-hamdist", "-1", "-seed", "42", SAMPLE, str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert result.stderr.strip() != ""


@needs_cpp
def test_cpp_cli_version_exits_zero():
    result = subprocess.run([CPP_BIN, "-version"], capture_output=True, text=True)
    assert result.returncode == 0
