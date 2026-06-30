# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""C++ CLI compatibility test — runs only when SKETCHSORT_CPP_CLI_BIN is set.

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

1. test_cpp_cli_pair_set_matches_golden — cross-platform: same id-pair set,
   cos_dist values close within float32 tolerance. The golden was generated
   on one platform; libm/FMA/SIMD differences across OS/arch can perturb
   cos_dist by a few ULPs even with the same seed and algorithm.

2. test_cpp_cli_matches_python_run_from_file — byte-exact: when the CLI and
   the .so are built in the same env (same compiler, same flags, same libm),
   their text output must match exactly.
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


def _load_pairs(path: str) -> np.ndarray:
    out = np.loadtxt(path, dtype=PAIR_DTYPE)
    order = np.lexsort((out["id2"], out["id1"]))
    return out[order]


@needs_cpp
def test_cpp_cli_pair_set_matches_golden(tmp_path):
    out = tmp_path / "cpp_out.txt"
    subprocess.run(
        [CPP_BIN, "-cosdist", "0.01", "-seed", "42", SAMPLE, str(out)],
        check=True, capture_output=True,
    )
    got    = _load_pairs(str(out))
    golden = _load_pairs(GOLDEN)

    got_set    = set(zip(got["id1"].tolist(),    got["id2"].tolist()))
    golden_set = set(zip(golden["id1"].tolist(), golden["id2"].tolist()))
    assert got_set == golden_set, (
        f"id-pair set diverged. "
        f"only_in_cpp={len(got_set - golden_set)}, only_in_golden={len(golden_set - got_set)}"
    )
    np.testing.assert_allclose(
        got["cos_dist"], golden["cos_dist"],
        rtol=1e-4, atol=1e-6,
    )


@needs_cpp
def test_cpp_cli_matches_python_run_from_file(tmp_path):
    """Byte-exact: same machine, same compiler env → identical text output."""
    import sketchsort
    py_out  = tmp_path / "py.txt"
    cpp_out = tmp_path / "cpp.txt"

    sketchsort.run_from_file(SAMPLE, str(py_out), cos_dist=0.01, seed=42)
    subprocess.run(
        [CPP_BIN, "-cosdist", "0.01", "-seed", "42", SAMPLE, str(cpp_out)],
        check=True, capture_output=True,
    )
    assert filecmp.cmp(py_out, cpp_out, shallow=False)
