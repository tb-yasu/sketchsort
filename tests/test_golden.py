# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""Cross-platform regression test: Python run_from_file() vs the golden file.

The golden file was generated on a specific platform (macOS arm64, CMake
Release build). On other platforms — Linux x86_64, Windows — the *id-pair set*
is deterministic given seed, but the textual cos_dist values can differ in
the last few bits because libm, FMA, and SIMD vectorization differ across
platforms. So we compare:

  - id pair set: must be EXACTLY equal
  - cos_dist values: must be close within float32 tolerance (rtol=1e-4)

If you need a byte-exact same-machine check, see tests/test_cpp_cli.py
(test_cpp_cli_matches_python_run_from_file).
"""

import os
import numpy as np

import sketchsort

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "dat", "sample.txt")
GOLDEN = os.path.join(ROOT, "golden", "sample.cos001.seed42.txt")

PAIR_DTYPE = np.dtype([("id1", "<u4"), ("id2", "<u4"), ("cos_dist", "<f4")])


def _load_pairs(path: str) -> np.ndarray:
    out = np.loadtxt(path, dtype=PAIR_DTYPE)
    # sort by (id1, id2) so two outputs with different report ordering compare cleanly
    order = np.lexsort((out["id2"], out["id1"]))
    return out[order]


def test_run_from_file_pair_set_matches_golden(tmp_path):
    out_path = tmp_path / "py_run_from_file.txt"
    sketchsort.run_from_file(
        input_path=SAMPLE,
        output_path=str(out_path),
        cos_dist=0.01,
        seed=42,
    )

    got    = _load_pairs(str(out_path))
    golden = _load_pairs(GOLDEN)

    got_set    = set(zip(got["id1"].tolist(),    got["id2"].tolist()))
    golden_set = set(zip(golden["id1"].tolist(), golden["id2"].tolist()))

    only_in_got    = got_set - golden_set
    only_in_golden = golden_set - got_set
    assert not only_in_got and not only_in_golden, (
        f"id-pair set diverged from golden. "
        f"only_in_output={len(only_in_got)}, only_in_golden={len(only_in_golden)}. "
        f"sample diff: got={list(only_in_got)[:3]} golden={list(only_in_golden)[:3]}"
    )


def test_run_from_file_cos_dist_close_to_golden(tmp_path):
    out_path = tmp_path / "py_run_from_file.txt"
    sketchsort.run_from_file(
        input_path=SAMPLE,
        output_path=str(out_path),
        cos_dist=0.01,
        seed=42,
    )

    got    = _load_pairs(str(out_path))
    golden = _load_pairs(GOLDEN)
    assert len(got) == len(golden)

    # rtol/atol are generous because golden may have been generated on a
    # different OS/arch; the algorithm is the same but FP rounding differs.
    np.testing.assert_allclose(
        got["cos_dist"], golden["cos_dist"],
        rtol=1e-4, atol=1e-6,
    )
