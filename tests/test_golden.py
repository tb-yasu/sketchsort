# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""Cross-platform regression test: Python run_from_file() vs the golden file.

The golden file is generated on one platform (macOS arm64, CMake Release).
On other platforms — Linux x86_64, Windows — algorithmic determinism holds
for everything *except* the final `cos_dist <= threshold` comparison: libm
and FMA differences shift checkCos() by a few ULPs, so pairs whose true
cos_dist is right at the boundary may be admitted on one OS and rejected on
another. The expected divergence is ~0 to ~few dozen out of ~132 000.

So this test:
  - Allows the id-pair symmetric difference to be small (<= 0.1% of |union|)
  - Compares cos_dist values on the INTERSECTION with float32 tolerance

For a true byte-exact same-machine check, see
tests/test_cpp_cli.py::test_cpp_cli_matches_python_run_from_file.
"""

import os
import numpy as np

import sketchsort

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "dat", "sample.txt")
GOLDEN = os.path.join(ROOT, "golden", "sample.cos001.seed42.txt")

PAIR_DTYPE = np.dtype([("id1", "<u4"), ("id2", "<u4"), ("cos_dist", "<f4")])


def _load_as_dict(path: str) -> dict:
    arr = np.loadtxt(path, dtype=PAIR_DTYPE)
    return {(int(r["id1"]), int(r["id2"])): float(r["cos_dist"]) for r in arr}


def _run_from_file(tmp_path):
    out_path = tmp_path / "py_run_from_file.txt"
    sketchsort.run_from_file(
        input_path=SAMPLE,
        output_path=str(out_path),
        cos_dist=0.01,
        seed=42,
    )
    return str(out_path)


def test_pair_set_close_to_golden(tmp_path):
    got_d    = _load_as_dict(_run_from_file(tmp_path))
    golden_d = _load_as_dict(GOLDEN)

    got_set    = set(got_d)
    golden_set = set(golden_d)
    sym_diff   = got_set ^ golden_set
    union      = got_set | golden_set

    # Boundary pairs (cos_dist ≈ 0.01) can flip across platforms due to
    # libm/FMA differences in checkCos(). Cap at 0.1% of |union| or 100,
    # whichever is larger — well above any real expected drift.
    allowed = max(100, int(len(union) * 0.001))
    assert len(sym_diff) <= allowed, (
        f"id-pair set diverged from golden by {len(sym_diff)} pairs out of "
        f"{len(union)} (allowed <= {allowed}). This suggests platform-level "
        f"non-determinism beyond expected libm boundary effects. "
        f"sample: {list(sym_diff)[:5]}"
    )


def test_cos_dist_close_to_golden_on_intersection(tmp_path):
    got_d    = _load_as_dict(_run_from_file(tmp_path))
    golden_d = _load_as_dict(GOLDEN)

    common = sorted(set(got_d) & set(golden_d))
    assert len(common) > 0, "empty intersection — output is completely wrong"

    got_arr    = np.array([got_d[k]    for k in common], dtype=np.float64)
    golden_arr = np.array([golden_d[k] for k in common], dtype=np.float64)

    # rtol / atol intentionally generous: golden cos_dist is float32 round-
    # tripped through text, and the OS may compute checkCos slightly
    # differently. We only check magnitude is the same.
    np.testing.assert_allclose(
        got_arr, golden_arr,
        rtol=1e-3, atol=1e-5,
    )
