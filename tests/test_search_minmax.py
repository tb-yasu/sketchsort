# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2017 Yasuo Tabei
"""Tests for sketchsort.search_minmax() / run_from_file_minmax().

Min-max (generalized Jaccard / Tanimoto) all-pairs search over dense real
vectors, sketched with generalized consistent weighted sampling (GCWS).

Correctness is checked against a brute-force NumPy reference: reported pairs
must be a subset of the true within-threshold neighbours (no false positives),
and auto mode must recover essentially all of them.
"""

import os
import numpy as np
import pytest

import sketchsort

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "dat", "sample_minmax.txt")
GOLDEN = os.path.join(ROOT, "golden", "sample_minmax.mm03.seed42.txt")

MM_DTYPE = np.dtype([("id1", "<u4"), ("id2", "<u4"), ("minmax_dist", "<f4")])


def _true_minmax_dists(X):
    """Full brute-force min-max distance matrix (upper triangle as a dict)."""
    N = X.shape[0]
    pairs = {}
    for i in range(N):
        a = X[i]
        aa = np.abs(a)[None, :]
        ab = np.abs(X)
        pos = (a > 0)[None, :] & (X > 0)
        neg = (a < 0)[None, :] & (X < 0)
        num = (np.where(pos, np.minimum(a[None, :], X), 0).sum(1)
               + np.where(neg, np.minimum(aa, ab), 0).sum(1))
        den = (np.where(pos, np.maximum(a[None, :], X), 0).sum(1)
               + np.where(neg, np.maximum(aa, ab), 0).sum(1)
               + np.where(~(pos | neg), aa + ab, 0).sum(1))
        d = np.where(den < 1e-30, 1.0, 1.0 - num / den)
        for j in range(i + 1, N):
            pairs[(i, j)] = float(d[j])
    return pairs


def _pairset(arr):
    s = {}
    for r in np.atleast_1d(arr):
        i, j = int(r["id1"]), int(r["id2"])
        if i > j:
            i, j = j, i
        s[(i, j)] = float(r["minmax_dist"])
    return s


def test_dtype_is_structured():
    X = np.eye(4, dtype=np.float32)
    out = sketchsort.search_minmax(X, minmax_dist=0.5, seed=0,
                                   ham_dist=1, num_blocks=4, num_chunks=3)
    assert out.dtype == MM_DTYPE


def test_finds_known_duplicate():
    # Rows 0 and 3 are identical → min-max distance 0.
    X = np.array(
        [
            [3.0, 0.0, 1.0, 0.0, 2.0],
            [0.0, 4.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 1.0],
            [3.0, 0.0, 1.0, 0.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    out = sketchsort.search_minmax(X, minmax_dist=0.05, seed=42,
                                   num_blocks=2, num_chunks=2)
    ps = _pairset(out)
    assert (0, 3) in ps, f"missing identical pair, got {ps}"
    np.testing.assert_allclose(ps[(0, 3)], 0.0, atol=1e-6)


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        sketchsort.search_minmax(np.zeros(5, dtype=np.float32))


def test_rejects_empty():
    with pytest.raises((RuntimeError, ValueError)):
        sketchsort.search_minmax(np.zeros((0, 8), dtype=np.float32))


def test_deterministic_with_seed():
    X = np.loadtxt(SAMPLE, dtype=np.float32, max_rows=300)
    a = sketchsort.search_minmax(X, minmax_dist=0.3, seed=123)
    b = sketchsort.search_minmax(X, minmax_dist=0.3, seed=123)
    np.testing.assert_array_equal(a["id1"], b["id1"])
    np.testing.assert_array_equal(a["id2"], b["id2"])
    np.testing.assert_array_equal(a["minmax_dist"], b["minmax_dist"])


def test_seed_changes_output():
    X = np.loadtxt(SAMPLE, dtype=np.float32, max_rows=500)
    a = sketchsort.search_minmax(X, minmax_dist=0.3, seed=1)
    b = sketchsort.search_minmax(X, minmax_dist=0.3, seed=2)
    assert not (a.shape == b.shape and np.array_equal(a, b))


def test_minmax_dist_within_threshold():
    X = np.loadtxt(SAMPLE, dtype=np.float32)
    threshold = 0.25
    out = sketchsort.search_minmax(X, minmax_dist=threshold, seed=42)
    if out.size:
        assert out["minmax_dist"].max() <= threshold + 1e-6


def test_no_false_positives_and_high_recall():
    # Auto mode guarantees missing ratio <= missing_ratio at the boundary, so
    # recall over all true within-threshold pairs should be essentially 1.0,
    # with zero false positives (reported pairs are a subset of truth).
    X = np.loadtxt(SAMPLE, dtype=np.float32, max_rows=400)
    threshold = 0.25
    truth = {k: v for k, v in _true_minmax_dists(X).items() if v <= threshold}
    out = sketchsort.search_minmax(X, minmax_dist=threshold, missing_ratio=1e-3, seed=42)
    got = _pairset(out)

    false_pos = [k for k in got if k not in truth]
    assert not false_pos, f"{len(false_pos)} false positives, e.g. {false_pos[:3]}"

    if truth:
        recall = len(set(got) & set(truth)) / len(truth)
        assert recall >= 0.98, f"recall {recall:.3f} below 0.98"

    # Reported distances agree with the exact ones.
    for k in set(got) & set(truth):
        assert abs(got[k] - truth[k]) < 1e-5


def test_search_matches_run_from_file(tmp_path):
    # The in-memory path and the file path must agree pair-for-pair.
    X = np.loadtxt(SAMPLE, dtype=np.float32)
    mem = _pairset(sketchsort.search_minmax(X, minmax_dist=0.3, seed=42))

    out = tmp_path / "mm.txt"
    sketchsort.run_from_file_minmax(SAMPLE, str(out), minmax_dist=0.3, seed=42)
    file = _pairset(np.loadtxt(str(out), dtype=MM_DTYPE))

    assert set(mem) == set(file)


def test_golden_regression(tmp_path):
    # Cross-platform regression: the vendored Boost RNG is identical on every
    # platform, so the projection matrices match exactly; only the final libm
    # distance comparison can flip a few boundary pairs. Allow a small
    # symmetric difference, like the cosine golden test.
    out = tmp_path / "mm.txt"
    sketchsort.run_from_file_minmax(SAMPLE, str(out),
                                    minmax_dist=0.3, missing_ratio=1e-4, seed=42)
    got = _pairset(np.loadtxt(str(out), dtype=MM_DTYPE))
    golden = _pairset(np.loadtxt(GOLDEN, dtype=MM_DTYPE))

    sym_diff = set(got) ^ set(golden)
    union = set(got) | set(golden)
    allowed = max(20, int(len(union) * 0.01))
    assert len(sym_diff) <= allowed, (
        f"min-max pair set diverged from golden by {len(sym_diff)} of "
        f"{len(union)} (allowed <= {allowed}); sample: {list(sym_diff)[:5]}"
    )

    common = sorted(set(got) & set(golden))
    assert common, "empty intersection with golden"
    gv = np.array([got[k] for k in common])
    xv = np.array([golden[k] for k in common])
    np.testing.assert_allclose(gv, xv, rtol=1e-3, atol=1e-5)


def test_rejects_ham_dist_not_less_than_num_blocks():
    X = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError):
        sketchsort.search_minmax(X, ham_dist=5, num_blocks=4, num_chunks=3)


@pytest.mark.parametrize("bad_dist", [-0.1, 1.5])
def test_rejects_out_of_range_minmax_dist(bad_dist):
    X = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError):
        sketchsort.search_minmax(X, minmax_dist=bad_dist,
                                 ham_dist=1, num_blocks=4, num_chunks=3)


def test_rejects_seed_below_minus_one():
    X = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError):
        sketchsort.search_minmax(X, seed=-5)


def test_normalization_runs():
    X = np.loadtxt(SAMPLE, dtype=np.float32, max_rows=200)
    a = sketchsort.search_minmax(X, minmax_dist=0.3, seed=42, z_normalization=True)
    b = sketchsort.search_minmax(X, minmax_dist=0.3, seed=42, minmax_normalization=True)
    assert a.dtype == MM_DTYPE and b.dtype == MM_DTYPE
