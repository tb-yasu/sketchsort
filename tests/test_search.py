# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""Tests for sketchsort.search() — the NumPy-native API.

Uses ID-set membership + numerical tolerance comparison, NOT byte-exact text
diff, because cos_dist is a float32 whose textual representation can vary
across builds.
"""

import os
import numpy as np
import pytest

import sketchsort

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "dat", "sample.txt")


def test_dtype_is_structured():
    # Use a tight cos_dist + explicit manual params to keep this trivial test fast
    # (auto mode does extra work searching for ham_dist/num_blocks/num_chunks).
    X = np.eye(4, dtype=np.float32)
    out = sketchsort.search(X, cos_dist=0.01, seed=0,
                            ham_dist=1, num_blocks=4, num_chunks=3)
    assert out.dtype == np.dtype([("id1", "<u4"), ("id2", "<u4"), ("cos_dist", "<f4")])


def test_finds_known_duplicate():
    # Row 0 and row 3 are intentionally identical → cos distance is ~0.
    X = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    out = sketchsort.search(X, cos_dist=0.01, seed=42, num_blocks=2, num_chunks=2)

    pair_set = {tuple(sorted((int(r["id1"]), int(r["id2"])))) for r in out}
    assert (0, 3) in pair_set, f"missing identical pair, got {pair_set}"

    for row in out:
        if tuple(sorted((int(row["id1"]), int(row["id2"])))) == (0, 3):
            np.testing.assert_allclose(row["cos_dist"], 0.0, atol=1e-6)


def test_centering_runs():
    rng = np.random.default_rng(7)
    X = rng.normal(loc=5.0, scale=1.0, size=(20, 8)).astype(np.float32)
    out_no_center  = sketchsort.search(X, cos_dist=0.05, seed=42, centering=False)
    out_centering  = sketchsort.search(X, cos_dist=0.05, seed=42, centering=True)
    # Just check both ran and produced structured arrays — centering shifts the
    # geometry so pair counts can differ.
    assert out_no_center.dtype == out_centering.dtype


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        sketchsort.search(np.zeros(5, dtype=np.float32))


def test_rejects_empty():
    with pytest.raises((RuntimeError, ValueError)):
        sketchsort.search(np.zeros((0, 8), dtype=np.float32))


def test_deterministic_with_seed():
    X = np.loadtxt(SAMPLE, dtype=np.float32, max_rows=200)
    a = sketchsort.search(X, cos_dist=0.05, seed=123)
    b = sketchsort.search(X, cos_dist=0.05, seed=123)
    assert a.shape == b.shape
    np.testing.assert_array_equal(a["id1"], b["id1"])
    np.testing.assert_array_equal(a["id2"], b["id2"])
    np.testing.assert_array_equal(a["cos_dist"], b["cos_dist"])


def test_seed_changes_output():
    X = np.loadtxt(SAMPLE, dtype=np.float32, max_rows=500)
    a = sketchsort.search(X, cos_dist=0.05, seed=1)
    b = sketchsort.search(X, cos_dist=0.05, seed=2)
    # Different seeds → not byte-identical (counts can match but the projection
    # is different so per-pair cos_dists differ).
    assert not (a.shape == b.shape and np.array_equal(a, b))


def test_cos_dist_within_threshold():
    X = np.loadtxt(SAMPLE, dtype=np.float32, max_rows=2000)
    threshold = 0.02
    out = sketchsort.search(X, cos_dist=threshold, seed=42)
    if out.size:
        assert out["cos_dist"].max() <= threshold + 1e-6


def test_rejects_ham_dist_not_less_than_num_blocks():
    X = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError):
        sketchsort.search(X, ham_dist=5, num_blocks=4, num_chunks=3)


def test_rejects_zero_num_chunks():
    X = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError):
        sketchsort.search(X, ham_dist=1, num_blocks=4, num_chunks=0)


@pytest.mark.parametrize("bad_cos_dist", [-0.1, 2.5])
def test_rejects_out_of_range_cos_dist(bad_cos_dist):
    X = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError):
        sketchsort.search(X, cos_dist=bad_cos_dist, ham_dist=1, num_blocks=4, num_chunks=3)


@pytest.mark.parametrize("bad_seed", [-1, 2**32])
def test_rejects_out_of_range_seed(bad_seed):
    X = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError):
        sketchsort.search(X, seed=bad_seed)


def test_rejects_zero_norm_row():
    # Row 1 is all-zero: its norm is zero, so a cosine distance to it is
    # undefined. This used to be silently dropped from every pair; it now
    # raises with the offending row number in the message.
    X = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match=r"row 1"):
        sketchsort.search(X, ham_dist=1, num_blocks=4, num_chunks=3)


def test_rejects_nan_row_no_centering():
    X = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match=r"row 1"):
        sketchsort.search(X, ham_dist=1, num_blocks=4, num_chunks=3, centering=False)


def test_rejects_nan_row_with_centering():
    # With centering on, the NaN in row 1 contaminates the column mean, so
    # every row's centered value in that column becomes NaN too -- the
    # error fires on row 0 (the first row checked), not row 1. This pins
    # that centering-specific behavior rather than a specific row number.
    X = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match=r"after centering"):
        sketchsort.search(X, ham_dist=1, num_blocks=4, num_chunks=3, centering=True)


def test_auto_mode_combination_fix_pin(capfd):
    # Pins auto mode's parameter choice at cos_dist=0.35 to the *fixed*
    # combination() (see CLAUDE.md's combination() gotcha): the old
    # integer-division bug picked hamDist=8/numBlocks=11/numChunks=20 for
    # missing_ratio=0.0001; the fix picks hamDist=7/numBlocks=10/
    # numChunks=25, which meets the same recall guarantee more efficiently.
    # This test fails against the pre-fix combination().
    X = np.eye(4, dtype=np.float32)
    sketchsort.search(X, cos_dist=0.35, seed=0, verbose=True)
    out = capfd.readouterr().out
    assert "hamming distance threshold: 7" in out
    assert "number of blocks: 10" in out
    assert "number of chunks: 25" in out
