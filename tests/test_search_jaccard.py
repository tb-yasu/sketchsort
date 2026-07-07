# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2011 Yasuo Tabei
"""Tests for sketchsort.search_jaccard() / run_from_file_jaccard().

Jaccard/Tanimoto all-pairs search over sparse integer-id sets, sketched with
MinHash. Unlike the cosine and min-max cores, the whole jaccard pipeline is
integer- and byte-only (deterministic MinHash permutations + exact
intersection/union), so results are reproducible across platforms.

Correctness is checked against a brute-force reference: reported pairs must be
a subset of the true within-threshold neighbours (no false positives), and
auto mode recovers essentially all of them.
"""

import os
import numpy as np
import pytest

import sketchsort

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "dat", "sample_jaccard.txt")
GOLDEN = os.path.join(ROOT, "golden", "sample_jaccard.jac005.seed42.txt")

JAC_DTYPE = np.dtype([("id1", "<u4"), ("id2", "<u4"), ("jaccard_dist", "<f4")])


def _load_sets(path, max_rows=None):
    sets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sets.append(sorted(set(int(x) for x in line.split())))
            if max_rows is not None and len(sets) >= max_rows:
                break
    return sets


def _true_pairs(sets, thr):
    bs = [frozenset(s) for s in sets]
    out = {}
    for i in range(len(bs)):
        a = bs[i]
        if not a:
            continue
        for j in range(i + 1, len(bs)):
            b = bs[j]
            inter = len(a & b)
            if inter == 0:
                continue
            union = len(a) + len(b) - inter
            d = 1.0 - inter / union
            if d <= thr:
                out[(i, j)] = d
    return out


def _pairset(arr):
    s = {}
    for r in np.atleast_1d(arr):
        i, j = int(r["id1"]), int(r["id2"])
        if i > j:
            i, j = j, i
        s[(i, j)] = float(r["jaccard_dist"])
    return s


def test_dtype_is_structured():
    out = sketchsort.search_jaccard([[1, 2, 3], [1, 2, 3], [4, 5]],
                                    jaccard_dist=0.1, num_blocks=2, num_chunks=2, seed=0)
    assert out.dtype == JAC_DTYPE


def test_finds_identical_set():
    # Rows 0 and 1 are the same set → Jaccard distance 0.
    sets = [[1, 5, 9], [1, 5, 9], [2, 3], [9, 100]]
    out = sketchsort.search_jaccard(sets, jaccard_dist=0.01,
                                    num_blocks=2, num_chunks=2, seed=1)
    ps = _pairset(out)
    assert (0, 1) in ps
    np.testing.assert_allclose(ps[(0, 1)], 0.0, atol=1e-6)


def test_duplicate_ids_within_set_are_deduped():
    # [5] and [5, 5] denote the same set → distance 0, not negative.
    out = sketchsort.search_jaccard([[5], [5, 5]], jaccard_dist=0.01,
                                    num_blocks=2, num_chunks=2, seed=1)
    ps = _pairset(out)
    assert ps.get((0, 1)) == 0.0


def test_empty_sets_keep_ids():
    # Empty sets must not shift ids: row 3 keeps id 3.
    sets = [[1, 2], [], [1, 2], []]
    out = sketchsort.search_jaccard(sets, jaccard_dist=0.01,
                                    num_blocks=2, num_chunks=2, seed=1)
    ps = _pairset(out)
    # (0,2) are identical and must be found; empty rows never pair.
    assert (0, 2) in ps
    assert all(1 not in k and 3 not in k for k in ps)


def test_rejects_negative_id():
    with pytest.raises((TypeError, ValueError, OverflowError)):
        sketchsort.search_jaccard([[1, 2], [-3, 4]], jaccard_dist=0.1,
                                  num_blocks=2, num_chunks=2)


def test_rejects_empty_collection():
    with pytest.raises((RuntimeError, ValueError)):
        sketchsort.search_jaccard([], jaccard_dist=0.1)


def test_deterministic_with_seed():
    sets = _load_sets(SAMPLE, max_rows=1000)
    a = sketchsort.search_jaccard(sets, jaccard_dist=0.05, seed=123)
    b = sketchsort.search_jaccard(sets, jaccard_dist=0.05, seed=123)
    np.testing.assert_array_equal(a["id1"], b["id1"])
    np.testing.assert_array_equal(a["id2"], b["id2"])
    np.testing.assert_array_equal(a["jaccard_dist"], b["jaccard_dist"])


def test_seed_changes_output():
    # Under non-exhaustive manual params (recall < 100%), different seeds give
    # different candidate sets. (In auto mode recall is ~100% for both, so the
    # result would be the full truth either way.)
    sets = _load_sets(SAMPLE, max_rows=3000)
    a = sketchsort.search_jaccard(sets, jaccard_dist=0.05, ham_dist=1, num_blocks=3, num_chunks=3, seed=1)
    b = sketchsort.search_jaccard(sets, jaccard_dist=0.05, ham_dist=1, num_blocks=3, num_chunks=3, seed=2)
    assert not (a.shape == b.shape and np.array_equal(a, b))


def test_jaccard_dist_within_threshold():
    sets = _load_sets(SAMPLE, max_rows=3000)
    threshold = 0.03
    out = sketchsort.search_jaccard(sets, jaccard_dist=threshold, seed=42)
    if out.size:
        assert out["jaccard_dist"].max() <= threshold + 1e-6


def test_no_false_positives_and_high_recall():
    sets = _load_sets(SAMPLE, max_rows=3000)
    threshold = 0.05
    truth = _true_pairs(sets, threshold)
    out = sketchsort.search_jaccard(sets, jaccard_dist=threshold, missing_ratio=1e-4, seed=42)
    got = _pairset(out)

    false_pos = [k for k in got if k not in truth]
    assert not false_pos, f"{len(false_pos)} false positives, e.g. {false_pos[:3]}"

    if truth:
        recall = len(set(got) & set(truth)) / len(truth)
        assert recall >= 0.99, f"recall {recall:.3f} below 0.99"

    for k in set(got) & set(truth):
        assert abs(got[k] - truth[k]) < 1e-5


def test_search_matches_run_from_file(tmp_path):
    sets = _load_sets(SAMPLE, max_rows=2000)
    mem = _pairset(sketchsort.search_jaccard(sets, jaccard_dist=0.05, seed=42))

    inp = tmp_path / "in.txt"
    inp.write_text("".join(" ".join(map(str, s)) + "\n" for s in sets))
    out = tmp_path / "out.txt"
    sketchsort.run_from_file_jaccard(str(inp), str(out), jaccard_dist=0.05, seed=42)
    file = _pairset(np.loadtxt(str(out), dtype=JAC_DTYPE))

    assert set(mem) == set(file)


def test_golden_regression(tmp_path):
    # The jaccard pipeline is fully integer/byte deterministic (MinHash with a
    # hand-rolled mt19937 Fisher-Yates + exact intersection/union), so the
    # golden should match essentially exactly on every platform.
    out = tmp_path / "jac.txt"
    sketchsort.run_from_file_jaccard(SAMPLE, str(out),
                                     jaccard_dist=0.05, missing_ratio=1e-4, seed=42)
    got = _pairset(np.loadtxt(str(out), dtype=JAC_DTYPE))
    golden = _pairset(np.loadtxt(GOLDEN, dtype=JAC_DTYPE))

    sym_diff = set(got) ^ set(golden)
    union = set(got) | set(golden)
    allowed = max(20, int(len(union) * 0.001))
    assert len(sym_diff) <= allowed, (
        f"jaccard pair set diverged from golden by {len(sym_diff)} of "
        f"{len(union)} (allowed <= {allowed}); sample: {list(sym_diff)[:5]}"
    )
    common = sorted(set(got) & set(golden))
    assert common
    gv = np.array([got[k] for k in common])
    xv = np.array([golden[k] for k in common])
    np.testing.assert_allclose(gv, xv, rtol=1e-4, atol=1e-6)


def test_rejects_ham_dist_not_less_than_num_blocks():
    with pytest.raises(ValueError):
        sketchsort.search_jaccard([[1, 2], [2, 3]], ham_dist=5, num_blocks=4, num_chunks=3)


@pytest.mark.parametrize("bad_dist", [-0.1, 1.5])
def test_rejects_out_of_range_jaccard_dist(bad_dist):
    with pytest.raises(ValueError):
        sketchsort.search_jaccard([[1, 2], [2, 3]], jaccard_dist=bad_dist,
                                  ham_dist=1, num_blocks=4, num_chunks=3)


@pytest.mark.parametrize("bad_seed", [-1, 2**32])
def test_rejects_out_of_range_seed(bad_seed):
    with pytest.raises(ValueError):
        sketchsort.search_jaccard([[1, 2], [2, 3]], seed=bad_seed)
