# sketchsort

Fast all-pairs similarity search via random projection sketches.

`sketchsort` finds all pairs of high-dimensional data whose distance is
below a given threshold, under three similarity measures:

| Metric | Input | Function | Distance |
| --- | --- | --- | --- |
| **Cosine** | dense float vectors | `search` | `1 - ⟨x,y⟩/(‖x‖·‖y‖)` |
| **Min-max** (generalized Jaccard / Tanimoto on real vectors) | dense float vectors | `search_minmax` | `1 - Σ min(xᵢ,yᵢ)/Σ max(xᵢ,yᵢ)` |
| **Jaccard / Tanimoto** | sparse integer-id sets | `search_jaccard` | `1 - |A∩B|/|A∪B|` |

Each metric sketches its input into a short byte string — sign of a random
projection (cosine), generalized consistent weighted sampling (min-max),
or MinHash (Jaccard) — then enumerates near-duplicate sketches using the
shared multiple-sorting technique of Tabei et al. (2010). The
missing-edge-ratio parameter (`missing_ratio`) controls how exhaustive
this enumeration is. See the Python API section below for details.

## Install

```
pip install sketchsort
```

Wheels are provided for CPython 3.9–3.13 on Linux (x86_64), macOS arm64
(Apple Silicon), and Windows (x86_64). Intel macOS is not yet provided as
a wheel; on that platform `pip install sketchsort` will fall back to a
source build (requires a C++17 compiler and CMake).

## Python API

```python
import numpy as np
import sketchsort

X = np.loadtxt("dat/sample.txt", dtype=np.float32)   # shape (N, D)

pairs = sketchsort.search(
    X,
    cos_dist=0.01,         # report pairs with cosine distance <= 0.01
    missing_ratio=0.0001,  # target bound on expected missed true-neighbor fraction
    seed=42,
)

for id1, id2, d in pairs:
    print(id1, id2, d)
```

`X` must be a 2D NumPy array of shape `(N, D)`. `float32` is recommended;
other floating-point dtypes are converted internally. `id1` and `id2` in
the result are row indices in `X`.

`sketchsort.search(...)` returns a NumPy structured array with fields
`id1` (uint32), `id2` (uint32), and `cos_dist` (float32). The output
contains each pair at most once, never the self-pair `(i, i)`. For a
fixed `seed` the output is deterministic, but the order is the algorithm's
internal enumeration order — **not** sorted by distance, and `id1 < id2`
is **not** guaranteed.

Smaller `missing_ratio` makes the search more exhaustive, reducing the
upper bound on the expected fraction of missed true neighbor pairs at
the cost of more time and memory. The bound is derived from the
random-projection model and applies to the expectation, not to every
individual run.

`seed` (default `0`) seeds the random-number generator used to draw the
random projection vectors. For a fixed build, two calls with the same
`seed`, parameters, and inputs produce the same output. To reproduce the
non-deterministic behaviour of upstream 0.0.8, pass a time-based seed
explicitly, e.g. `seed=int(time.time())`.

Two additional optional parameters:

- `centering` (default `False`) — when set to `True`, the coordinate-wise
  mean of `X` is subtracted from every row before both sketching and
  distance computation. In this mode, the reported `cos_dist` is the
  cosine distance between the *mean-shifted* vectors, not the raw input
  vectors. Recommended when input vectors are non-negative and share a
  strong bias (e.g. raw bag-of-words counts, histograms, molecular
  fingerprints — the original SketchSort use case).
- `verbose` (default `False`) — when set to `True`, the underlying C++
  core prints algorithm progress to stdout/stderr. Default is quiet;
  turn on for diagnostics.

### Manual parameter control

For full control of the sketch enumeration, pass all of `ham_dist`,
`num_blocks`, and `num_chunks`. Providing any of these switches the call
into manual mode, where `missing_ratio` is ignored. If you specify only
some of the three, the rest fall back to defaults (`ham_dist=1`,
`num_blocks=4`, `num_chunks=3`), so it is safer to set them together:

```python
pairs = sketchsort.search(
    X,
    cos_dist=0.01,
    ham_dist=1, num_blocks=4, num_chunks=3,
    seed=42,
)
```

### File I/O

If you have a file in the legacy text format and want output compatible
with the CMake-built C++ CLI from the same source tree, call:

```python
sketchsort.run_from_file("input.txt", "output.txt", cos_dist=0.01, seed=42)
```

The input file is whitespace-separated float vectors, one per line, no ID
column. The output file is `id1 id2 cos_dist` triples.

Every line must carry exactly as many numeric tokens as the first line
(that count becomes the file's dimension). Blank or whitespace-only
lines, non-numeric tokens, lines with too many values, and a completely
empty file are all rejected with an error naming the offending line
rather than being silently skipped or truncated.

## Other similarity metrics

Besides cosine, the package exposes two more metrics with the same
auto/manual parameter model and the same reproducible-by-`seed` behaviour.
Each returns a NumPy structured array and has a matching `run_from_file_*`
for streaming file pipelines.

### Min-max (generalized Jaccard on real vectors)

`search_minmax(X, ...)` takes the same dense `(N, D)` float matrix as
`search`, and reports pairs whose min-max distance is at most
`minmax_dist`. Sketches come from generalized consistent weighted sampling
(Li, 2017). Negative values are allowed. Two optional normalizers,
`z_normalization` and `minmax_normalization`, rescale each column before
sketching.

```python
import numpy as np, sketchsort
X = np.loadtxt("dat/sample_minmax.txt", dtype=np.float32)
pairs = sketchsort.search_minmax(X, minmax_dist=0.2, missing_ratio=1e-4, seed=42)
# dtype: [('id1','<u4'), ('id2','<u4'), ('minmax_dist','<f4')]
```

### Jaccard / Tanimoto (sparse integer-id sets)

`search_jaccard(sets, ...)` takes a sequence of integer-id sets — `sets[i]`
is the collection of non-negative ids present in row `i` (e.g. the "on"
bits of a molecular fingerprint) — and reports pairs whose Jaccard
distance is at most `jaccard_dist`. Sketches are MinHash values. Duplicate
ids within a set are removed; empty sets are kept so row `i` always has
id `i`.

```python
import sketchsort
sets = [[0, 3, 7], [0, 3, 7, 9], [1, 2], [7, 9]]
pairs = sketchsort.search_jaccard(sets, jaccard_dist=0.5, missing_ratio=1e-4, seed=42)
# dtype: [('id1','<u4'), ('id2','<u4'), ('jaccard_dist','<f4')]
```

The Jaccard pipeline is fully integer/byte deterministic (MinHash with a
hand-rolled mt19937 Fisher-Yates shuffle plus exact intersection/union),
so for a fixed `seed` its output is identical across platforms. The cosine
and min-max cores are deterministic per platform but their final
distance/threshold comparison can flip a few boundary pairs across
platforms due to libm differences.

`run_from_file_minmax(in, out, ...)` reads dense float vectors (like the
cosine file format); `run_from_file_jaccard(in, out, ...)` reads one
whitespace-separated integer-id set per line.

## Command line

Installing the package also installs a `sketchsort` console script. Pick
the metric with `-metric {cosine,minmax,jaccard}` (default: cosine). It
uses the same defaults as the Python API: automatic parameter selection
unless you pass any of `-hamdist` / `-numblocks` / `-numchunks`.

```
# Cosine (default metric): cos_dist + missing_ratio
sketchsort -cosdist 0.01 -missingratio 0.0001 -seed 42 input.txt output.txt

# Cosine with manual parameter control
sketchsort -cosdist 0.01 -hamdist 1 -numblocks 4 -numchunks 3 -seed 42 input.txt output.txt

# Min-max on dense float vectors
sketchsort -metric minmax -minmax 0.2 -missingratio 0.0001 -seed 42 input.txt output.txt

# Jaccard on integer-id sets (one whitespace-separated set per line)
sketchsort -metric jaccard -jaccard 0.05 -missingratio 0.0001 -seed 42 sets.txt output.txt
```

Shared flags: `-metric`, `-missingratio`, `-hamdist`, `-numblocks`,
`-numchunks`, `-auto`, `-seed`, `-quiet`. `-auto` forces automatic
parameter selection even if any of `-hamdist` / `-numblocks` /
`-numchunks` is also given.

Per-metric flags: `-cosdist` (cosine threshold) and `-centering`
(cosine only: subtract the coordinate-wise mean before both sketching
and distance computation, so the reported `cos_dist` is between
mean-shifted vectors); `-minmax` (min-max threshold), `-znormalization`
and `-minmaxnormalization` (min-max only: per-dimension rescaling before
sketching); `-jaccard` (Jaccard threshold).

## Memory note

The in-memory functions (`search`, `search_minmax`, `search_jaccard`)
collect every reported pair into memory before returning. For large
inputs at loose thresholds the result can be very large (tens of millions
of pairs are realistic). If memory is a concern, use the corresponding
`run_from_file*` function instead — it streams pairs to disk while the
algorithm runs.

## 0.3.0 release notes

- **Two new similarity metrics** alongside cosine, sharing the same
  multiple-sort enumeration engine but living in separate C++ sub-namespaces
  (`sketchsort::minmax`, `sketchsort::jaccard`):
  - `search_minmax` / `run_from_file_minmax` — min-max (generalized Jaccard /
    Tanimoto) on dense real vectors via generalized consistent weighted
    sampling, with optional `z_normalization` / `minmax_normalization`.
  - `search_jaccard` / `run_from_file_jaccard` — Jaccard / Tanimoto on sparse
    integer-id sets via MinHash.
- Both new cores were built with the same in-memory + file APIs, exception
  (not `exit()`) error handling, and deterministic-by-`seed` behaviour as the
  cosine core. The Jaccard MinHash uses a hand-rolled mt19937 Fisher-Yates
  shuffle so it is reproducible across platforms.
- No change to the existing cosine `search` / `run_from_file` API or output.

Internal (no behavior change — output is byte-identical):

- The multiple-sort enumeration machinery (radix/insertion sorting, block
  grouping, canonicality checks, candidate-pair reporting, missing-edge-ratio
  math, parameter auto-selection and validation) previously existed as three
  near-identical copies, one per metric. It now lives once in
  `src/multi_sort_engine.hpp` (`sketchsort::engine::MultiSortEngine`), with
  each metric supplying only its projection, exact-distance verification, and
  an optional O(1) prefilter through an inlined template hook. Verified
  byte-identical output across all three metrics on nine seed/parameter
  configurations, including the `-ffast-math` legacy Makefile build.

## 0.2.0 release notes

Behavior changes from hardening the C++ core and CLI against invalid
input:

- **Invalid input now raises instead of silently producing wrong output.**
  Zero-norm or non-finite (NaN/Inf) rows, out-of-range parameters
  (`cos_dist` outside `[0, 2]`, `missing_ratio` outside `(0, 1)` in auto
  mode, `ham_dist >= num_blocks`, `num_blocks` outside `[1, 32]`,
  `num_chunks < 1`), and malformed input-file lines (blank/whitespace-only
  lines, non-numeric tokens, lines with more values than the file's
  dimension, or a completely empty input file) now raise `ValueError` /
  `RuntimeError` from Python (`std::invalid_argument` / `std::runtime_error`
  from the C++ core) instead of being silently dropped or corrupting
  output.
- **C++ CLI exit codes.** A missing `OUTFILE`, a value-taking flag with no
  value, a non-numeric flag value, or extra positional arguments after
  `OUTFILE` now print an error to stderr and exit with status 1 instead of
  crashing or silently continuing with garbage arguments. `-version` and
  the new `-h`/`--help` exit 0 immediately instead of falling through to a
  (previously crash-prone) attempt to run with missing file arguments.
  Options must precede `INFILE`/`OUTFILE`; anything after them is now a
  positional-argument error rather than being silently ignored.
- **C++ CLI: negative parameter values rejected at parse time.**
  `-hamdist` / `-numblocks` / `-numchunks` are now parsed as unsigned
  integers (matching `-seed`), so a negative value such as `-hamdist -1`
  exits with a clear `invalid unsigned integer value` error instead of
  silently wrapping to a huge value and failing later with a confusing
  out-of-range message.
- **`combination()` integer-division fix.** The internal binomial
  coefficient helper used by auto mode's parameter search performed
  integer division and underestimated `C(n, m)` for `m >= 3`. Auto mode
  now picks more accurate — typically smaller and more efficient —
  `ham_dist` / `num_blocks` / `num_chunks` when `cos_dist` is roughly
  `>= 0.1`; the `missing_ratio` recall guarantee holds either way. Smaller
  `cos_dist` values, including the `golden/` fixture (`cos_dist=0.01`),
  are unaffected — the search already converged at `ham_dist <= 2`, where
  the old and new `combination()` agree exactly.

Internal (no behavior change — output is byte-identical):

- The C++ sources were renamed to snake_case (`src/sketch_sort.hpp`,
  `src/sketch_sort.cpp`, `src/main.cpp`), all C++ code now lives in
  `namespace sketchsort`, and internal identifiers were unified to one
  naming convention (PascalCase types, snake_case functions/variables).
  CLI flags, the Python API, the output format, and the legacy
  `cd src && make` build are all unchanged.

## 0.1.0 release notes (based on upstream 0.0.8)

Breaking changes:

- **Deterministic by default.** v0.0.8 seeded the projection RNG with
  `time(0)` on every run, so results were non-reproducible. This release
  exposes a `seed` parameter (default `0`) and removes the time-based
  seeding on every code path — C++ CLI, Python `search()`, Python
  `run_from_file()`, and the `sketchsort` console script. To reproduce
  the old non-deterministic behaviour pass an explicit time-based seed,
  e.g. `sketchsort -seed $(date +%s) ...`.

- `exit()` calls in the C++ core were replaced with `std::runtime_error`,
  surfaced to Python as `RuntimeError`. Callers that previously relied on
  the process exiting on bad input must now catch the exception.

New:

- `sketchsort.search(X, ...)` Python API returning a NumPy structured array.
- `sketchsort.run_from_file(...)` Python API for file-based pipelines.
- `sketchsort` console script (entry point from pip install).
- `-seed <int>` CLI flag (default `0`).
- `-quiet` CLI flag (Python entry point only) — suppresses algorithm
  progress output. Python `search()` / `run_from_file()` default to
  quiet.

Internal:

- The C++ standard requirement moved from C++98 (`-ansi`) to C++17.

## Build from source

Requires CMake ≥ 3.20 and a C++17 compiler.

```
pip install .
```

`pip install -e ".[test]"` for an editable install with the test deps,
then `pytest tests/`.

A legacy Makefile is preserved under `src/` for users who only want the
C++ CLI without a Python toolchain:

```
cd src && make
./sketchsort -cosdist 0.01 -seed 42 ../dat/sample.txt out.txt
```

The Makefile build uses `-ffast-math`, which lets the compiler reorder
floating-point operations. For most inputs the reported pair set is
unchanged, but `cos_dist` text values can differ in the 5th–6th
significant digit compared to the CMake/Python build, and pairs whose
true distance is right at the threshold may also differ between builds.
If you need output that matches the wheel/Python build exactly, build
the CLI via CMake instead:

```
cmake -B build -DSKETCHSORT_BUILD_CLI=ON -DSKETCHSORT_BUILD_PYTHON=OFF \
               -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sketchsort_cli
./build/sketchsort -cosdist 0.01 -seed 42 dat/sample.txt out.txt
```

## Citation

If you use SketchSort in published work, please cite the original papers:

**Methodology** (the multiple-sorting technique):

Tabei, Y., Uno, T., Sugiyama, M., and Tsuda, K. (2010).
*Single versus Multiple Sorting in All Pairs Similarity Search*.
In Proceedings of the 2nd Asian Conference on Machine Learning (ACML 2010),
JMLR Workshop and Conference Proceedings, **13**: 145–160.
[PDF](http://proceedings.mlr.press/v13/tabei10a/tabei10a.pdf)

```bibtex
@inproceedings{tabei2010sketchsort,
  title     = {Single versus Multiple Sorting in All Pairs Similarity Search},
  author    = {Tabei, Yasuo and Uno, Takeaki and Sugiyama, Masashi and Tsuda, Koji},
  booktitle = {Proceedings of the 2nd Asian Conference on Machine Learning (ACML)},
  series    = {JMLR Workshop and Conference Proceedings},
  volume    = {13},
  pages     = {145--160},
  year      = {2010},
  address   = {Tokyo, Japan},
}
```

**Application to molecular fingerprints**:

Tabei, Y. and Tsuda, K. (2011). *SketchSort: Fast All Pairs Similarity Search
for Large Databases of Molecular Fingerprints*. Molecular Informatics
**30**(9): 801–807. doi:[10.1002/minf.201100050](https://doi.org/10.1002/minf.201100050)

```bibtex
@article{tabei2011sketchsort,
  title   = {SketchSort: Fast All Pairs Similarity Search for Large Databases of Molecular Fingerprints},
  author  = {Tabei, Yasuo and Tsuda, Koji},
  journal = {Molecular Informatics},
  volume  = {30},
  number  = {9},
  pages   = {801--807},
  year    = {2011},
  doi     = {10.1002/minf.201100050},
}
```

## License

The cosine core (`src/sketch_sort.*`, `src/main.cpp`) and the Python
packaging are MIT-licensed (see `LICENSE`). The min-max and Jaccard cores
(`src/sketch_sort_minmax.*`, `src/sketch_sort_jaccard.*`) are BSD-3-Clause,
as noted in their per-file SPDX headers. The bundled Boost headers under
`src/boost/` are distributed under the Boost Software License 1.0 (see
`NOTICE`).

The min-max method additionally builds on:

Li, P. (2017). *Linearized GMM Kernels and Normalized Random Fourier
Features*. In Proceedings of the 23rd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining (KDD), 315–324.
