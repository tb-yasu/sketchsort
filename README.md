# sketchsort

Fast all-pairs cosine-similarity search via random projection sketches.

`sketchsort` finds pairs of high-dimensional float vectors whose cosine
distance is below a given threshold. Here, the cosine distance is defined
as `1 - cosine_similarity`, where
`cosine_similarity = ⟨x, y⟩ / (‖x‖ · ‖y‖)`. Input vectors do not need to
be pre-normalized; norms are computed internally. 

The algorithm sketches vectors into binary sequences by random
projection, then enumerates near-duplicate sketches using the
multiple-sorting technique of Tabei et al. (2010). The missing-edge-ratio
parameter (`missing_ratio`) controls how exhaustive this enumeration is.
See the Python API section below for details.

## Install

```
pip install sketchsort
```

Wheels are provided for CPython 3.9–3.13 on Linux (x86_64) and macOS arm64
(Apple Silicon). Intel macOS and Windows are not yet provided as wheels;
on those platforms `pip install sketchsort` will fall back to a source
build (requires a C++17 compiler and CMake).

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

## Command line

Installing the package also installs a `sketchsort` console script. It
uses the same defaults as the Python API: automatic parameter selection
unless you pass any of `-hamdist` / `-numblocks` / `-numchunks`.

```
# Typical: cos_dist + missing_ratio
sketchsort -cosdist 0.01 -missingratio 0.0001 -seed 42 input.txt output.txt

# Manual parameter control
sketchsort -cosdist 0.01 -hamdist 1 -numblocks 4 -numchunks 3 -seed 42 input.txt output.txt
```

Flags: `-cosdist`, `-missingratio`, `-hamdist`, `-numblocks`, `-numchunks`,
`-auto`, `-centering`, `-seed`, `-quiet`. `-auto` forces automatic
parameter selection even if any of `-hamdist` / `-numblocks` /
`-numchunks` is also given. `-centering` subtracts the coordinate-wise
mean from the input vectors before both sketching and distance
computation (reported `cos_dist` is then between mean-shifted vectors).

## Memory note

`sketchsort.search(...)` collects every reported pair into memory before
returning. For large inputs at loose thresholds the result can be very
large (tens of millions of pairs are realistic). If memory is a concern,
use `sketchsort.run_from_file(...)` instead — it streams pairs to disk
while the algorithm runs.

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

MIT for the SketchSort source (see `LICENSE`). The bundled Boost headers
under `src/boost/` are distributed under the Boost Software License 1.0
(see `NOTICE`).
