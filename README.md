# sketchsort

SketchSort is a batch **all-pairs similarity search** library: given a
dataset, it enumerates every pair of records whose distance is at most a
given threshold. Typical uses include near-duplicate detection, molecular
fingerprint search, and nearest-neighbor-style filtering over embeddings.

It is an **approximate-recall, exact-verification** method: every reported
pair's distance is computed exactly (no false positives), while some true
pairs may be missed depending on the sketch parameters. The `missing_ratio`
parameter bounds the expected fraction of missed pairs — smaller values
search more exhaustively at the cost of more time and memory.

Three distance functions are supported — **cosine**, **min-max**
(generalized Jaccard/Tanimoto on real vectors), and **Jaccard/Tanimoto**
(sparse integer-id sets) — via `search`, `search_minmax`, and
`search_jaccard` respectively. See [Supported metrics](#supported-metrics).

## Install

```
pip install sketchsort
```

Wheels are provided for CPython 3.9–3.13 on Linux (x86_64), macOS arm64
(Apple Silicon), and Windows (x86_64). Intel macOS is not yet provided as
a wheel; on that platform `pip install sketchsort` will fall back to a
source build (requires a C++17 compiler and CMake).

## Quick start

```python
import numpy as np
import sketchsort

X = np.loadtxt("vectors.txt", dtype=np.float32)  # shape (N, D), row i has id i

# Cosine distance
pairs = sketchsort.search(X, cos_dist=0.01, missing_ratio=1e-4, seed=42)

# Min-max distance (generalized Jaccard on real vectors)
pairs = sketchsort.search_minmax(X, minmax_dist=0.2, missing_ratio=1e-4, seed=42)

# Jaccard / Tanimoto distance (sparse integer-id sets)
sets = [[0, 3, 7], [0, 3, 7, 9], [1, 2], [7, 9]]
pairs = sketchsort.search_jaccard(sets, jaccard_dist=0.5, missing_ratio=1e-4, seed=42)

for id1, id2, dist in pairs:
    print(id1, id2, dist)
```

Each function returns a NumPy structured array
`[('id1', '<u4'), ('id2', '<u4'), ('<metric>_dist', '<f4')]`. `id1`/`id2`
are row indices into the input; each pair appears at most once. See
[Python API](#python-api) for the full parameter reference and
[Command line](#command-line) for the `sketchsort` CLI.

## Supported metrics

| Metric | Input | Function | Distance |
| --- | --- | --- | --- |
| **Cosine** | dense float vectors | `search` | `1 - ⟨x,y⟩/(‖x‖·‖y‖)` |
| **Min-max** (generalized Jaccard/Tanimoto on real vectors) | dense float vectors | `search_minmax` | `1 - Σ min(xᵢ,yᵢ)/Σ max(xᵢ,yᵢ)` |
| **Jaccard / Tanimoto** | sparse integer-id sets | `search_jaccard` | <code>1 - &#124;A∩B&#124;/&#124;A∪B&#124;</code> |

Each metric sketches its input into a short byte string — sign of a random
projection (cosine), generalized consistent weighted sampling (min-max), or
MinHash (Jaccard) — then enumerates near-duplicate sketches using the
shared multiple-sorting technique of Tabei et al. (2010); see
[Citation](#citation).

## Python API

```python
import numpy as np
import sketchsort

X = np.loadtxt("dat/sample.txt", dtype=np.float32)   # shape (N, D)

pairs = sketchsort.search(X, cos_dist=0.01, missing_ratio=0.0001, seed=42)

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

`search_minmax` and `search_jaccard` return the analogous
`minmax_dist`/`jaccard_dist` dtype — see
[Supported metrics](#supported-metrics).

### Other metrics

`search_minmax(X, ...)` takes the same dense `(N, D)` float matrix as
`search`, and reports pairs whose min-max distance is at most
`minmax_dist`. Sketches come from generalized consistent weighted sampling
(Li, 2017). Negative values in `X` are accepted; make sure min-max distance
is the appropriate measure for your data, and see `z_normalization` /
`minmax_normalization` under [Advanced options](#advanced-options) if your
columns are on different scales.

```python
X = np.loadtxt("dat/sample_minmax.txt", dtype=np.float32)
pairs = sketchsort.search_minmax(X, minmax_dist=0.2, missing_ratio=1e-4, seed=42)
# dtype: [('id1','<u4'), ('id2','<u4'), ('minmax_dist','<f4')]
```

`search_jaccard(sets, ...)` takes a sequence of integer-id sets — `sets[i]`
is the collection of non-negative ids present in row `i` (e.g. the "on"
bits of a molecular fingerprint) — and reports pairs whose Jaccard
distance is at most `jaccard_dist`. Sketches are MinHash values. Duplicate
ids within a set are removed; empty sets are kept so row `i` always has
id `i`.

```python
sets = [[0, 3, 7], [0, 3, 7, 9], [1, 2], [7, 9]]
pairs = sketchsort.search_jaccard(sets, jaccard_dist=0.5, missing_ratio=1e-4, seed=42)
# dtype: [('id1','<u4'), ('id2','<u4'), ('jaccard_dist','<f4')]
```

The Jaccard pipeline is fully deterministic at the integer/byte level
(MinHash with a hand-rolled mt19937 Fisher-Yates shuffle plus exact
intersection/union), so for a fixed `seed` its output is identical across
platforms. The cosine and min-max cores are deterministic per platform, but
their final distance/threshold comparison can flip a few boundary pairs
across platforms due to libm differences.

### File I/O

If you have a file in the legacy text format and want output compatible
with the CMake-built C++ CLI from the same source tree, use the matching
`run_from_file*` function instead of `search*`:

```python
sketchsort.run_from_file("input.txt", "output.txt", cos_dist=0.01, seed=42)
sketchsort.run_from_file_minmax("input.txt", "output.txt", minmax_dist=0.2, seed=42)
sketchsort.run_from_file_jaccard("sets.txt", "output.txt", jaccard_dist=0.05, seed=42)
```

`run_from_file`/`run_from_file_minmax` read whitespace-separated float
vectors, one per line, no ID column; `run_from_file_jaccard` reads one
whitespace-separated integer-id set per line. All three write
`id1 id2 <metric>_dist` triples to the output file.

Every line of a float-vector input file must carry exactly as many numeric
tokens as the first line (that count becomes the file's dimension). Blank
or whitespace-only lines, non-numeric tokens, lines with too many values,
and a completely empty file are all rejected with an error naming the
offending line rather than being silently skipped or truncated.

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

## Advanced options

### `missing_ratio` and auto mode

By default (`ham_dist`/`num_blocks`/`num_chunks` left at `None`),
SketchSort picks its enumeration parameters automatically so that the
expected fraction of missed true-neighbor pairs is at most `missing_ratio`.
Smaller `missing_ratio` values search more exhaustively, at the cost of
more time and memory. The bound is derived from the random-projection
model and applies to the expectation, not to every individual run.

### Manual parameter control

For full control of the sketch enumeration, pass all of `ham_dist`,
`num_blocks`, and `num_chunks` to any `search*`/`run_from_file*` function.
Providing any of these switches the call into manual mode, where
`missing_ratio` is ignored. If you specify only some of the three, the
rest fall back to defaults (`ham_dist=1`, `num_blocks=4`, `num_chunks=3`),
so it is safer to set them together:

```python
pairs = sketchsort.search(X, cos_dist=0.01, ham_dist=1, num_blocks=4, num_chunks=3, seed=42)
```

### Reproducibility (`seed`)

`seed` (default `0` for cosine and Jaccard) seeds the random-number
generator used to draw the projection vectors or hash permutations. For a
fixed build, two calls with the same `seed`, parameters, and input produce
the same output. To reproduce the non-deterministic behaviour of upstream
0.0.8, pass a time-based seed explicitly, e.g. `seed=int(time.time())`.
Min-max also accepts `seed=-1` to derive the seed from the clock.

### Cosine: `centering`

`centering` (default `False`) — when `True`, the coordinate-wise mean of
`X` is subtracted from every row before both sketching and distance
computation. The reported `cos_dist` is then the cosine distance between
the *mean-shifted* vectors, not the raw input. Recommended when input
vectors are non-negative and share a strong bias (e.g. raw bag-of-words
counts, histograms, molecular fingerprints — the original SketchSort use
case).

### Min-max: `z_normalization` / `minmax_normalization`

Two optional, mutually-exclusive normalizers rescale each column of `X`
before sketching: `z_normalization` (zero mean, unit variance) and
`minmax_normalization` (rescale to `[0, 1]`).

### `verbose`

`verbose` (default `False`) — when `True`, the underlying C++ core prints
algorithm progress to stdout/stderr. Default is quiet; turn on for
diagnostics.

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

## Release notes

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

**0.3.0** adds the min-max and Jaccard/Tanimoto metrics alongside cosine,
unifies all three onto one shared multiple-sort enumeration engine
(verified byte-identical output), and relicenses the whole source tree to
MIT.

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

**Min-max sketching** (generalized consistent weighted sampling, used by
`search_minmax`):

Li, P. (2017). *Linearized GMM Kernels and Normalized Random Fourier
Features*. In Proceedings of the 23rd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining (KDD), 315–324.
doi:[10.1145/3097983.3098081](https://doi.org/10.1145/3097983.3098081)

```bibtex
@inproceedings{li2017linearized,
  title     = {Linearized GMM Kernels and Normalized Random Fourier Features},
  author    = {Li, Ping},
  booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)},
  pages     = {315--324},
  year      = {2017},
  doi       = {10.1145/3097983.3098081},
}
```

## License

MIT for the entire SketchSort source — all three metric cores, the shared
enumeration engine, the CLI, and the Python packaging (see `LICENSE`).
The bundled Boost headers under `src/boost/` are distributed under the
Boost Software License 1.0 (see `NOTICE`).
