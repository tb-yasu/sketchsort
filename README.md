# sketchsort

Fast all-pairs cosine-similarity search via random projection sketches.

For a set of `N` dense float vectors, `sketchsort` finds every pair whose
cosine distance is below a given threshold. It does so by sketching the
vectors into binary sequences via random projection and enumerating
near-duplicate pairs with a "multiple sort" (radix-sort + block-equality)
scheme. 

## Install

```
pip install sketchsort
```

Wheels are provided for CPython 3.9–3.13 on Linux (x86_64) and macOS
(x86_64 / arm64). Windows wheels are not yet shipped.

## Python API

```python
import numpy as np
import sketchsort

X = np.loadtxt("dat/sample.txt", dtype=np.float32)   # shape (N, D)
pairs = sketchsort.search(
    X,
    cos_dist=0.01,
    num_blocks=4,
    ham_dist=1,
    num_chunks=3,
    seed=42,
)

# pairs.dtype == [('id1', '<u4'), ('id2', '<u4'), ('cos_dist', '<f4')]
for id1, id2, d in pairs:
    print(id1, id2, d)
```

`sketchsort.search(...)` returns a NumPy structured array. `id1` / `id2` are
row indices in the input matrix `X`.

If you have a file in the legacy text format and want byte-identical output
to the C++ CLI, call:

```python
sketchsort.run_from_file("input.txt", "output.txt", cos_dist=0.01, seed=42)
```

## Command line

The pip install ships a `sketchsort` console script that mirrors the legacy
C++ CLI:

```
sketchsort -cosdist 0.01 -seed 42 input.txt output.txt
```

Supported flags: `-cosdist`, `-hamdist`, `-numblocks`, `-numchunks`, `-auto`,
`-missingratio`, `-centering`, `-seed`. The input file is whitespace-separated
float vectors, one per line, no ID column. The output file is `id1 id2
cos_dist` triples.

## 0.1.0 release notes (based on upstream 0.0.8)

Breaking changes:

- **Deterministic by default.** v0.0.8 seeded the projection RNG with
  `time(0)` on every run, so results were non-reproducible. This release
  exposes a `seed` parameter (default `0`) and removes the time-based seeding
  on every code path — C++ CLI, Python `search()`, Python `run_from_file()`,
  and the `sketchsort` console script. To reproduce the old non-deterministic
  behaviour pass an explicit time-based seed, e.g.
  `sketchsort -seed $(date +%s) ...`.

- `exit()` calls in the C++ core were replaced with `std::runtime_error`,
  surfaced to Python as `RuntimeError`. Callers that previously relied on
  the process exiting on bad input must now catch the exception.

New:

- `sketchsort.search(X, ...)` Python API returning a NumPy structured array.
- `sketchsort.run_from_file(...)` Python API for byte-exact CLI parity.
- `sketchsort` console script (entry point from pip install).
- `-seed <int>` CLI flag (default `0`).
- `-quiet` CLI flag (Python entry point only) — suppresses algorithm
  progress chatter. Python `search()` / `run_from_file()` default to quiet.

Internal:

- The C++ standard requirement moved from C++98 (`-ansi`) to C++17.

## Memory note

`sketchsort.search(...)` collects every reported pair into memory before
returning. For large inputs at loose thresholds the result can be very large
(tens of millions of pairs are realistic). If memory is a concern, use
`sketchsort.run_from_file(...)` instead — it streams pairs to disk while the
algorithm runs.

## Build from source

Requires CMake ≥ 3.20 and a C++17 compiler.

```
pip install .
```

`pip install -e ".[test]"` for an editable install with the test deps, then
`pytest tests/`.

A legacy Makefile is preserved under `src/` for users who only want the C++
CLI without a Python toolchain:

```
cd src && make
./sketchsort -cosdist 0.01 -seed 42 ../dat/sample.txt out.txt
```

The Makefile build uses `-ffast-math`, which lets the compiler reorder
floating-point operations. The pair set reported is unchanged, but the
`cos_dist` text values can differ in the 5th–6th significant digit compared
to the CMake/Pythonbuild. If you need byte-identical output to the Python
package, build the CLI via CMake instead:

```
cmake -B build -DSKETCHSORT_BUILD_CLI=ON -DSKETCHSORT_BUILD_PYTHON=OFF \
               -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sketchsort_cli
./build/sketchsort -cosdist 0.01 -seed 42 dat/sample.txt out.txt
```

## Citation

If you use SketchSort in published work, please cite the original papers:

**Methodology** (the multiple-sort enumeration scheme):

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
under `src/boost/` are distributed under the Boost Software License 1.0 (see
`NOTICE`).
