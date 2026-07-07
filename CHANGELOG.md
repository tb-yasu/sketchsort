# Changelog

All notable changes to this project are documented here.

## [0.3.0] - 2026-07-07

### Added

- Two new similarity metrics alongside cosine, sharing the same
  multiple-sort enumeration engine but living in separate C++
  sub-namespaces (`sketchsort::minmax`, `sketchsort::jaccard`):
  - `search_minmax` / `run_from_file_minmax` — min-max (generalized
    Jaccard/Tanimoto) on dense real vectors via generalized consistent
    weighted sampling, with optional `z_normalization` /
    `minmax_normalization`.
  - `search_jaccard` / `run_from_file_jaccard` — Jaccard/Tanimoto on
    sparse integer-id sets via MinHash.
- Both new cores share the in-memory + file APIs, exception (not
  `exit()`) error handling, and deterministic-by-`seed` behaviour of the
  cosine core. The Jaccard MinHash uses a hand-rolled mt19937
  Fisher-Yates shuffle so it is reproducible across platforms.

### Changed

- License unified to MIT across the whole source tree (the min-max core
  was BSD-3-Clause upstream). Vendored Boost remains BSL-1.0.

No change to the existing cosine `search` / `run_from_file` API or output.

### Internal

No behavior change — output is byte-identical.

- The multiple-sort enumeration machinery (radix/insertion sorting,
  block grouping, canonicality checks, candidate-pair reporting,
  missing-edge-ratio math, parameter auto-selection and validation)
  previously existed as three near-identical copies, one per metric. It
  now lives once in `src/multi_sort_engine.hpp`
  (`sketchsort::engine::MultiSortEngine`), with each metric supplying
  only its projection, exact-distance verification, and an optional
  O(1) prefilter through an inlined template hook. Verified
  byte-identical output across all three metrics on nine seed/parameter
  configurations, including the `-ffast-math` legacy Makefile build.

## [0.2.0] - 2026-07-04

### Changed

Behavior changes from hardening the C++ core and CLI against invalid
input:

- **Invalid input now raises instead of silently producing wrong
  output.** Zero-norm or non-finite (NaN/Inf) rows, out-of-range
  parameters (`cos_dist` outside `[0, 2]`, `missing_ratio` outside
  `(0, 1)` in auto mode, `ham_dist >= num_blocks`, `num_blocks` outside
  `[1, 32]`, `num_chunks < 1`), and malformed input-file lines
  (blank/whitespace-only lines, non-numeric tokens, lines with more
  values than the file's dimension, or a completely empty input file)
  now raise `ValueError` / `RuntimeError` from Python
  (`std::invalid_argument` / `std::runtime_error` from the C++ core)
  instead of being silently dropped or corrupting output.
- **C++ CLI exit codes.** A missing `OUTFILE`, a value-taking flag with
  no value, a non-numeric flag value, or extra positional arguments
  after `OUTFILE` now print an error to stderr and exit with status 1
  instead of crashing or silently continuing with garbage arguments.
  `-version` and the new `-h`/`--help` exit 0 immediately instead of
  falling through to a (previously crash-prone) attempt to run with
  missing file arguments. Options must precede `INFILE`/`OUTFILE`;
  anything after them is now a positional-argument error rather than
  being silently ignored.
- **C++ CLI: negative parameter values rejected at parse time.**
  `-hamdist` / `-numblocks` / `-numchunks` are now parsed as unsigned
  integers (matching `-seed`), so a negative value such as
  `-hamdist -1` exits with a clear `invalid unsigned integer value`
  error instead of silently wrapping to a huge value and failing later
  with a confusing out-of-range message.
- **`combination()` integer-division fix.** The internal binomial
  coefficient helper used by auto mode's parameter search performed
  integer division and underestimated `C(n, m)` for `m >= 3`. Auto
  mode now picks more accurate — typically smaller and more efficient
  — `ham_dist` / `num_blocks` / `num_chunks` when `cos_dist` is roughly
  `>= 0.1`; the `missing_ratio` recall guarantee holds either way.
  Smaller `cos_dist` values, including the `golden/` fixture
  (`cos_dist=0.01`), are unaffected — the search already converged at
  `ham_dist <= 2`, where the old and new `combination()` agree exactly.

### Internal

No behavior change — output is byte-identical.

- The C++ sources were renamed to snake_case (`src/sketch_sort.hpp`,
  `src/sketch_sort.cpp`, `src/main.cpp`), all C++ code now lives in
  `namespace sketchsort`, and internal identifiers were unified to one
  naming convention (PascalCase types, snake_case functions/variables).
  CLI flags, the Python API, the output format, and the legacy
  `cd src && make` build are all unchanged.

## [0.1.1] - 2026-06-30

### Added

- Windows wheel support.

## [0.1.0] - 2026-06-30

Based on upstream SketchSort 0.0.8 (Yasuo Tabei, 2011).

### Breaking changes

- **Deterministic by default.** v0.0.8 seeded the projection RNG with
  `time(0)` on every run, so results were non-reproducible. This
  release exposes a `seed` parameter (default `0`) and removes the
  time-based seeding on every code path — C++ CLI, Python `search()`,
  Python `run_from_file()`, and the `sketchsort` console script. To
  reproduce the old non-deterministic behaviour pass an explicit
  time-based seed, e.g. `sketchsort -seed $(date +%s) ...`.
- `exit()` calls in the C++ core were replaced with
  `std::runtime_error`, surfaced to Python as `RuntimeError`. Callers
  that previously relied on the process exiting on bad input must now
  catch the exception.

### Added

- `sketchsort.search(X, ...)` Python API returning a NumPy structured
  array.
- `sketchsort.run_from_file(...)` Python API for file-based pipelines.
- `sketchsort` console script (entry point from pip install).
- `-seed <int>` CLI flag (default `0`).
- `-quiet` CLI flag (Python entry point only) — suppresses algorithm
  progress output. Python `search()` / `run_from_file()` default to
  quiet.

### Internal

- The C++ standard requirement moved from C++98 (`-ansi`) to C++17.
