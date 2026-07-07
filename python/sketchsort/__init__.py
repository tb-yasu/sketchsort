# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""Public Python API for sketchsort.

Two top-level functions:

- `search(X, ...)`           — NumPy in / NumPy structured array out
- `run_from_file(in, out, ...)` — text file in / text file out (byte-exact
                                   compatible with the C++ CLI when called
                                   with matching parameters)

Both delegate to the C++ extension `_core`. The wrappers here add **auto
parameter selection**: when none of `ham_dist` / `num_blocks` / `num_chunks`
is given, the library uses the `missing_ratio` knob to pick those internally
(`auto_mode` in the C++ layer). When any of them IS given, the explicit
values are used and `missing_ratio` is ignored.
"""

from typing import Optional

from ._core import search        as _search_core
from ._core import run_from_file as _run_from_file_core
from ._core import search_minmax        as _search_minmax_core
from ._core import run_from_file_minmax as _run_from_file_minmax_core

__all__ = [
    "search",
    "run_from_file",
    "search_minmax",
    "run_from_file_minmax",
]
__version__ = "0.3.0"


# Internal defaults for the explicit-mode parameters. Only used to fill in
# whichever of ham_dist / num_blocks / num_chunks the caller left at None
# while specifying at least one of the others. These are NOT the defaults
# the user sees; from the user's perspective the defaults are "auto".
_MANUAL_HAM_DIST   = 1
_MANUAL_NUM_BLOCKS = 4
_MANUAL_NUM_CHUNKS = 3


def _resolve_auto(ham_dist, num_blocks, num_chunks):
    """Return True if all three manual knobs are unset (= use auto_mode)."""
    return ham_dist is None and num_blocks is None and num_chunks is None


def _validate_seed(seed):
    if not (0 <= seed < 2**32):
        raise ValueError(f"seed must satisfy 0 <= seed < 2**32, got {seed}")


def _validate_seed_minmax(seed):
    # The min-max core treats a negative seed as "derive from the clock", so
    # -1 is a legal sentinel here (unlike the cosine seed, which must be >= 0).
    if not (-1 <= seed < 2**31):
        raise ValueError(
            f"seed must satisfy -1 <= seed < 2**31 (-1 = derive from clock), got {seed}"
        )


def search(
    X,
    *,
    cos_dist: float = 0.01,
    ham_dist: Optional[int] = None,
    num_blocks: Optional[int] = None,
    num_chunks: Optional[int] = None,
    missing_ratio: float = 0.0001,
    centering: bool = False,
    seed: int = 0,
    verbose: bool = False,
):
    """Find all vector pairs in `X` whose cosine distance is at most `cos_dist`.

    Auto vs manual:

    - **Auto (default)**: leave `ham_dist` / `num_blocks` / `num_chunks` at
      their default `None`. The library picks them so that at most
      `missing_ratio` fraction of true neighbours can be missed.
    - **Manual**: pass any of `ham_dist` / `num_blocks` / `num_chunks`. The
      explicit values are used and `missing_ratio` is ignored.

    Parameters
    ----------
    X : ndarray, shape (N, D), float32 (other dtypes are cast)
        Row i has implicit id i.
    cos_dist : float
        Maximum cosine distance for a pair to be reported.
    ham_dist, num_blocks, num_chunks : int, optional
        Power-user override of the multiple-sort enumeration parameters.
        Setting any disables auto mode for this call.
    missing_ratio : float
        Target probability of missing a true neighbour (auto mode only).
        Smaller is more thorough but slower. Default 0.0001 (0.01%).
    centering : bool
        Subtract per-dimension mean from X before sketching.
    seed : int
        Random seed for the projection RNG. Default 0 = deterministic.
    verbose : bool
        Print algorithm progress to stdout. Default False (quiet).

    Returns
    -------
    ndarray with dtype [('id1', '<u4'), ('id2', '<u4'), ('cos_dist', '<f4')]
    """
    _validate_seed(seed)
    auto = _resolve_auto(ham_dist, num_blocks, num_chunks)
    return _search_core(
        X,
        cos_dist      = cos_dist,
        ham_dist      = _MANUAL_HAM_DIST   if ham_dist   is None else ham_dist,
        num_blocks    = _MANUAL_NUM_BLOCKS if num_blocks is None else num_blocks,
        num_chunks    = _MANUAL_NUM_CHUNKS if num_chunks is None else num_chunks,
        auto_mode     = auto,
        missing_ratio = missing_ratio,
        centering     = centering,
        seed          = seed,
        verbose       = verbose,
    )


def run_from_file(
    input_path: str,
    output_path: str,
    *,
    cos_dist: float = 0.01,
    ham_dist: Optional[int] = None,
    num_blocks: Optional[int] = None,
    num_chunks: Optional[int] = None,
    missing_ratio: float = 0.0001,
    centering: bool = False,
    seed: int = 0,
    verbose: bool = False,
):
    """Read whitespace-separated float vectors from `input_path`, write
    'id1 id2 cos_dist' triples to `output_path`. Auto / manual selection
    follows the same rule as `search()`.
    """
    _validate_seed(seed)
    auto = _resolve_auto(ham_dist, num_blocks, num_chunks)
    _run_from_file_core(
        input_path    = input_path,
        output_path   = output_path,
        cos_dist      = cos_dist,
        ham_dist      = _MANUAL_HAM_DIST   if ham_dist   is None else ham_dist,
        num_blocks    = _MANUAL_NUM_BLOCKS if num_blocks is None else num_blocks,
        num_chunks    = _MANUAL_NUM_CHUNKS if num_chunks is None else num_chunks,
        auto_mode     = auto,
        missing_ratio = missing_ratio,
        centering     = centering,
        seed          = seed,
        verbose       = verbose,
    )


def search_minmax(
    X,
    *,
    minmax_dist: float = 0.1,
    ham_dist: Optional[int] = None,
    num_blocks: Optional[int] = None,
    num_chunks: Optional[int] = None,
    missing_ratio: float = 0.0001,
    z_normalization: bool = False,
    minmax_normalization: bool = False,
    seed: int = 0,
    verbose: bool = False,
):
    """Find all vector pairs in `X` whose min-max distance is at most `minmax_dist`.

    The min-max (a.k.a. generalized Jaccard / Tanimoto) similarity of two real
    vectors is ``sum(min(a_i, b_i)) / sum(max(a_i, b_i))`` over the coordinates,
    and the reported distance is ``1 - similarity``. Sketches come from
    generalized consistent weighted sampling (GCWS).

    Auto vs manual selection follows the same rule as `search()`: leave
    `ham_dist` / `num_blocks` / `num_chunks` at `None` for auto mode.

    Parameters
    ----------
    X : ndarray, shape (N, D), float32 (other dtypes are cast)
        Row i has implicit id i. Negative values are allowed.
    minmax_dist : float
        Maximum min-max distance for a pair to be reported. Default 0.1.
    ham_dist, num_blocks, num_chunks : int, optional
        Power-user override of the multiple-sort enumeration parameters.
    missing_ratio : float
        Target probability of missing a true neighbour (auto mode only).
    z_normalization : bool
        Z-normalize each dimension before sketching.
    minmax_normalization : bool
        Min-max normalize each dimension to [0, 1] before sketching.
    seed : int
        Random seed for the projection RNG. Default 0 = deterministic.
        A negative seed derives the seed from the clock.
    verbose : bool
        Print algorithm progress to stdout. Default False (quiet).

    Returns
    -------
    ndarray with dtype [('id1', '<u4'), ('id2', '<u4'), ('minmax_dist', '<f4')]
    """
    _validate_seed_minmax(seed)
    auto = _resolve_auto(ham_dist, num_blocks, num_chunks)
    return _search_minmax_core(
        X,
        minmax_dist          = minmax_dist,
        ham_dist             = _MANUAL_HAM_DIST   if ham_dist   is None else ham_dist,
        num_blocks           = _MANUAL_NUM_BLOCKS if num_blocks is None else num_blocks,
        num_chunks           = _MANUAL_NUM_CHUNKS if num_chunks is None else num_chunks,
        auto_mode            = auto,
        missing_ratio        = missing_ratio,
        z_normalization      = z_normalization,
        minmax_normalization = minmax_normalization,
        seed                 = seed,
        verbose              = verbose,
    )


def run_from_file_minmax(
    input_path: str,
    output_path: str,
    *,
    minmax_dist: float = 0.1,
    ham_dist: Optional[int] = None,
    num_blocks: Optional[int] = None,
    num_chunks: Optional[int] = None,
    missing_ratio: float = 0.0001,
    z_normalization: bool = False,
    minmax_normalization: bool = False,
    seed: int = 0,
    verbose: bool = False,
):
    """Read whitespace-separated float vectors from `input_path`, write
    'id1 id2 minmax_dist' triples to `output_path`. Auto / manual selection
    follows the same rule as `search_minmax()`.
    """
    _validate_seed_minmax(seed)
    auto = _resolve_auto(ham_dist, num_blocks, num_chunks)
    _run_from_file_minmax_core(
        input_path           = input_path,
        output_path          = output_path,
        minmax_dist          = minmax_dist,
        ham_dist             = _MANUAL_HAM_DIST   if ham_dist   is None else ham_dist,
        num_blocks           = _MANUAL_NUM_BLOCKS if num_blocks is None else num_blocks,
        num_chunks           = _MANUAL_NUM_CHUNKS if num_chunks is None else num_chunks,
        auto_mode            = auto,
        missing_ratio        = missing_ratio,
        z_normalization      = z_normalization,
        minmax_normalization = minmax_normalization,
        seed                 = seed,
        verbose              = verbose,
    )
