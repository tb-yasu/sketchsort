# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SketchSort contributors
"""Command-line entry point.

Argument names mirror the legacy C++ sketchsort CLIs for back-compat. Pick
the similarity metric with ``-metric {cosine,minmax,jaccard}`` (default
cosine). If none of ``-hamdist`` / ``-numblocks`` / ``-numchunks`` is given,
the run defaults to **auto mode** (the library picks them from
``-missingratio``); pass any of those three to switch to manual mode. The
``-auto`` flag forces auto mode even when manual params are given.

Input format depends on the metric:
  - cosine, minmax : one whitespace-separated float vector per line
  - jaccard        : one whitespace-separated integer-id set per line
"""

import argparse
import sys

from . import run_from_file, run_from_file_minmax, run_from_file_jaccard


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sketchsort",
        description="Fast all-pairs similarity search via random projection sketches.",
        add_help=True,
    )
    p.add_argument("-metric", choices=["cosine", "minmax", "jaccard"], default="cosine",
                   help="similarity metric (default: cosine)")
    # Per-metric distance thresholds (only the selected metric's is used).
    p.add_argument("-cosdist", type=float, default=0.01,
                   help="cosine: maximum cosine distance (default: 0.01)")
    p.add_argument("-minmax", dest="minmaxdist", type=float, default=0.1,
                   help="minmax: maximum min-max distance (default: 0.1)")
    p.add_argument("-jaccard", dest="jaccarddist", type=float, default=0.05,
                   help="jaccard: maximum Jaccard distance (default: 0.05)")
    # Shared enumeration knobs.
    p.add_argument("-missingratio", type=float, default=0.0001,
                   help="target missing-edge ratio used in auto mode (default: 0.0001)")
    p.add_argument("-hamdist", type=int, default=None,
                   help="manual override: Hamming distance per chunk")
    p.add_argument("-numblocks", type=int, default=None,
                   help="manual override: number of blocks per chunk")
    p.add_argument("-numchunks", type=int, default=None,
                   help="manual override: number of independent sketch chunks")
    p.add_argument("-auto", dest="force_auto", action="store_true",
                   help="force auto mode even if -hamdist/-numblocks/-numchunks are given")
    # cosine-only.
    p.add_argument("-centering", dest="centering", action="store_true",
                   help="cosine: subtract per-dimension mean before sketching")
    # minmax-only.
    p.add_argument("-znormalization", dest="znorm", action="store_true",
                   help="minmax: z-normalize each dimension before sketching")
    p.add_argument("-minmaxnormalization", dest="mmnorm", action="store_true",
                   help="minmax: min-max normalize each dimension before sketching")
    p.add_argument("-seed", type=int, default=0,
                   help="RNG seed (default: 0)")
    p.add_argument("-quiet", dest="quiet", action="store_true",
                   help="suppress algorithm progress messages on stdout/stderr")
    p.add_argument("infile", help="input file (see --help for per-metric format)")
    p.add_argument("outfile", help="output file: 'id1 id2 dist' per line")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    # When force_auto is set, blank the manual knobs so the wrapper selects
    # auto mode. Otherwise pass whatever the user gave; the wrapper detects
    # auto vs manual from whether any of the three is set.
    if args.force_auto:
        ham_dist = num_blocks = num_chunks = None
    else:
        ham_dist   = args.hamdist
        num_blocks = args.numblocks
        num_chunks = args.numchunks

    try:
        if args.metric == "cosine":
            run_from_file(
                input_path=args.infile,
                output_path=args.outfile,
                cos_dist=args.cosdist,
                ham_dist=ham_dist,
                num_blocks=num_blocks,
                num_chunks=num_chunks,
                missing_ratio=args.missingratio,
                centering=args.centering,
                seed=args.seed,
                verbose=not args.quiet,
            )
        elif args.metric == "minmax":
            run_from_file_minmax(
                input_path=args.infile,
                output_path=args.outfile,
                minmax_dist=args.minmaxdist,
                ham_dist=ham_dist,
                num_blocks=num_blocks,
                num_chunks=num_chunks,
                missing_ratio=args.missingratio,
                z_normalization=args.znorm,
                minmax_normalization=args.mmnorm,
                seed=args.seed,
                verbose=not args.quiet,
            )
        else:  # jaccard
            run_from_file_jaccard(
                input_path=args.infile,
                output_path=args.outfile,
                jaccard_dist=args.jaccarddist,
                ham_dist=ham_dist,
                num_blocks=num_blocks,
                num_chunks=num_chunks,
                missing_ratio=args.missingratio,
                seed=args.seed,
                verbose=not args.quiet,
            )
    except (RuntimeError, OSError, ValueError, TypeError, OverflowError) as e:
        print(f"sketchsort: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
